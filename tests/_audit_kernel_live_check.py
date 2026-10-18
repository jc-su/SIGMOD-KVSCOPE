"""Gate for the audit kernel: does it record the true attention share of each block?

``runtime.audit_kernel`` records the statistic with a Triton kernel of its own, for backends whose
attention kernels the compiler pass cannot instrument. This check applies the gate of
``_attn_share_live_check.py`` to it, on the same inputs and with the same limits: real queries and
keys of Llama-3.1-8B-Instruct at every layer (six texts of 512 to 6,144 tokens) and two synthetic
controls. The kernel's records go through ``observe.tile_shares.read_tile_lse`` and ``normalize_tile_lse``,
the code the controller uses, and are compared with a float64 reference:

  total variation distance to the true share below TV_LIMIT, per tile
  the observed top 25% of tiles holds as much true share as the true top 25%

It also checks that rows the step did not select, and a step with the flag off, leave the block unwritten.
"""

import ctypes
import os
import statistics
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _attn_share_live_check import (  # ty: ignore[unresolved-import]  # sibling module
    BLOCK,
    KV_HEADS,
    LAYERS,
    TV_LIMIT,
    Copy,
    Sequence,
    jaccard,
    kept_ratio,
    real_sequences,
    reference,
    synthetic_sequences,
    tiles,
    tv,
)

from observe.tile_shares import normalize_tile_lse, read_tile_lse
from runtime.audit_kernel import TILE_BLOCKS, RecordingPlane


def record(
    plane: RecordingPlane, seqs: list[Sequence], layer: int, select: list[bool], flag: bool
) -> np.ndarray:
    """One decode step as ``layer``: lay the requests out in a paged cache, record, and return the layer's cells
    ``u[row, block, query_row]`` as the controller reads them."""
    b = len(seqs)
    hq, d = seqs[0].q.shape
    hkv = seqs[0].k.shape[1]
    lens = [s.k.shape[0] for s in seqs]
    nblk = [-(-n // BLOCK) for n in lens]
    perm = torch.randperm(sum(nblk), device="cuda") + 1  # block 0 stays unused
    keys = torch.zeros(sum(nblk) + 1, BLOCK, hkv, d, dtype=torch.float16, device="cuda")
    table = torch.zeros(plane.rows, plane.width, dtype=torch.int32, device="cuda")
    cursor = 0
    for i, s in enumerate(seqs):
        ids = perm[cursor : cursor + nblk[i]]
        cursor += nblk[i]
        table[i, : nblk[i]] = ids.int()
        pad = nblk[i] * BLOCK - lens[i]
        keys[ids] = torch.nn.functional.pad(s.k, (0, 0, 0, 0, 0, pad)).reshape(nblk[i], BLOCK, hkv, d)
    query = torch.zeros(plane.rows, hq, d, dtype=torch.float16, device="cuda")
    query[:b] = torch.stack([s.q for s in seqs])
    lengths = torch.zeros(plane.rows, dtype=torch.int32, device="cuda")
    lengths[:b] = torch.tensor(lens, dtype=torch.int32)
    plane.select(np.array(select + [False] * (plane.rows - b)), np.arange(plane.rows))
    layer_cells = plane.rows * plane.width * plane.group_rows
    torch.cuda.synchronize()
    plane_bytes = plane._layers * layer_cells * 4
    for offset in (0, plane_bytes):  # NaN in both planes: nothing written yet
        plane._cuda.cudaMemset(
            ctypes.c_void_p(plane._lse + offset + layer * layer_cells * 4),
            0xFF,
            ctypes.c_size_t(layer_cells * 4),
        )
    plane._copy_to_device(plane._flag_dev, np.array([1 if flag else 0], dtype=np.uint32))
    plane.launch(layer, table, lengths, query, keys, d**-0.5)
    base = plane._lse + layer * layer_cells * 4
    return read_tile_lse(
        Copy(), base, plane._layers, 1, plane.rows, plane.width, plane.group_rows, plane.rows, TILE_BLOCKS
    )[0]


def run_case(plane: RecordingPlane, name: str, launches: list[tuple[int, list[Sequence]]]) -> None:
    stats: dict[str, list[float]] = {k: [] for k in ("tv", "jac", "kept", "recency", "random")}
    rng = np.random.default_rng(0)
    for layer, seqs in launches:
        b = len(seqs)
        nblk = np.array([-(-s.k.shape[0] // BLOCK) for s in seqs])
        cells = record(plane, seqs, layer, [True] * b, True)
        shares = normalize_tile_lse(cells[None, :b], nblk, TILE_BLOCKS)
        for row, s in enumerate(seqs):
            t = reference(s, layer % KV_HEADS)
            assert not np.isnan(shares[row, : len(t)]).any(), f"{name}: unwritten cells in row {row}"
            obs = np.nan_to_num(shares[row, : len(t)], nan=0.0)
            obs, t = tiles(obs, TILE_BLOCKS), tiles(t, TILE_BLOCKS)
            stats["tv"].append(tv(obs, t))
            stats["jac"].append(jaccard(obs, t))
            stats["kept"].append(kept_ratio(t, obs))
            stats["recency"].append(kept_ratio(t, np.arange(len(t), dtype=np.float64)))
            stats["random"].append(kept_ratio(t, rng.random(len(t))))

    def cell(key: str) -> str:
        v = stats[key]
        return f"{statistics.median(v):.3f}/{min(v) if key in ('jac', 'kept') else max(v):.3f}"

    print(
        f"  {name:<14} n={len(stats['kept']):<4} TV {cell('tv'):<13} jac25 {cell('jac'):<13}"
        f" kept {cell('kept'):<13} recency {statistics.median(stats['recency']):.3f}"
        f" random {statistics.median(stats['random']):.3f}",
        flush=True,
    )
    assert max(stats["tv"]) < TV_LIMIT, (
        f"{name}: total variation distance {max(stats['tv']):.2e} over the limit"
    )
    assert min(stats["kept"]) > 1.0 - 1e-9, f"{name}: the observed top 25% is not the true top 25%"


def main() -> None:
    assert torch.cuda.is_available()
    real = real_sequences()
    controls = {"flat": synthetic_sequences("flat"), "ramp": synthetic_sequences("ramp")}
    widest = max(s.k.shape[0] for seqs in real.values() for s in seqs)
    rows = max(len(seqs) for seqs in real.values())
    plane = RecordingPlane(
        layers=len(LAYERS), rows=rows, width=-(-widest // BLOCK) + TILE_BLOCKS, group_rows=4, block_size=BLOCK
    )
    print("cells are median/worst over launches x requests (worst = max for TV, min for jac and kept)")
    run_case(plane, "llama-8b real", [(layer, real[layer]) for layer in LAYERS])
    for name, seqs in controls.items():
        run_case(plane, name, [(0, seqs)])

    seqs = real[0][:3]
    off = record(plane, seqs, 0, [True, True, True], flag=False)
    assert np.isnan(off).all(), "a step with the flag off wrote the statistic"
    part = record(plane, seqs, 0, [True, False, True], flag=True)
    assert np.isnan(part[1]).all() and not np.isnan(part[0, 0]).all(), (
        "an unselected row was recorded, or a selected one was not"
    )
    print("PASS: the audit kernel records the true attention share; the flag and the row selection gate it")


if __name__ == "__main__":
    main()
