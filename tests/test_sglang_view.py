"""The compaction of SGLang's key-value index list, against an explicit reference, on the CPU.

The compaction is two Triton kernels. Triton's interpreter runs them on the CPU, but only in a process that sets
``TRITON_INTERPRET`` before it imports Triton, so the test runs this file as a script in such a process, and the
script checks every case."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np

BLOCK = 16
WIDTH = 20


def reference(
    indptr: list[int], slots: list[int], keep: np.ndarray, lens: list[int], whole: np.ndarray
) -> list[list[int]]:
    """The slots each request reads: its kept blocks in order, the first and last block always kept."""
    out = []
    for row, length in enumerate(lens):
        blocks = -(-length // BLOCK)
        kept = [b for b in range(blocks) if keep[row, b] or whole[row] or b in (0, blocks - 1)]
        positions = [p for b in kept for p in range(b * BLOCK, (b + 1) * BLOCK) if p < length]
        out.append([slots[indptr[row] + p] for p in positions])
    return out


def run(
    lens: list[int],
    keep: np.ndarray,
    padding: int = 0,
    audited: np.ndarray | None = None,
    active: bool = True,
    seed: int = 0,
) -> list[int]:
    """Compact one batch, check the list against the reference, and return the read counters."""
    import torch

    from runtime.sglang_compact import SlotCompactor

    rng = np.random.default_rng(seed)
    full_lens = lens + [1] * padding  # a CUDA graph fills the batch with one-token rows
    indptr = [0, *np.cumsum(full_lens).tolist()]
    slots = rng.permutation(10_000)[: indptr[-1]].tolist()
    whole = np.zeros(len(lens), dtype=bool) if audited is None else audited[: len(lens)].copy()
    whole |= not active
    device_indptr = torch.tensor(indptr, dtype=torch.int32)
    device_slots = torch.tensor(
        slots + [-1] * 7, dtype=torch.int64
    )  # the engine's buffer is longer than the list
    compactor = SlotCompactor(WIDTH, BLOCK, torch.device("cpu"))
    audit = None if audited is None else torch.from_numpy(audited.astype(np.uint8))
    new_lens = compactor.compact(
        device_indptr,
        device_slots,
        torch.from_numpy(keep),
        audit,
        active,
        len(lens),
        len(full_lens),
        indptr[-1],
    )
    expected = reference(indptr, slots, keep, lens, whole) + [
        [slots[indptr[len(lens) + i]]] for i in range(padding)
    ]
    assert device_indptr.tolist() == [0, *np.cumsum([len(r) for r in expected]).tolist()]
    assert new_lens.tolist() == [len(r) for r in expected]
    for row, want in enumerate(expected):
        got = device_slots[device_indptr[row] : device_indptr[row + 1]].tolist()
        assert got == want, f"row {row}"
    assert device_slots[indptr[-1] :].tolist() == [-1] * 7  # nothing after the list is written
    return compactor.counts.tolist()


def check_all() -> None:
    lens = [5, 16, 17, 100, 33]
    blocks = sum(-(-n // BLOCK) for n in lens)
    counts = run(lens, np.ones((len(lens), WIDTH), dtype=bool))  # keep-all is the identity
    assert counts == [blocks, blocks, 0, blocks, 0], counts

    lens = [200, 64, 150, 17, 1]
    counts = run(lens, (np.arange(WIDTH)[None, :] % 2 == 0).repeat(len(lens), axis=0), seed=1)  # half
    assert counts[0] < counts[1], counts

    lens = [200, 150, 90]
    keep = np.zeros((len(lens), WIDTH), dtype=bool)
    counts = run(lens, keep, audited=np.array([False, True, False]), seed=2)
    assert counts[2] == -(-150 // BLOCK), counts  # the audited row reads, and counts, all its blocks
    counts = run(lens, keep, active=False, seed=3)
    assert counts[0] == counts[1] == counts[4], counts  # no live policy: all read, counted as expired

    run([35], np.zeros((1, WIDTH), dtype=bool))  # the last, partial block is always kept

    rng = np.random.default_rng(7)
    for seed in range(20):
        rows = int(rng.integers(1, 9))
        keep = rng.random((rows, WIDTH)) < rng.random()
        audited = rng.random(rows) < 0.3 if seed % 2 else None
        run(rng.integers(1, BLOCK * WIDTH, rows).tolist(), keep, int(rng.integers(0, 4)), audited, seed=seed)


def test_compaction_in_the_triton_interpreter() -> None:
    root = Path(__file__).resolve().parents[1]
    env = os.environ | {"TRITON_INTERPRET": "1", "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(root)}
    done = subprocess.run([sys.executable, __file__], env=env, capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]


if __name__ == "__main__":
    check_all()
