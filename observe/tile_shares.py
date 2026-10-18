"""Attention share of each block, from the tile records that the compiler pass or the audit kernel
(``runtime.audit_kernel``) writes. Both write the same layout.

On an audit step the recording kernel stores, for every layer, request row, block, and query row
of the sampled KV group, two values of the tile that holds the block: the largest score ``m`` and
the sum ``l`` of ``exp(score - m)`` over the tile's keys, in the units of the kernel's own softmax.
They are the two values the kernel's softmax already holds, so the kernel computes nothing for the
statistic. The table has two planes, ``[2][layer][row][block][query_row]``: plane 0 holds ``m`` and
plane 1 holds ``l``. ``read_tile_lse`` turns them into the log of the tile's total attention weight,
``u = m + log(l)``, split evenly over the tile's blocks. A cell that no launch wrote is NaN, and a
tile with no keys has ``l = 0``, so ``u = -inf``.

Normalizing ``u`` over the blocks of one (layer, row, query row) gives the share of that query's
attention that falls on each block. The shares of every layer and query row of a request average
to the request's share per block. Because the shares come from log-sum-exp values, they do not
depend on the running maximum, on how the kernel split its tiles, or on the head weighting.
"""

from typing import Literal

import numpy as np

from observe.zero_touch import pub_path, read_json_lines


def share_descriptor(pid: int) -> dict[str, int | str]:
    """The statistic block of process ``pid``, with the tile size of the kernel that records into it.

    The pass writes the block's addresses when it allocates them and the tile size when it
    instruments a kernel, so the two come from different files. Raises ``RuntimeError`` until a
    kernel that records the statistic has been compiled.

    An engine can have several attention kernels that record into the one block, each with its own
    tile size: SGLang's extend kernel has a tile of 128 positions and its decode kernel one of 32. The
    statistic is recorded on audit steps only, and an audit step decodes, so the reader takes the
    narrowest tile, which is the decode kernel's. Engines with one kernel have one candidate.
    """
    descriptors = pub_path(pid)
    coverage = {
        int(item.get("host_va", 0) or 0): item
        for item in read_json_lines(descriptors.replace("ekv_pass_", "ekv_coverage_"))
    }
    candidates = []
    for entry in read_json_lines(descriptors):
        if not int(entry.get("tile_lse_dev", 0)):
            continue
        tile_blocks = int(coverage.get(int(entry["host_va"]), {}).get("attn_share_tile_blocks", 0))
        if tile_blocks:
            candidates.append({**entry, "attn_share_tile_blocks": tile_blocks})
    if not candidates:
        raise RuntimeError(f"process {pid} has not compiled a kernel that records the attention statistic")
    return min(candidates, key=lambda entry: int(entry["attn_share_tile_blocks"]))


Aggregate = Literal["mean", "max"]


def normalize_tile_lse(
    lse: np.ndarray, row_blocks: np.ndarray, tile_blocks: int, aggregate: Aggregate = "mean"
) -> np.ndarray:
    """Return the attention share per (row, block) from ``lse[layer, row, block, query_row]``.

    The kernel records one value per tile of ``tile_blocks`` blocks, and ``read_tile_lse`` gives
    every block of the tile that value minus the log of ``tile_blocks``, so the attention weight of
    a tile is split evenly over its blocks. A row of ``row_blocks[row]`` blocks usually ends in a
    partial tile, whose weight is then split over blocks that do not exist. This function gives that weight back
    to the blocks that do, and drops the cells beyond the row.

    Each (layer, row, query_row) with at least one finite value contributes a share vector that
    sums to 1. With ``aggregate="mean"`` the result averages those vectors over layers and query
    rows. With ``"max"`` it takes, for each block, the largest share that any layer and query row
    gives it, then scales the row to sum to 1: a block that one attention head needs scores high even
    if the average head ignores it. Each row of the result sums to 1. A row with no finite value
    anywhere is all NaN: there is no evidence about it.
    """
    if aggregate not in ("mean", "max"):
        raise ValueError(f"unknown aggregate {aggregate!r}")
    if lse.ndim != 4:
        raise ValueError("lse must be [layers, rows, blocks, query_rows]")
    if row_blocks.shape != (lse.shape[1],) or tile_blocks < 1:
        raise ValueError("row_blocks must give one length per row, and tile_blocks must be positive")
    blocks = np.arange(lse.shape[2])[None, :]
    length = row_blocks[:, None]
    tail_start = (length // tile_blocks) * tile_blocks
    in_tail = (length % tile_blocks != 0) & (blocks >= tail_start) & (blocks < length)
    valid_in_tail = np.maximum(length - tail_start, 1)
    regained = np.where(in_tail, np.log(tile_blocks / valid_in_tail), 0.0)
    lse = np.where(blocks[None, :, :, None] < row_blocks[None, :, None, None], lse, np.nan)
    lse = lse + regained[None, :, :, None]
    finite = np.isfinite(lse)
    peak = np.where(finite, lse, -np.inf).max(axis=2, keepdims=True)
    has_data = np.isfinite(peak)
    weight = np.where(finite, np.exp(lse - np.where(has_data, peak, 0.0)), 0.0)
    total = weight.sum(axis=2, keepdims=True)
    share = np.divide(weight, total, out=np.zeros_like(weight), where=total > 0)
    contributors = has_data[:, :, 0, :].sum(axis=(0, 2))
    if aggregate == "max":
        peak_share = share.max(axis=(0, 3))
        total = peak_share.sum(axis=1, keepdims=True)
        combined = np.divide(peak_share, total, out=np.zeros_like(peak_share), where=total > 0)
    else:
        combined = share.sum(axis=(0, 3)) / np.maximum(contributors, 1)[:, None]
    return np.where(contributors[:, None] > 0, combined, np.nan)


def read_tile_lse(
    cuda: object,
    base: int,
    table_layers: int,
    layers: int,
    rows: int,
    width: int,
    group_rows: int,
    live_rows: int,
    tile_blocks: int,
) -> np.ndarray:
    """Return ``u[layers, live_rows, width, group_rows]`` for the first ``live_rows`` rows of the first ``layers`` layers.

    ``base`` is the device address of the table: two planes of ``table_layers`` layers, each laid
    out ``[layer][row][block][query_row]`` for ``rows`` rows in all. ``u = m + log(l) - log(tile_blocks)``
    is the log attention weight of each block, with the tile's weight split evenly over its
    ``tile_blocks`` blocks. ``cuda`` provides ``d2h_many([(address, nbytes), ...])``.
    """
    return read_tile_rows(
        cuda, base, table_layers, layers, rows, width, group_rows, np.arange(live_rows), tile_blocks
    )


def read_tile_rows(
    cuda: object,
    base: int,
    table_layers: int,
    layers: int,
    rows: int,
    width: int,
    group_rows: int,
    row_ids: np.ndarray,
    tile_blocks: int,
) -> np.ndarray:
    """Return ``u[layers, len(row_ids), width, group_rows]`` for the rows ``row_ids`` (ascending), as
    ``read_tile_lse`` does for the first rows. It copies only those rows, one copy per run of adjacent rows in
    each layer of each plane, so the cost follows the rows asked for, not the rows that are live."""
    if np.any(np.diff(row_ids) <= 0):
        raise ValueError("row_ids must be ascending and distinct")
    layer_bytes = rows * width * group_rows * 4
    plane_bytes = table_layers * layer_bytes
    row_bytes = width * group_rows * 4
    breaks = np.flatnonzero(np.diff(row_ids) != 1) + 1
    runs = [(int(run[0]), len(run)) for run in np.split(row_ids, breaks) if len(run)]
    requests = [
        (base + plane * plane_bytes + layer * layer_bytes + first * row_bytes, count * row_bytes)
        for plane in (0, 1)
        for layer in range(layers)
        for first, count in runs
    ]
    blobs = cuda.d2h_many(requests)  # ty: ignore[unresolved-attribute]
    per_layer = len(runs)
    m, total = (
        np.stack(
            [
                np.frombuffer(
                    b"".join(
                        blobs[(plane * layers + layer) * per_layer : (plane * layers + layer + 1) * per_layer]
                    ),
                    dtype=np.float32,
                ).reshape(len(row_ids), width, group_rows)
                for layer in range(layers)
            ]
        )
        for plane in (0, 1)
    )
    with np.errstate(divide="ignore"):  # a tile with no keys has l = 0: its weight is exp(-inf) = 0
        return m + np.log(total) - np.float32(np.log(tile_blocks))
