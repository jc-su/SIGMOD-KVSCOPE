"""The compaction of SGLang's key-value index list on the GPU, in two Triton kernels and no host synchronization.

SGLang's decode kernel reads request ``r`` from ``slots[indptr[r]:indptr[r+1]]``, one token slot per entry. The view
groups ``BLOCK_TOKENS`` consecutive entries of a request into a block and keeps some blocks. ``plan_kernel`` runs as
one program: for each row it chooses the blocks the row reads, writes their order to a table, and writes the new
length and the new ``indptr``; it also adds the row's blocks to the read counters. ``gather_kernel`` then writes each
entry of the new list, in parallel, from the entry of the old list that it comes from. The host launches both
kernels and two copies back into the engine's buffers, and never waits for the device.

A row keeps its first block and its last block, which holds its newest token and may be partial, always. A row reads
all its blocks when the policy is not live (``active`` is 0), when it is audited in this step, and when it is a row
that a CUDA graph adds to fill its batch (rows from ``n_real`` on).
"""

import torch
import triton
import triton.language as tl

COUNTS = 5  # blocks read, blocks a full read would read, read in audits, read untrimmed, read without a live policy


@triton.jit
def plan_kernel(
    indptr,
    keep,
    audit,
    has_audit,
    active,
    n_real,
    n_rows,
    new_indptr,
    lens,
    table,
    counts,
    keep_stride,
    table_stride,
    BLOCK_TOKENS: tl.constexpr,
    WIDTH: tl.constexpr,
):
    blocks_at = tl.arange(0, WIDTH)
    cursor = tl.full([], 0, tl.int32)
    tl.store(new_indptr, cursor)
    read = tl.full([], 0, tl.int64)
    full = tl.full([], 0, tl.int64)
    audit_read = tl.full([], 0, tl.int64)
    untrimmed = tl.full([], 0, tl.int64)
    for row in range(n_rows):
        length = tl.load(indptr + row + 1) - tl.load(indptr + row)
        blocks = (length + BLOCK_TOKENS - 1) // BLOCK_TOKENS
        real = row < n_real
        audited = tl.load(audit + row, mask=(has_audit != 0) & real, other=0) != 0
        whole = (active == 0) | audited | (row >= n_real)
        inside = blocks_at < blocks
        chosen = tl.load(keep + row * keep_stride + blocks_at, mask=inside & real, other=0) != 0
        chosen = (chosen | whole | (blocks_at == 0) | (blocks_at == blocks - 1)) & inside
        rank = tl.cumsum(chosen.to(tl.int32), 0) - 1
        tl.store(table + row * table_stride + rank, blocks_at, mask=chosen)
        kept = tl.sum(chosen.to(tl.int32), 0)
        new_length = tl.where(blocks > 0, (kept - 1) * BLOCK_TOKENS + length - (blocks - 1) * BLOCK_TOKENS, 0)
        tl.store(lens + row, new_length)
        cursor += new_length
        tl.store(new_indptr + row + 1, cursor)
        real_rows = tl.where(real, 1, 0).to(tl.int64)
        counted = real_rows * kept.to(tl.int64)
        read += counted
        full += real_rows * blocks.to(tl.int64)
        audit_read += counted * audited.to(tl.int64)
        untrimmed += counted * ((new_length == length) & (audited == 0) & (active != 0)).to(tl.int64)
    tl.store(counts, tl.load(counts) + read)
    tl.store(counts + 1, tl.load(counts + 1) + full)
    tl.store(counts + 2, tl.load(counts + 2) + audit_read)
    tl.store(counts + 3, tl.load(counts + 3) + untrimmed)
    tl.store(counts + 4, tl.load(counts + 4) + read * (1 - active))  # active is 0 or 1


@triton.jit
def gather_kernel(
    indptr,
    new_indptr,
    table,
    slots,
    out,
    n_rows,
    table_stride,
    BLOCK_TOKENS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    position = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = position < tl.load(new_indptr + n_rows)
    row = tl.zeros([BLOCK], tl.int32)
    for r in range(1, n_rows):
        row += (position >= tl.load(new_indptr + r)).to(tl.int32)
    offset = position - tl.load(new_indptr + row, mask=valid, other=0)
    block = tl.load(table + row * table_stride + offset // BLOCK_TOKENS, mask=valid, other=0)
    source = tl.load(indptr + row, mask=valid, other=0) + block * BLOCK_TOKENS + offset % BLOCK_TOKENS
    tl.store(out + position, tl.load(slots + source, mask=valid, other=0), mask=valid)


class SlotCompactor:
    """The buffers of the compaction, kept from step to step. They grow when a batch needs more and never shrink."""

    GATHER_BLOCK = 1024

    def __init__(self, width: int, block_tokens: int, device: torch.device) -> None:
        self.width, self.block_tokens, self.device = width, block_tokens, device
        self.counts = torch.zeros(COUNTS, dtype=torch.int64, device=device)
        self.no_audit = torch.zeros(1, dtype=torch.uint8, device=device)
        self._rows = 0
        self.new_indptr = self.lens = self.table = torch.empty(0, dtype=torch.int32, device=device)
        self._out: torch.Tensor = torch.empty(0, dtype=torch.int64, device=device)

    def _reserve(self, rows: int, entries: int, dtype: torch.dtype) -> None:
        if rows > self._rows:
            self._rows = rows
            self.new_indptr = torch.empty(rows + 1, dtype=torch.int32, device=self.device)
            self.lens = torch.empty(rows, dtype=torch.int32, device=self.device)
            self.table = torch.empty(rows, self.width, dtype=torch.int32, device=self.device)
        if entries > self._out.shape[0] or dtype != self._out.dtype:
            self._out = torch.empty(max(entries, self._out.shape[0]), dtype=dtype, device=self.device)

    def compact(
        self,
        indptr: torch.Tensor,
        slots: torch.Tensor,
        keep: torch.Tensor,
        audit: torch.Tensor | None,
        active: bool,
        n_real: int,
        n_rows: int,
        entries: int,
    ) -> torch.Tensor:
        """Rewrite ``indptr[:n_rows + 1]`` and ``slots[:entries]`` in place to the view, and return the new lengths.

        ``keep`` is the bool ``[rows, width]`` policy of the view, ``audit`` the uint8 rows audited in this step (or
        ``None``), and ``entries`` the length of the engine's list, ``indptr[n_rows]``, which the host knows."""
        self._reserve(n_rows, entries, slots.dtype)
        plan_kernel[(1,)](
            indptr,
            keep,
            self.no_audit if audit is None else audit,
            int(audit is not None),
            int(active),
            n_real,
            n_rows,
            self.new_indptr,
            self.lens,
            self.table,
            self.counts,
            keep.stride(0),
            self.table.stride(0),
            BLOCK_TOKENS=self.block_tokens,  # ty: ignore[invalid-argument-type]  # Triton takes an int for a constexpr parameter
            WIDTH=triton.next_power_of_2(self.width),
        )
        gather_kernel[(triton.cdiv(entries, self.GATHER_BLOCK),)](
            indptr,
            self.new_indptr,
            self.table,
            slots,
            self._out,
            n_rows,
            self.table.stride(0),
            BLOCK_TOKENS=self.block_tokens,  # ty: ignore[invalid-argument-type]  # Triton takes an int for a constexpr parameter
            BLOCK=self.GATHER_BLOCK,  # ty: ignore[invalid-argument-type]  # Triton takes an int for a constexpr parameter
        )
        indptr[: n_rows + 1].copy_(self.new_indptr[: n_rows + 1])
        slots[:entries].copy_(self._out[:entries])
        return self.lens[:n_rows]
