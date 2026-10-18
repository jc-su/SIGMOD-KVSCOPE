"""The fuser — resolves each eBPF copy to a KV block and emits a costed ``KVEdge``.

    Copy(addr, bytes, direction, cost)                    [eBPF: physical cost]
          │  BlockAddressMap.resolve(addr) -> block, content_hash
          ▼
    KVEdge(block, hash, src→dst tier, bytes, cost, request)

When ``kv_uses`` (the request↔block tie) is supplied, each edge also carries the
``request`` whose use of the block is nearest in time. ``content_hash`` is the
cross-rank key (addresses differ per TP rank); time proximity disambiguates repeated
hashes. A copy that resolves to no block is dropped — the address-range match
self-selects KV traffic.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace

from .blockmap import Block, BlockAddressMap
from .events import Copy, KVEdge, RequestKVUse
from .types import Nanoseconds, Op, SizeBytes, Tier, dir_tiers

_DEFAULT_WINDOW = Nanoseconds(5_000_000)
# directions with no declared src/dst — resolve by which side is a KV block:
# a driver BATCH element, or cudaMemcpyDefault (the reader carries both addresses).
_AMBIGUOUS = frozenset(("BATCH", "DEFAULT"))


def _resolve(c: Copy, m: BlockAddressMap) -> tuple[Block, Tier | None, Tier | None] | None:
    """Resolve a copy to (block, src_tier, dst_tier). For a declared direction the
    device side is fixed; for an ambiguous copy (BATCH / DEFAULT) try the device src
    then the dst — whichever lands in a KV block decides the tiers (device src ⇒
    offload HBM→DRAM, device dst ⇒ load DRAM→HBM). This infers direction from block
    membership; it does not guess."""
    if c.direction not in _AMBIGUOUS:
        b = m.resolve(c.addr)
        if b is None:
            return None
        src, dst = dir_tiers(c.direction)
        return b, src, dst
    src = m.resolve(c.addr)                      # ambiguous: addr is the candidate src
    dst = m.resolve(c.other_addr) if c.other_addr is not None else None
    if src is not None and dst is not None:
        # Both endpoints are known device KV blocks. Match the declared D2D convention
        # (the device-side address is the destination block) instead of mislabeling the
        # source as an HBM->DRAM offload.
        return dst, Tier.HBM, Tier.HBM
    if src is not None:
        return src, Tier.HBM, Tier.DRAM
    if dst is not None:
        return dst, Tier.DRAM, Tier.HBM
    return None


def _note_drop(c: Copy, m: BlockAddressMap, diag: dict) -> None:
    """Account a copy the map could not resolve, split by WHY — the distinction that decides
    whether a coverage hole is fixable:

      * ``beyond_extent`` — the address is on this pool's own lattice but past the map's
        extent. Real KV traffic we could attribute and did not: the extent is too small.
      * ``unresolved``    — not this pool's traffic. Correctly ignored.

    Both used to be one silent ``continue``, so a truncated extent was indistinguishable from
    an idle pool. ``max_unit`` carries the highest lattice unit seen so the caller can widen."""
    ext = m.extent
    top = -1
    for a in (c.addr, c.other_addr):
        u = m.lattice_unit(a)
        if u is not None and ext is not None and u >= ext:
            top = max(top, u)
    if top >= 0:
        diag["beyond_extent"] = diag.get("beyond_extent", 0) + 1
        diag["max_unit"] = max(diag.get("max_unit", -1), top)
    else:
        diag["unresolved"] = diag.get("unresolved", 0) + 1


def _bid(block_id) -> int | None:
    """The physical block-pool index from a KVEdge block_id ((layer, bid) or bid)."""
    if isinstance(block_id, tuple):
        return block_id[1] if len(block_id) > 1 else block_id[0]
    return block_id if isinstance(block_id, int) else None


def _nearest_request(uses: list, at: Nanoseconds, window: Nanoseconds):
    """The request whose KV use of this block is closest in time to the copy — the
    sound resolution of the many-to-many (a shared block has several users; the
    eviction/offload is charged to the temporally-closest one)."""
    best, best_d = None, window
    for request_id, _role, t in uses:
        d = abs(t - at)
        if d <= best_d:
            best_d, best = d, request_id
    return best


def fuse_copies(copies: Iterable[Copy], addr_map: BlockAddressMap, *,
                step: int | None = None, kv_uses: Iterable[RequestKVUse] | None = None,
                time_window: Nanoseconds = _DEFAULT_WINDOW,
                diag: dict | None = None) -> list[KVEdge]:
    """Resolve each copy to a KV block and emit a costed σ edge. When `kv_uses` (the
    request↔block tie) is supplied, attach the `request` whose use of the block is
    nearest in time — sound per-request attribution that content_hash alone cannot give.

    Pass `diag` (a dict) to have dropped copies ACCOUNTED rather than merely skipped:
    `beyond_extent` / `unresolved` / `max_unit` (see `_note_drop`). Optional so no caller
    changes; a live source should always pass it, because an unaccounted drop is how a
    truncated address map masquerades as a quiet pool."""
    by_block_req: dict[int, list] = {}
    for u in (kv_uses or ()):
        for b in u.block_ids:
            by_block_req.setdefault(b, []).append((u.request_id, u.role, u.at))

    edges: list[KVEdge] = []
    for c in copies:
        res = _resolve(c, addr_map)
        if res is None:
            if diag is not None:
                _note_drop(c, addr_map, diag)
            continue
        blk, src_tier, dst_tier = res
        h = blk.content_hash if blk.content_hash is not None else addr_map.hash_of(blk.block_id)
        note = "ebpf+blockmap"
        request = None
        if by_block_req:
            bid = _bid(blk.block_id)
            request = _nearest_request(by_block_req.get(bid, []), c.enqueued_at, time_window)
            if request is not None:
                note += "+reqbridge"
        edges.append(KVEdge(
            op=Op.COPY, block_id=blk.block_id, content_hash=h, prefix_hash=None,
            src_tier=src_tier, dst_tier=dst_tier, nbytes=c.nbytes, cost=c.cost,
            step=step, request=request, note=note))
    return edges


@dataclass(slots=True)
class _Shards:
    """Accumulates per-TP-rank shards of one logical block during de-dup."""
    first: KVEdge
    nbytes: int
    cost: Nanoseconds | None
    count: int

    def absorb(self, e: KVEdge) -> None:
        self.nbytes += e.nbytes or 0
        if e.cost is not None:
            self.cost = e.cost if self.cost is None else max(self.cost, e.cost)
        self.count += 1

    def to_edge(self) -> KVEdge:
        return replace(self.first, nbytes=SizeBytes(self.nbytes), cost=self.cost,
                       note=f"{self.first.note} shards={self.count}")


def dedup_by_hash(edges: Iterable[KVEdge]) -> list[KVEdge]:
    """Collapse per-TP-rank shards of the SAME logical KV block into one edge.

    Under tensor parallelism vLLM broadcasts ONE block table to every rank, so a logical
    block has the SAME physical block index on all ranks — only its address differs per
    rank. eBPF sees one copy per rank; grouping COPY edges by (op, block_id, src_tier,
    dst_tier) collapses them with no engine-internal read — ``block_id`` is the
    cross-rank-stable key (``content_hash``, when present, is an equivalent fallback for
    an edge that carries it but no block_id). Bytes are summed (true aggregate traffic),
    cost is the max (the move is done when the slowest rank finishes), and ``shards``
    records the rank count. Edges with nothing to key on pass through unchanged.

    COMPUTE edges (no block_id) are collapsed by (op, request, step) instead: under TP
    each rank computes its head-shard of the SAME request's attention, so the ranks
    produce one COMPUTE edge each — keyed on the request they share, with cost = max (the
    ranks run in parallel, so the request's GPU time is the slowest rank, not their sum)."""
    groups: dict[tuple, _Shards] = {}
    order: list[tuple] = []
    passthrough: list[KVEdge] = []
    for e in edges:
        if e.block_id is not None:
            key = (e.op, e.block_id, e.src_tier, e.dst_tier)
        elif e.content_hash is not None:            # carries content but no block_id
            key = (e.op, e.content_hash, e.src_tier, e.dst_tier)
        elif e.request is not None:                 # cross-rank COMPUTE shards
            key = (e.op, "request", e.request, e.step)
        else:
            passthrough.append(e)
            continue
        g = groups.get(key)
        if g is None:
            groups[key] = _Shards(e, e.nbytes or 0, e.cost, 1)
            order.append(key)
        else:
            g.absorb(e)
    return [groups[k].to_edge() for k in order] + passthrough

