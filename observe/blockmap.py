"""The address bridge — resolves an eBPF-observed physical address to the KV block
that occupies it, and that block to its content hash.

eBPF copies are keyed by physical ADDRESS; the rest of the system reasons about
BLOCKS, and across TP ranks about CONTENT HASH (the same logical block has a
different address per rank). This module is that connection:

    eBPF Address --resolve()--> BlockId --hash_of()--> ContentHash

``from_vllm`` builds it in-process (KV-tensor addresses + block-pool hashes) — the
oracle/convenience path. The engine-agnostic, zero-touch path needs no in-process read:
``from_alloc(base, num_blocks, stride)`` builds address->block from a KV-pool base+stride
(captured by eBPF from the pool's cudaMalloc), and ``content_hashes_from_tokens`` derives
content identity from the tokens the network face already captures. Pure-data ``add`` is
used by tests.
"""
from __future__ import annotations

import bisect
import zlib
from dataclasses import dataclass

from .types import Address, BlockId, ContentHash


@dataclass(frozen=True, slots=True)
class Block:
    """A physical KV block and the address range it occupies."""
    block_id: BlockId
    lo: Address                # base address (inclusive)
    hi: Address                # lo + nbytes (exclusive)
    content_hash: ContentHash | None


class BlockAddressMap:
    """A sorted, range-queryable Address -> Block map with O(log n) ``resolve``.

    Addresses come from the KV-cache tensors; content hashes from the block pool.
    The two are joined here so an anonymous device address from eBPF resolves to the
    content hash KV Events speak in."""

    def __init__(self) -> None:
        self._los: list[int] = []
        self._blocks: list[Block] = []
        self._id2hash: dict[BlockId, ContentHash] = {}
        # The uniform LATTICE this map was built on, when it has one: (base, stride, n_units).
        # A "unit" is a physical KV unit at the engine.s configured KV block size -- a 16-token block on vLLM's
        # paged pool, a single token slot on SGLang's token pool -- so the lattice is block-size-
        # neutral (GENERALIZATION.md S2). Set by from_alloc; None for tensor-built maps.
        self._lattice: tuple[int, int, int] | None = None

    def add(self, block_id: BlockId, addr: int, nbytes: int,
            content_hash: ContentHash | None = None) -> None:
        if nbytes <= 0:
            raise ValueError(f"block {block_id!r} has non-positive size {nbytes}")
        hi = addr + nbytes
        i = bisect.bisect_right(self._los, addr)
        if i > 0 and self._blocks[i - 1].hi > addr:
            raise ValueError(f"block {block_id!r} overlaps {self._blocks[i - 1].block_id!r}")
        if i < len(self._blocks) and hi > self._blocks[i].lo:
            raise ValueError(f"block {block_id!r} overlaps {self._blocks[i].block_id!r}")
        self._los.insert(i, addr)
        self._blocks.insert(i, Block(block_id, Address(addr), Address(hi), content_hash))
        if content_hash is not None:
            self._id2hash[block_id] = content_hash

    def set_hash(self, block_id: BlockId, content_hash: ContentHash) -> None:
        self._id2hash[block_id] = content_hash

    def resolve(self, addr: int) -> Block | None:
        i = bisect.bisect_right(self._los, addr) - 1
        if 0 <= i < len(self._blocks):
            b = self._blocks[i]
            if b.lo <= addr < b.hi:
                return b
        return None

    def hash_of(self, block_id: BlockId) -> ContentHash | None:
        return self._id2hash.get(block_id)

    @property
    def extent(self) -> int | None:
        """How many KV units this map covers, if it was built on a lattice."""
        return self._lattice[2] if self._lattice else None

    def lattice_unit(self, addr: int | None) -> int | None:
        """This map's KV-unit index for ``addr`` — *even when the index lies beyond the map's
        extent*. None when the map has no lattice, or the address is not on it.

        This is the distinction ``resolve`` cannot make and whose absence hid a real bug: an
        address that IS this pool's traffic but sits past a too-small derived extent looked
        exactly like an address that is not KV traffic at all — both resolved to None and were
        dropped silently. One is a recoverable coverage hole (widen the extent); the other is
        correct filtering. Callers use this to tell them apart and say so."""
        if addr is None or self._lattice is None:
            return None
        base, stride, _ = self._lattice
        off = int(addr) - base
        if off < 0 or off % stride:
            return None
        return off // stride

    def is_valid_for(self, kv_caches, *, block_dim: int = 0) -> bool:
        """Cheap drift check: do the live KV tensors still sit at the addresses this
        map was built from? Returns False if any base moved (then rebuild via
        from_vllm). NOTE: vLLM allocates the KV cache once and addresses are stable
        even under PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (the VMM allocator
        keeps a tensor's VA; only IB-pinned connectors care about physical remap), so
        this is defensive — it would only fire on a model reload / different allocator."""
        bases = {}
        for b in self._blocks:
            bases.setdefault(b.block_id[0] if isinstance(b.block_id, tuple) else 0, b.lo)
        for layer, t in enumerate(kv_caches):
            want = bases.get(layer)
            if want is not None and t.data_ptr() != want:
                return False
        return True

    def __len__(self) -> int:
        return len(self._blocks)

    @classmethod
    def from_kv_tensors(cls, kv_caches, *, block_dim: int = 0,
                        hashes: dict | None = None) -> BlockAddressMap:
        """From per-layer KV tensors: block_id -> base + id*stride. ``block_dim`` is
        the dimension indexed by block (adapter-configurable for vLLM's layout)."""
        m = cls()
        for layer, t in enumerate(kv_caches):
            base = t.data_ptr()
            nblocks = t.shape[block_dim]
            stride = t.stride(block_dim) * t.element_size()
            for bid in range(nblocks):
                h = hashes.get(bid) if hashes else None
                m.add((layer, bid), base + bid * stride, stride, h)
        return m

    @classmethod
    def from_vllm(cls, kv_caches, block_pool, *, block_dim: int = 0) -> BlockAddressMap:
        """Build from the live vLLM engine: addresses from the KV tensors, content
        hashes from the block pool (``block.block_hash``). This is the thin in-process
        external-controller read the join needs that KV Events cannot supply."""
        id2hash: dict[int, ContentHash] = {}
        for blk in getattr(block_pool, "blocks", []):
            h = getattr(blk, "block_hash", None)
            if h is not None:
                id2hash[blk.block_id] = _as_hash(h)
        m = cls()
        for layer, t in enumerate(kv_caches):
            base = t.data_ptr()
            nblocks = t.shape[block_dim]
            stride = t.stride(block_dim) * t.element_size()
            for bid in range(nblocks):
                m.add((layer, bid), base + bid * stride, stride, id2hash.get(bid))
        return m

    @classmethod
    def from_alloc(cls, base: int, num_blocks: int, block_stride: int, *,
                   layer: int | None = None,
                   content_hashes: dict | None = None) -> BlockAddressMap:
        """Build the bridge from a KV-pool base address + per-block stride (bytes) — no
        in-process tensor read, so address->block is engine-agnostic and zero-touch. The
        base+size come from eBPF capture of the pool's ``cudaMalloc`` (observe/ebpf/
        kv_alloc); ``block_stride`` is size/num_blocks or the model's per-block byte size.
        Block i occupies [base + i*block_stride, base + (i+1)*block_stride). ``layer``
        keys block ids as (layer, i) for a per-layer pool, else as plain i.
        ``content_hashes`` (block_id -> hash, e.g. from ``content_hashes_from_tokens``) is
        optional — content identity is itself token-derivable, so the whole transfer face
        needs no engine-internal read."""
        if num_blocks <= 0 or block_stride <= 0:
            raise ValueError(f"num_blocks={num_blocks} block_stride={block_stride} must be positive")
        m = cls()
        for i in range(num_blocks):
            bid = (layer, i) if layer is not None else i
            h = None
            if content_hashes:
                h = content_hashes.get(bid, content_hashes.get(i))
            m.add(bid, base + i * block_stride, block_stride, content_hash=h)
        m._lattice = (int(base), int(block_stride), int(num_blocks))
        return m

    @classmethod
    def from_alloc_family(cls, bases, num_blocks: int, block_stride: int) -> BlockAddressMap:
        """CONTAINMENT-grade map over a family of same-sized slabs: resolve() tells you traffic
        is pool traffic and which slab-local tile it hit. The ids are NOT engine block ids --
        pooled allocators place tensors at nonzero offsets inside slabs, so only the published
        kv_ptr lattice yields exact ids (observe.ebpf_source.derive_geometry). Use this map for
        classification and coverage, never for the ledger join."""
        if num_blocks <= 0 or block_stride <= 0:
            raise ValueError(f"num_blocks={num_blocks} block_stride={block_stride} must be positive")
        m = cls()
        for base in bases:
            for i in range(num_blocks):
                m.add(i, int(base) + i * block_stride, block_stride)
        return m


def content_hashes_from_tokens(slot_blocks: dict[int, list[int]],
                               slot_tokens: dict[int, list[int]],
                               block_size: int = 16) -> dict[int, ContentHash]:
    """Engine-agnostic content identity: derive each block's content hash from the TOKENS
    it covers — captured zero-touch by the network/input_ids face — instead of reading the
    engine's block pool. The per-block hash chains its parent's, so sequences sharing a
    token prefix get identical per-block hashes (prefix sharing) and a re-stored prefix is
    recognizable (recompute), with no engine-internal read. Deterministic (CRC32-chained),
    so it agrees across TP ranks and processes (unlike Python's salted ``hash``)."""
    out: dict[int, ContentHash] = {}
    for s, blocks in slot_blocks.items():
        toks = slot_tokens.get(s)
        if not toks:
            continue
        parent = 0
        for i, bid in enumerate(blocks):
            span = toks[i * block_size:(i + 1) * block_size]
            if not span:
                break
            buf = bytearray(parent.to_bytes(8, "little"))
            for t in span:
                buf += int(t).to_bytes(4, "little", signed=False)
            parent = zlib.crc32(bytes(buf))
            out[int(bid)] = ContentHash(parent)
    return out


def identify_kv_pool(allocs, *, kv_ptr: int | None = None, kv_addrs=()):
    """Identify the KV pool among eBPF-captured device allocations (kv_alloc) WITHOUT a
    size heuristic — the answer to a reviewer's 'which allocation is the KV cache, and
    isn't the largest one the model weights?'. The pool is fixed by an INDEPENDENT signal,
    never by being largest:

      * ``kv_ptr`` — the KV-cache base pointer the attention kernel is handed (the same
        kind of 'input the kernel must be handed' as the block table; the pass can publish
        it). The pool is the allocation containing it. Exact, works for any workload.
      * ``kv_addrs`` — addresses the eBPF-observed KV-offload transfers touch; the pool is
        the allocation that CONTAINS them — a membership test over independently-captured
        KV traffic that a weight/activation buffer fails. Needs offload activity.

    Returns the (base, size) of the pool, or None if no allocation matches. ``allocs`` is
    kv_alloc records ({addr, size}) or (base, size) pairs."""
    norm = [((int(a["addr"]), int(a["size"])) if isinstance(a, dict)
             else (int(a[0]), int(a[1]))) for a in allocs]

    def containing(x: int):
        x = int(x)
        for base, size in norm:
            if base <= x < base + size:
                return (base, size)
        return None

    if kv_ptr is not None:
        return containing(kv_ptr)
    if kv_addrs:
        from collections import Counter
        votes: Counter = Counter()
        for x in kv_addrs:
            a = containing(x)
            if a is not None:
                votes[a] += 1
        return votes.most_common(1)[0][0] if votes else None
    return None


def _as_hash(h) -> ContentHash:
    if isinstance(h, int):
        return ContentHash(h)
    if isinstance(h, (bytes, bytearray)):
        return ContentHash(int.from_bytes(bytes(h[:8]), "big", signed=False))
    if hasattr(h, "__int__"):
        return ContentHash(int(h))
    raise TypeError(f"unsupported vLLM block hash type: {type(h).__name__}")
