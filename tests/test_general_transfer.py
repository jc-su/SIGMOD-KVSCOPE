"""The transfer face made engine-agnostic — no in-process engine read (no from_vllm):
  - dedup_by_hash collapses TP shards by block_id (cross-rank-stable), even with NO
    content_hash (so it needs no block-pool read);
  - content_hashes_from_tokens derives content identity from the captured tokens
    (shared prefix -> one hash, diverging content -> different), deterministically;
  - BlockAddressMap.from_alloc resolves addr->block from a KV-pool base+stride (the
    base captured by eBPF from cudaMalloc), with no KV-tensor read.
No GPU. Proves the generality the paper's characterized boundary now claims."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from observe import (BlockAddressMap, KVEdge, Op, Tier,  # noqa: E402
                     content_hashes_from_tokens, dedup_by_hash, identify_kv_pool)


def test_dedup_by_block_id_without_content_hash():
    # two TP ranks offload the SAME logical block (same block_id), NO content_hash set
    edges = [
        KVEdge(op=Op.COPY, block_id=5, content_hash=None, prefix_hash=None,
               src_tier=Tier.HBM, dst_tier=Tier.DRAM, nbytes=8192, cost=300_000, note="rank0"),
        KVEdge(op=Op.COPY, block_id=5, content_hash=None, prefix_hash=None,
               src_tier=Tier.HBM, dst_tier=Tier.DRAM, nbytes=8192, cost=350_000, note="rank1"),
    ]
    dd = dedup_by_hash(edges)
    assert len(dd) == 1 and dd[0].nbytes == 16384 and dd[0].cost == 350_000 and "shards=2" in dd[0].note
    # distinct block ids must NOT collapse
    edges2 = edges + [KVEdge(op=Op.COPY, block_id=6, content_hash=None, prefix_hash=None,
                             src_tier=Tier.HBM, dst_tier=Tier.DRAM, nbytes=8192, cost=100, note="x")]
    assert len(dedup_by_hash(edges2)) == 2
    print("dedup: TP shards collapse by block_id with NO content_hash (engine-agnostic)  OK")


def test_token_content_hashes():
    pre = list(range(16))            # a shared 16-token prefix (one block)
    slot_tokens = {0: pre + [100, 101], 1: pre + [200, 201]}
    slot_blocks = {0: [5, 8], 1: [5, 9]}   # block 5 is the SHARED prefix block
    h = content_hashes_from_tokens(slot_blocks, slot_tokens, block_size=16)
    assert {5, 8, 9} <= set(h), h
    # shared prefix -> ONE consistent hash from either sequence; diverging blocks differ
    assert h[8] != h[9], "diverging content must hash differently"
    assert content_hashes_from_tokens(slot_blocks, slot_tokens, 16) == h, "must be deterministic"
    print(f"content-from-tokens: shared block 5 -> one hash; diverging 8!=9; deterministic  OK")


def test_from_alloc_addr_to_block():
    base, nb, stride = 0x70000000, 64, 4096      # eBPF-captured KV-pool base + per-block stride
    m = BlockAddressMap.from_alloc(base, nb, stride)
    assert len(m) == nb
    b = m.resolve(base + 10 * stride + 17)
    assert b is not None and b.block_id == 10, b
    assert m.resolve(base - 1) is None and m.resolve(base + nb * stride) is None
    m2 = BlockAddressMap.from_alloc(base, nb, stride, content_hashes={10: 0xABCD})
    assert m2.resolve(base + 10 * stride).content_hash == 0xABCD
    print("from_alloc: addr->block from base+stride (no tensor read) + token hashes  OK")


def test_identify_kv_pool_not_by_size():
    # the model WEIGHTS are the largest allocation; the KV pool is smaller.
    weights = (0x10_000000000, 8 << 30)   # 8 GB  (largest!)
    kv_pool = (0x70_000000000, 2 << 30)   # 2 GB
    acts = (0x90_000000000, 1 << 30)      # 1 GB
    allocs = [dict(addr=a, size=s) for a, s in (weights, kv_pool, acts)]
    # (A) the kernel's KV-cache pointer lands in the KV pool -> picks it, NOT weights
    assert identify_kv_pool(allocs, kv_ptr=kv_pool[0] + 12345) == kv_pool
    # (B) observed KV-offload addresses are contained by the KV pool -> picks it
    addrs = [kv_pool[0] + i * 4096 for i in (0, 5, 99, 1000)]
    assert identify_kv_pool(allocs, kv_addrs=addrs) == kv_pool
    # a pointer in no captured allocation -> None (no false positive)
    assert identify_kv_pool(allocs, kv_ptr=0xDEADBEEF) is None
    # 'largest' would have (wrongly) returned weights; we never use size
    assert max(allocs, key=lambda a: a["size"])["addr"] == weights[0]
    print("identify_kv_pool: picks the KV pool by kernel pointer / KV-traffic containment, "
          "NOT by size (weights was 4x larger)  OK")


def main():
    test_dedup_by_block_id_without_content_hash()
    test_token_content_hashes()
    test_from_alloc_addr_to_block()
    test_identify_kv_pool_not_by_size()
    print("PASS: transfer face is engine-agnostic (block_id de-dup + token content + "
          "from_alloc + non-heuristic pool id)")


if __name__ == "__main__":
    main()
