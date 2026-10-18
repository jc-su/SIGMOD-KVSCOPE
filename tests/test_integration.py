"""End-to-end fusion (no live deps): eBPF copies -> costed KVEdges -> KV Causal Flame,
with optional request attribution from the request↔block tie. Proves the load-bearing
transfer-cost fuser + flame + TP de-dup on synthetic-but-realistic inputs. The live
faces are validated separately (zero-touch: _served_zero_touch / _zero_touch_tie; OS
copies: _driver_offload / _join / _inline_load)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from observe import (BlockAddressMap, Copy, KVEdge, Op, RequestKVUse, Tier,  # noqa: E402
                     build_flame, dedup_by_hash, differential_flame, fuse_copies)

BASE, STRIDE, N = 0x700000000, 16384, 8


def _kvmap():
    m = BlockAddressMap()
    for bid in range(N):
        m.add(bid, BASE + bid * STRIDE, STRIDE, content_hash=9000 + bid)
    return m


def main():
    m = _kvmap()

    # OS face: offload (D2H) blocks 2,5,7 with measured completion latencies.
    copies = []
    for i, bid in enumerate([2, 5, 7]):
        enq = 1_000_000 + i * 1000
        copies.append(Copy(addr=BASE + bid * STRIDE, nbytes=STRIDE, direction="D2H",
                           enqueued_at=enq, finished_at=enq + 300_000 + bid * 1000,
                           via="event-sync"))
    copies.append(Copy(addr=0x1, nbytes=64, direction="D2H",   # non-KV -> must be dropped
                       enqueued_at=1, finished_at=2))

    edges = fuse_copies(copies, m, step=0)
    assert len(edges) == 3, f"non-KV copy not filtered: {edges}"
    by_block = {e.block_id: e for e in edges}
    for bid in [2, 5, 7]:
        e = by_block[bid]
        assert e.content_hash == 9000 + bid
        assert e.src_tier == "GPU" and e.dst_tier == "CPU"      # D2H = HBM->DRAM
        assert e.cost and e.cost > 0
    print("fuse: 3 offload copies -> 3 costed KVEdges (hash+tiers+cost); non-KV dropped  OK")

    # request attribution from the request↔block tie (kv_uses): block 5 was used by reqB
    # nearest in time -> the offload of block 5 is charged to reqB.
    uses = [RequestKVUse(request_id="reqB", role="store", block_ids=(5,), at=1_001_000)]
    redges = fuse_copies(copies, m, kv_uses=uses, time_window=10_000_000)
    e5 = next(e for e in redges if e.block_id == 5)
    assert str(e5.request) == "reqB" and "reqbridge" in e5.note, e5
    print("fuse: kv_uses tie attributes block 5's offload to request reqB  OK")

    # Ambiguous DEFAULT/BATCH records carry both addresses. If BOTH endpoints resolve
    # to KV blocks, this is device-to-device KV traffic, not an offload to DRAM.
    d2d = fuse_copies([Copy(addr=BASE + 2 * STRIDE, other_addr=BASE + 3 * STRIDE,
                            nbytes=STRIDE, direction="DEFAULT",
                            enqueued_at=10, finished_at=20)], m)
    assert len(d2d) == 1 and d2d[0].block_id == 3
    assert d2d[0].src_tier == Tier.HBM and d2d[0].dst_tier == Tier.HBM
    print("fuse: ambiguous two-KV-endpoint copy resolves as HBM->HBM on dst block  OK")

    # Flame: additive width (root == sum of children).
    fl = build_flame(edges, weight="cost")
    assert fl["value"] > 0 and fl["children"]
    assert abs(fl["value"] - sum(c["value"] for c in fl["children"])) < 1e-3
    print(f"flame: root={fl['value']:.0f}ns, top paths={[c['name'] for c in fl['children']]}  OK")

    # Differential: a policy that cut offload latency in half -> negative delta.
    after = [Copy(addr=BASE + bid * STRIDE, nbytes=STRIDE, direction="D2H",
                  enqueued_at=2_000_000, finished_at=2_000_000 + 150_000) for bid in [2, 5, 7]]
    d = differential_flame(edges, fuse_copies(after, m, step=1))
    assert d["delta"] < 0, f"expected improvement (negative delta), got {d['delta']}"
    print(f"differential flame: delta={d['delta']:.0f}ns (policy improved offload cost)  OK")

    # TP de-dup: 2 ranks offload the SAME logical block -> one edge, bytes summed, cost=max.
    rank_edges = [
        KVEdge(op=Op.COPY, block_id=(0, 5), content_hash=9005, prefix_hash=None,
               src_tier=Tier.HBM, dst_tier=Tier.DRAM, nbytes=8192, cost=300_000, note="rank0"),
        KVEdge(op=Op.COPY, block_id=(0, 5), content_hash=9005, prefix_hash=None,
               src_tier=Tier.HBM, dst_tier=Tier.DRAM, nbytes=8192, cost=350_000, note="rank1"),
    ]
    dd = dedup_by_hash(rank_edges)
    assert len(dd) == 1 and dd[0].nbytes == 16384 and dd[0].cost == 350_000 and "shards=2" in dd[0].note
    print(f"TP de-dup: 2 rank shards -> 1 edge, bytes={dd[0].nbytes} cost={dd[0].cost:.0f}ns  OK")
    print("PASS: eKV transfer-cost fusion integrated end-to-end (copies -> KVEdges -> flame)")


if __name__ == "__main__":
    main()
