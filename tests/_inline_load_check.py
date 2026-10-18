"""Validate the inline_load latency path end-to-end on REAL captured copies: the H2D
reloads from _inline_load_workload, fused to DRAM->HBM (CPU->GPU) KVEdges with a cost
and a request, then run through attribute_latency -> inline_load latency > 0. (Request
attribution via a RequestKVUse(load) as in the validated many-to-many test; the bridge
is validated live separately. Direction is INFERRED by block membership, not declared:
src=host misses the GPU map, dst=GPU block hits -> DRAM->HBM.)"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from observe import (BlockAddressMap, Tier, attribute_latency,  # noqa: E402
                     fuse_copies, read_copies_file)
from observe.events import RequestKVUse
from observe.types import Nanoseconds, RequestId

meta = json.load(open("/tmp/ekv_inmap.json"))
m = BlockAddressMap()
for bid in range(meta["n"]):
    m.add(bid, meta["base"] + bid * meta["stride"], meta["stride"], content_hash=3000 + bid)

copies = read_copies_file("/tmp/ekv_copy.jsonl")
batch = [c for c in copies if c.direction == "BATCH"]
edges = fuse_copies(copies, m)
loads = [e for e in edges if e.src_tier == Tier.DRAM and e.dst_tier == Tier.HBM]
print(f"copy events: {len(copies)} (BATCH: {len(batch)})")
print(f"resolved DRAM->HBM (inline reload) edges: {len(loads)}  "
      f"blocks={sorted(e.block_id for e in loads)}  expected={sorted(meta['loaded'])}")
assert {e.block_id for e in loads} >= set(meta["loaded"]), "FAIL: H2D reloads not resolved as loads"
costed = [e for e in loads if e.cost]
assert costed, "FAIL: no inline reload carried a completion latency (cost)"

# attribute the reloads to a request and fold to inline_load latency
T = max(c.enqueued_at for c in batch)
uses = [RequestKVUse(request_id=RequestId("rLoad"), role="load",
                     block_ids=tuple(meta["loaded"]), at=Nanoseconds(T))]
edges_r = fuse_copies(copies, m, kv_uses=uses, time_window=10_000_000)
latencies = attribute_latency(edges=edges_r)
print(f"per-request inline_load latency (ms): "
      f"{ {r: round(h.inline_load/1e6, 3) for r, h in latencies.items()} }")
assert any(h.inline_load > 0 for h in latencies.values()), "FAIL: inline_load latency not attributed"
assert all(h.async_load == 0 for h in latencies.values()), "inline reload must NOT count as async_load"
print("PASS: inline_load latency path validated on real captured H2D reloads — CPU->GPU "
      "reloads fused as DRAM->HBM edges with cost, attributed to the request as inline_load.")
