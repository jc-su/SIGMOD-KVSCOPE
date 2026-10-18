"""Verify the collector captured the driver-API cuMemcpyBatchAsync offload and the
fuser resolved each batch element to its KV block as a D2H (HBM->DRAM) edge."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from observe import BlockAddressMap, fuse_copies, read_copies_file  # noqa: E402

meta = json.load(open("/tmp/ekv_drvmap.json"))
m = BlockAddressMap()
for bid in range(meta["n"]):
    m.add(bid, meta["base"] + bid * meta["stride"], meta["stride"],
          content_hash=2000 + bid)

copies = read_copies_file("/tmp/ekv_copy.jsonl")
batch = [c for c in copies if c.direction == "BATCH"]
edges = fuse_copies(copies, m)
resolved = {e.block_id for e in edges}
expected = set(meta["copied"])

print(f"copy events: {len(copies)} (BATCH: {len(batch)})")
print(f"resolved blocks: {sorted(resolved)}  expected: {sorted(expected)}")
missing = expected - resolved
assert not missing, f"FAIL: batch offload missed blocks {sorted(missing)}"
costed = 0
for e in edges:
    if e.block_id in expected:
        assert e.src_tier == "GPU" and e.dst_tier == "CPU", \
            f"FAIL: wrong tiers for block {e.block_id}: {e.src_tier}->{e.dst_tier}"
        assert e.content_hash == 2000 + e.block_id
        if e.cost:
            costed += 1
print(f"batch edges with completion latency (cost): {costed}/{len(expected)}")
assert costed >= len(expected), "FAIL: batch completion latency not correlated (#5)"
print("PASS: driver-API cuMemcpyBatchAsync offload captured, fused to KV D2H edges, "
      "AND completion latency correlated via the batch stream (#2 driver path + #5)")
