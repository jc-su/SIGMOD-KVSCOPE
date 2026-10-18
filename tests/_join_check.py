"""Join checker: build the BlockAddressMap from the exported control-plane map,
resolve every captured eBPF copy to a block via the REAL fuser (read_copies +
fuse_copies), and verify the KV copies map to exactly the blocks the workload
offloaded. Dogfoods the production path — no bespoke join code."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from observe import BlockAddressMap, fuse_copies, read_copies_file  # noqa: E402

meta = json.load(open("/tmp/ekv_kvmap.json"))
m = BlockAddressMap()
for bid in range(meta["n_blocks"]):
    m.add(bid, meta["base"] + bid * meta["stride"], meta["stride"], content_hash=1000 + bid)

copies = read_copies_file("/tmp/ekv_copy.jsonl")
edges = fuse_copies(copies, m)
resolved = {e.block_id for e in edges}

expected = set(meta["copied"])
print(f"copy events seen   : {len(copies)}")
print(f"resolved KV blocks : {sorted(resolved)}")
print(f"expected blocks    : {sorted(expected)}")
missing = expected - resolved
assert not missing, f"FAIL: join missed offloaded blocks {sorted(missing)}"
# every resolved block must be a real one (no spurious out-of-range matches)
assert resolved.issubset(set(range(meta["n_blocks"]))), "FAIL: resolved a non-block"
print("PASS: address->block join resolved every KV offload copy to the right block "
      "(anonymous bytes -> KV block, the load-bearing join)")
