"""Reload-aware attribution. From the kv_copy batch copies (KV offload/reload), the kv_alloc KV-pool
device allocations, and the index_lookup request->blocks (here taken from the in-process oracle; eKV's own
index_lookup is validated equal to it in the e2e), attribute the per-request and per-prefix reload bytes:
the offload->reload tax the control plane reports only in aggregate. A reload is a batch copy whose
destination lands in a KV-pool allocation; its destination address maps to a physical block, and the
block maps to the request(s) that hold it. Usage: _reload_check.py <cap.kv> <alloc> <oracle> [num_blocks]"""
import json
import sys
from collections import Counter, defaultdict

cap, alloc, oracle = sys.argv[1], sys.argv[2], sys.argv[3]
NUM_BLOCKS = int(sys.argv[4]) if len(sys.argv) > 4 else 160

pool = [(a["addr"], a["addr"] + a["size"], a["size"])
        for a in (json.loads(l) for l in open(alloc) if '"addr"' in l)]


def find_alloc(addr):
    for base, end, size in pool:
        if base <= addr < end:
            return base, size
    return None


recs = [json.loads(l) for l in open(cap) if '"type":"enqueue"' in l and 'BATCH' in l]
reloads, offloads = [], 0
for r in recs:
    da, sa = find_alloc(r["dst"]), find_alloc(r["src"])
    if da and not sa:                                  # reload: destination is in the KV pool
        base, size = da
        reloads.append(((r["dst"] - base) // (size // NUM_BLOCKS), r["bytes"]))
    elif sa and not da:
        offloads += 1
blk_bytes = Counter()
for blk, b in reloads:
    blk_bytes[blk] += b
print(f"pool allocs: {len(pool)} (MiB: {sorted({round(s / 2**20, 1) for _, _, s in pool})[:6]})")
print(f"batch copies {len(recs)}: reloads(in) {len(reloads)}, offloads(out) {offloads}, "
      f"reload {sum(b for _, b in reloads) / 2**20:.0f} MiB; distinct blocks reloaded {len(blk_bytes)}"
      + (f", idx {min(blk_bytes)}..{max(blk_bytes)}" if blk_bytes else ""))

od = json.load(open(oracle))
req_blocks = defaultdict(set)
req_pref, prompts = {}, od["prompts"]
for snap in od["oracle"]:
    for rid, d in snap.items():
        req_blocks[rid].update(d["b"])
        if rid not in req_pref:
            for p in prompts:
                if tuple(p["tok"][:20]) == tuple(d["p"][:20]):
                    req_pref[rid] = p["prefix"]
                    break
blk_to_reqs = defaultdict(set)
for rid, bs in req_blocks.items():
    for b in bs:
        blk_to_reqs[b].add(rid)

matched = sum(1 for b in blk_bytes if b in blk_to_reqs)
req_reload, pref_reload = Counter(), Counter()
for blk, byts in blk_bytes.items():
    owners = blk_to_reqs.get(blk, set())
    for rid in owners:
        req_reload[rid] += byts / len(owners)
        if rid in req_pref:
            pref_reload[req_pref[rid]] += byts / len(owners)
print(f"reloaded blocks matched to a request: {matched}/{len(blk_bytes)}")
print(f"requests attributed: {len(req_reload)}; per-prefix reload (MiB): "
      f"{dict(sorted({k: round(v / 2**20) for k, v in pref_reload.items()}.items()))}")
