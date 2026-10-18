"""Validate the SGLang serving e2e + accounting use case (gap 2).

Input: the accumulator's dump {anchor: {"ekv":[slots], "truth":[slots]}} -- per request (keyed by
its content-addressed first-slot anchor), eKV's pass-captured KV slot set and SGLang's own
kv_indices set, aggregated over the whole serving run. Checks:
  (1) RECONSTRUCTION: eKV's per-request slot set == SGLang's exactly, for every request (the pass
      index_lookup reproduces the radix allocator's per-request assignment over serving, not just one
      fwd call);
  (2) ACCOUNTING use case: per-request KV footprint (slots held) follows directly -- the signal
      SGLang exposes only in aggregate; report the distribution + the top request's share.

Usage: _sglang_e2e_check.py <result.json>
"""
import json
import sys

res = json.load(open(sys.argv[1]))
if not res:
    print("FAIL: no per-request records (pass/accumulator never ran)")
    sys.exit(1)

reqs = sorted(res.items(), key=lambda kv: len(kv[1]["truth"]))
n = len(reqs)
exact = sum(1 for _a, v in reqs if sorted(v["ekv"]) == sorted(v["truth"]))
print(f"requests reconstructed over the serving run: {n}")
print(f"{'anchor':>8}  {'eKV slots':>9}  {'SGLang':>7}  match")
for a, v in reqs:
    e, t = sorted(v["ekv"]), sorted(v["truth"])
    ok = e == t
    print(f"{a:>8}  {len(e):>9}  {len(t):>7}  {'EXACT' if ok else f'miss={sorted(set(t)-set(e))[:4]}'}")

# accounting use case: per-request KV footprint (slots) -- SGLang reports only the aggregate pool use
foot = {a: len(v["truth"]) for a, v in reqs}
tot = sum(foot.values())
top = max(foot.values()) if foot else 0
print(f"\nper-request KV footprint (slots): min={min(foot.values())} max={top} "
      f"total={tot}; top request holds {100*top/max(tot,1):.0f}% "
      f"(SGLang exposes only the aggregate; eKV attributes it per request, no engine hook)")

ok = n >= 2 and exact == n
print("\nPASS: on SGLang serving (real scheduler + radix allocator + continuous batching), eKV "
      "reconstructs every request's KV slot set EXACTLY from the pass index_lookup and attributes "
      "per-request KV footprint -- the accounting use case, engine untouched"
      if ok else f"\nFAIL: reconstructed {n}, exact {exact}/{n}")
sys.exit(0 if ok else 1)
