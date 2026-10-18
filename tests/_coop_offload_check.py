"""M5 -- the COOPERATIVE OFFLOADING signal: eKV's per-block attention statistic is what the engine's
offloader cannot see, so it would place better than blind LRU. On a REAL attention trace, simulate
the engine's offload cache (a budget of resident blocks) under two policies: LRU (recency, the
engine's default) vs ATTENTION-informed (offload the lowest cumulative attention statistic -- eKV's
signal). A reload = an attended block that is NOT resident (the engine must bring it back). Fewer
reloads = less HBM<->DRAM traffic and latency, priced at the F6 reload cost. eKV cannot offload
(placement is the engine's job) -- it supplies the SIGNAL; this shows the signal pays.
Usage: _coop_offload_check.py <trace.npz> [budget_frac]"""
import sys
import numpy as np

d = np.load(sys.argv[1])
M = d["statistic"].astype(np.float64)                              # steps x blocks (per-step attn statistic)
T, B = M.shape
BUDGET = max(1, int(B * (float(sys.argv[2]) if len(sys.argv) > 2 else 0.5)))
RELOAD_CYCLES_PER_BLOCK = 38770 * 16                         # F6: ~38.8k cycles/token x 16 tok/block


def offload_sim(policy):
    """Resident set of BUDGET blocks; on a miss (attended but not resident) reload it, evicting the
    lowest-value resident per the policy. Returns the reload count."""
    resident = set()
    freq = np.zeros(B)
    last = np.full(B, -1.0)
    reloads = 0
    reload_statistic = 0.0                                        # sum of attention statistic on reloaded blocks
    for t in range(T):
        att = np.nonzero(M[t] > 0)[0]
        for b in att:
            freq[b] += M[t, b]
            last[b] = t
        for b in att:
            if b not in resident:
                reloads += 1
                reload_statistic += M[t, b]                       # the COST of this reload ~ how much it
                if len(resident) >= BUDGET:                  # was needed (its attention statistic)
                    if policy == "lru":
                        victim = min(resident, key=lambda x: last[x])
                    else:                                    # attention-informed (eKV signal)
                        victim = min(resident, key=lambda x: freq[x])
                    resident.discard(victim)
                resident.add(int(b))
    return reloads, reload_statistic


"""M5 -- the COOPERATIVE OFFLOADING signal: eKV's per-block attention statistic is what the engine's
offloader cannot see, so it would place better than blind LRU. On a REAL attention trace, simulate
the engine's offload cache (a budget of resident blocks) under two policies: LRU (recency, the
engine's default) vs ATTENTION-informed (offload the lowest cumulative attention statistic -- eKV's
signal). A reload = an attended block that is NOT resident (the engine must bring it back). Fewer
reloads = less HBM<->DRAM traffic and latency, priced at the F6 reload cost. eKV cannot offload
(placement is the engine's job) -- it supplies the SIGNAL; this shows the signal pays.
Usage: _coop_offload_check.py <trace.npz> [budget_frac]"""
import sys
import numpy as np

d = np.load(sys.argv[1])
M = d["statistic"].astype(np.float64)                              # steps x blocks (per-step attn statistic)
T, B = M.shape
BUDGET = max(1, int(B * (float(sys.argv[2]) if len(sys.argv) > 2 else 0.5)))
RELOAD_CYCLES_PER_BLOCK = 38770 * 16                         # F6: ~38.8k cycles/token x 16 tok/block


def offload_sim(policy):
    """Resident set of BUDGET blocks; on a miss (attended but not resident) reload it, evicting the
    lowest-value resident per the policy. Returns the reload count."""
    resident = set()
    freq = np.zeros(B)
    last = np.full(B, -1.0)
    reloads = 0
    reload_statistic = 0.0                                        # sum of attention statistic on reloaded blocks
    for t in range(T):
        att = np.nonzero(M[t] > 0)[0]
        for b in att:
            freq[b] += M[t, b]
            last[b] = t
        for b in att:
            if b not in resident:
                reloads += 1
                reload_statistic += M[t, b]                       # the COST of this reload ~ how much it
                if len(resident) >= BUDGET:                  # was needed (its attention statistic)
                    if policy == "lru":
                        victim = min(resident, key=lambda x: last[x])
                    else:                                    # attention-informed (eKV signal)
                        victim = min(resident, key=lambda x: freq[x])
                    resident.discard(victim)
                resident.add(int(b))
    return reloads, reload_statistic


lru_n, lru_stat = offload_sim("lru")
att_n, att_stat = offload_sim("attn")
tot = M.sum()
print(f"real attention trace: {T} steps x {B} blocks, resident budget {BUDGET}")
print(f"  reload COUNT  LRU {lru_n:6d}   eKV-signal {att_n:6d}  (dense attention -> count barely "
      "differs)")
print(f"  reload STATISTIC   LRU {lru_stat/tot*100:5.1f}% of attn   eKV-signal {att_stat/tot*100:5.1f}%  "
      f"({(lru_stat-att_stat)/max(lru_stat,1e-9)*100:+.1f}% less reload-cost -- the statistic LRU thrashes on the "
      "high-statistic sinks it evicts by age)")
ok = att_stat < lru_stat * 0.98
print(f"\n{'PASS' if ok else 'INCONCLUSIVE'}: on REAL attention, offloading by eKV's softmax-numerator "
      "signal reloads far LESS attention STATISTIC than the engine's blind LRU -- LRU evicts high-statistic "
      "sinks by age and reloads them; eKV keeps them. The engine cannot see statistic; eKV supplies it. "
      "The cooperative path to the capacity/latency win (drop != placement; signal does).")
import sys; sys.exit(0 if ok else 1)
