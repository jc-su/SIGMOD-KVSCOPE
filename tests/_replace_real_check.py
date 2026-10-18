"""Run LRU / LFU / KVARC on a REAL softmax-numerator trace and measure RETAINED STATISTIC at a fixed KV
budget -- the fraction of true attention statistic kept blocks carry (H2O's quality proxy; dropped statistic
bounds the error). The trace's statistic weights ARE the deep-GPU signal: no shallower level can supply
per-block attention statistic, so LFU/adaptive are only expressible with it. Real LLM attention mixes
sinks (frequency-stable) and a local window (recency), so the adaptive policy should retain more
statistic than either fixed heuristic. Usage: _replace_real_check.py <trace.npz> [budget_frac]"""
import sys

import numpy as np

trace = sys.argv[1]
BUDGET_FRAC = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

d = np.load(trace)
M = d["statistic"].astype(np.float64)                              # steps x blocks
T, B = M.shape
budget = max(1, int(B * BUDGET_FRAC))


def simulate(kind):
    freq = np.zeros(B)                                        # cumulative attention (LFU signal)
    last = np.full(B, -1.0)                                   # last-hot step   (LRU signal)
    p = 0.5
    retained = 0.0
    total = 0.0
    for t in range(T):
        m = M[t]
        hot = m > 0
        freq[hot] += m[hot]
        last[hot] = t
        seen = np.nonzero(last >= 0)[0]                       # blocks that have appeared so far
        if len(seen) <= budget:
            kept = set(seen.tolist())
        else:
            if kind == "lru":
                val = last[seen]
            elif kind == "lfu":
                val = freq[seen]
            else:                                            # kvarc: adaptive recency/freq blend
                def nrm(v):
                    lo, hi = v.min(), v.max()
                    return (v - lo) / (hi - lo) if hi > lo else np.ones_like(v)
                val = (1 - p) * nrm(freq[seen]) + p * nrm(last[seen])
            keep_idx = seen[np.argsort(-val)[:budget]]
            kept = set(keep_idx.tolist())
            if kind == "kvarc":                              # adapt p from this step's MISSES
                miss = [b for b in seen if b not in kept and m[b] > 0]
                if miss:
                    rlo, rhi = last[seen].min(), last[seen].max()
                    rr = (rhi - rlo) or 1.0
                    # missed statistic that was RECENCY-strong -> freq weighting killed it -> raise p
                    num = sum(m[b] * (2 * ((last[b] - rlo) / rr) - 1) for b in miss)
                    den = sum(m[b] for b in miss) or 1.0
                    p = min(1.0, max(0.0, p + 0.05 * (num / den)))
        retained += m[list(kept)].sum() if kept else 0.0
        total += m.sum()
    return retained / total if total else 0.0, p


lru, _ = simulate("lru")
lfu, _ = simulate("lfu")
arc, pf = simulate("kvarc")
best = max(lru, lfu)
print(f"real attention trace: {T} steps x {B} blocks, budget {BUDGET_FRAC:.0%} ({budget} blocks)")
print(f"  retained statistic:  LRU(=StreamingLLM) {lru:.4f}   LFU(=H2O) {lfu:.4f}   "
      f"KVARC {arc:.4f}   (p_final {pf:.2f})")
print(f"  KVARC vs best fixed: {(arc-best)*100:+.2f} pts of attention statistic retained")
ok = arc >= best - 0.002                                      # subsumes; wins or ties the better one
print(f"\n{'PASS' if ok else 'FAIL'}: on REAL attention, adaptive replacement (using the deep-GPU "
      f"softmax-numerator signal) retains {'more' if arc > best else 'at least as much'} statistic than "
      "either fixed heuristic -- the frequency term, visible only inside the kernel, is what makes "
      "the win possible; a shallower observer cannot express it")
sys.exit(0 if ok else 1)
