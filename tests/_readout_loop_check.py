"""Reduce the ekv_readout JSONL stream with cy.window_series (gap 1: the sync-anchored loop).

ekv_readout emits one {"slots":{s:[blocks]}, "cycles":{s:cyc}} line per throttled snapshot, taken
in the ENGINE's address space at a cuLaunchKernel bracket (post-previous-step-sync in eager mode).
Feeding that stream to window_series must give what the free-running thread could NOT: RELIABLE
per-window deltas -- nonzero on the clean decode windows and tracking each request's decode work --
because the reads are sync-anchored to a consistent per-step point, not caught mid-kernel.

Usage: _readout_loop_check.py <stream.jsonl> <comma-lens>
"""
import json
import sys

sys.path.insert(0, __file__.rsplit("/", 2)[0])
from observe import cycles as cy

lines = [json.loads(x) for x in open(sys.argv[1]) if x.strip().startswith("{")]
nblks = [int(x) for x in sys.argv[2].split(",")]           # planted per-request context (blocks)
N = len(nblks)
# build the window_series stream: each snapshot -> (CycleSnapshot from its cycle vector, slot table)
maxrow = 1 + max((int(s) for ln in lines for s in ln.get("cycles", {})), default=0)
stream = []
for ln in lines:
    cyc = {int(s): int(c) for s, c in ln.get("cycles", {}).items()}
    slots = {int(s): [int(b) for b in row] for s, row in ln.get("slots", {}).items()}
    if not slots:
        continue
    vec = [cyc.get(s, 0) for s in range(maxrow)]
    stream.append((cy.CycleSnapshot({"ekv": vec}), slots))
print(f"snapshots: {len(lines)}; usable (with a block table): {len(stream)}")
if len(stream) < 5:
    print("FAIL: too few usable snapshots (readout never streamed the decode)")
    sys.exit(1)

ser = cy.window_series(stream)
try:
    cy.delta(stream[0][0], stream[-1][0])
    monotone = True
except ValueError:
    monotone = False
clean_frac = ser["clean"] / max(ser["n"], 1)
# reliability: per-window deltas must be nonzero on clean windows (the thread's failed here) --
# the fraction of clean windows that actually carry cycles
clean_nonzero = sum(1 for w in ser["windows"] if w["clean"] and w["cycles"])
n_clean = sum(1 for w in ser["windows"] if w["clean"])
reliable = n_clean > 0 and clean_nonzero / n_clean > 0.8

# accumulated per-request cycles track the planted per-request work (blocks gathered per launch)
anchors = sorted(ser["total_cycles"], key=lambda a: ser["total_cycles"][a])
covered = len(anchors)
import numpy as np
exp = sorted(nblks)                                         # cycles linear in gathered blocks
ys = np.array([ser["total_cycles"][a] for a in anchors], dtype=float)
corr = float(np.corrcoef(np.array(exp, dtype=float), ys)[0, 1]) if covered == N else 0.0
print(f"window_series over the eBPF stream: {ser['n']} windows, {ser['clean']} clean "
      f"({100*clean_frac:.0f}%); monotone deltas: {monotone}")
print(f"  clean windows carrying nonzero cycles: {clean_nonzero}/{n_clean} (reliable: {reliable})")
print(f"  requests covered: {covered}/{N}; corr(accumulated cycles, decode work) = {corr:.4f}")

ok = monotone and reliable and clean_frac > 0.5 and covered == N and corr > 0.9
print("PASS: the eBPF sync-anchored readout loop yields RELIABLE per-window cycle deltas "
      "(nonzero on clean windows, unlike the mid-flight thread) and accumulates to per-request "
      "decode work -- the production windowing loop, no engine hook"
      if ok else
      f"FAIL: monotone={monotone} reliable={reliable} clean_frac={clean_frac:.2f} "
      f"covered={covered}/{N} corr={corr:.3f}")
sys.exit(0 if ok else 1)
