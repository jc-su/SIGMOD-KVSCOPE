"""Algebraic invariants of the cycles attribution layer (observe/cycles.py). Each test pins one
clause of the module contract; all are pure CPU (snapshots constructed directly)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from observe.cycles import (CycleSnapshot, attribute_window, delta, request_cycles, rows_match,
                            shares, stable_slots, total, window_series)

FAIL = 0


def check(name, cond):
    global FAIL
    print(f"  {'ok' if cond else 'FAIL'}  {name}")
    if not cond:
        FAIL = 1


# --- monotone deltas: >= 0, negatives raise (never silently clamped) ---------------------------
a = CycleSnapshot({"k@0x1": [10, 0, 5]})
b = CycleSnapshot({"k@0x1": [15, 7, 5]})
check("delta is rowwise end-start, zero rows dropped", delta(a, b) == {0: 5, 1: 7})
try:
    delta(b, a)
    check("negative delta raises", False)
except ValueError:
    check("negative delta raises", True)

# --- additivity: sum across variants commutes with differencing; window-born variant counts ----
s0 = CycleSnapshot({"k@0x1": [10, 20]})
s1 = CycleSnapshot({"k@0x1": [30, 25], "k@0x2": [4, 6]})       # k@0x2 born inside the window
d = delta(s0, s1)
check("variant sum: per-row deltas add across variants", d == {0: 20 + 4, 1: 5 + 6})
check("total == delta from empty", total(s1) == {0: 34, 1: 31})
lhs = delta(s0, s1)
per_variant = [delta(CycleSnapshot({k: s0.rows.get(k, [])}), CycleSnapshot({k: v}))
               for k, v in s1.rows.items()]
rhs = {}
for pv in per_variant:
    for r, c in pv.items():
        rhs[r] = rhs.get(r, 0) + c
check("delta-then-sum == sum-then-delta", lhs == rhs)

# --- shares: partition of unity, scale invariance, empty-safe ----------------------------------
rc = {"A": 300, "B": 100}
sh = shares(rc)
check("shares partition unity", abs(sum(sh.values()) - 1.0) < 1e-12 and abs(sh["A"] - 0.75) < 1e-12)
check("shares scale-invariant (per-CTA multiplier cancels)",
      shares({k: 7 * v for k, v in rc.items()}) == sh)
check("empty shares stay empty", shares({}) == {} and shares({"A": 0}) == {})

# --- attribution: exact partition of the measured window ---------------------------------------
att = attribute_window(rc, 12.0)
check("attribution sums to the window exactly", abs(sum(att.values()) - 12.0) < 1e-9)
check("attribution keeps the split", abs(att["A"] - 9.0) < 1e-9)

# --- slot stability: same first block + grow-only prefix; else unstable ------------------------
st = stable_slots({0: [5, 6, 0], 1: [7, 0], 2: [9]},
                  {0: [5, 6, 8], 1: [3, 4], 3: [2]})
check("grow-only same-anchor slot is stable", st[0] is True)
check("first-block change (reuse) is unstable", st[1] is False)
check("slot absent at an edge is unstable", st[2] is False and st[3] is False)
check("non-prefix rewrite is unstable",
      stable_slots({0: [5, 6, 7]}, {0: [5, 9, 7]})[0] is False)

# --- request join: dummy rows excluded, anchor identity, stability filter ----------------------
rows = {0: 100, 1: 50, 2: 40, 7: 999}
blocks = {0: [11, 12], 1: [21, 0], 2: [0, 0], 7: []}           # 2,7 = dummy/pad (no real table)
req = request_cycles(rows, blocks)
check("dummy rows excluded; entity = own first block", req == {11: 100, 21: 50})
req2 = request_cycles(rows, blocks, fb_to_request={11: "reqA", 21: "reqB"})
check("anchor join names requests", req2 == {"reqA": 100, "reqB": 50})
req3 = request_cycles(rows, blocks, stable={0: True, 1: False})
check("unstable slots dropped, not apportioned", req3 == {11: 100})

# --- content-addressed variant selection ---------------------------------------------------------
ref = {0: [11, 12, 0], 1: [21, 22, 23, 0]}
check("real variant matches reference (prefix-nested rows)",
      rows_match({0: [11, 12, 13, 0]}, ref) is True)
check("dummy table does not match", rows_match({0: [64] * 4, 1: [64] * 4}, ref) is False)
check("zero/pad rows do not match", rows_match({0: [0, 0], 1: []}, ref) is False)

# --- window_series: the streaming reduction over sync-bounded snapshots ------------------------
# two requests hold slots 0,1 (anchors 5,7) across two clean windows, then request in slot 1
# finishes and a NEW request (anchor 9) reuses the slot -> the third window is a compaction
# boundary (unclean). timer accumulates monotonically; block tables are the per-window slot maps.
def snap(v0, v1):
    return CycleSnapshot({"k@0x1": [v0, v1]})


stream = [
    (snap(0, 0),   {0: [5, 6], 1: [7, 0]}),
    (snap(100, 40), {0: [5, 6], 1: [7, 8]}),        # window0: +100 / +40, clean
    (snap(180, 90), {0: [5, 6], 1: [7, 8]}),        # window1: +80 / +50, clean
    (snap(260, 130), {0: [5, 6], 1: [9, 0]}),       # window2: slot1 reused (7->9), UNCLEAN
]
res = window_series(stream, wall_times=[2.0, 2.0, 2.0])
check("window_series: n windows", res["n"] == 3)
check("clean windows counted (2 of 3; the reuse window is a boundary)", res["clean"] == 2)
check("clean windows attribute both slots", res["windows"][0]["cycles"] == {5: 100, 7: 40})
check("unclean window keeps only the stable slot (slot0), drops the reused slot1",
      res["windows"][2]["cycles"] == {5: 80})
check("accumulated cycles sum over windows (slot0: 100+80+80; slot1 stable only in w0,w1: 40+50)",
      res["total_cycles"] == {5: 260, 7: 90})
check("wall-time accumulates over CLEAN windows only, each partitioned exactly",
      abs(sum(res["total_time"].values()) - 4.0) < 1e-9)      # 2 clean windows x 2.0s
check("anchor join threads through the series",
      window_series(stream, fb_to_request={5: "A", 7: "B", 9: "C"})["total_cycles"]
      == {"A": 260, "B": 90})
try:
    window_series([(snap(50, 0), {0: [5]}), (snap(10, 0), {0: [5]})])   # decreasing -> not monotone
    check("series propagates the monotonicity mask", False)
except ValueError:
    check("series propagates the monotonicity mask", True)

print("PASS: cycles attribution algebra holds" if not FAIL else "FAIL: see rows above")
if __name__ == "__main__":
    sys.exit(FAIL)
