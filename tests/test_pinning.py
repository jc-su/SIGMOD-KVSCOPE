"""Cost-aware pinning (paper S5.5): the per-block miss cost (which only eKV measures) lets a
cost-aware pin policy beat the frequency/LRU heuristic when miss costs are heterogeneous, and ties
it when they are uniform. Validates the SHAPE of the claim on a representative skewed trace; the
exact percentage depends on trace + cost params (a live-captured trace is a drop-in)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from observe.pinning import (assign_costs, gen_zipf_trace,  # noqa: E402
                             pinning_benefit, skew)

N_BLOCKS, N_ACCESS, EXP = 674, 58547, 1.0       # match the paper's trace shape
N_SLOTS, PIN = 200, 40


def _trace():
    return gen_zipf_trace(N_BLOCKS, N_ACCESS, exponent=EXP, seed=0)


def test_trace_skew_realistic():
    s = skew(_trace())
    assert 0.55 <= s <= 0.80, f"top-10% skew {s:.0%} unrealistic (paper ~66%)"


def test_uniform_cost_ties():
    # frequency IS expected cost when every miss costs the same -> the two policies coincide
    costs = assign_costs(N_BLOCKS, hi_frac=0.2, hi=1.0, lo=1.0)
    r = pinning_benefit(_trace(), costs, N_SLOTS, PIN)
    assert abs(r["reduction"]) < 1e-9, f"uniform cost must tie, got {r['reduction']:.3f}"


def test_heterogeneous_cost_aware_wins():
    costs = assign_costs(N_BLOCKS, hi_frac=0.2, hi=30.0, lo=1.0)   # eKV's host~1 / remote~30
    r = pinning_benefit(_trace(), costs, N_SLOTS, PIN)
    assert r["reduction"] > 0.10, f"cost-aware should cut stall, got {r['reduction']:.0%}"


def test_reduction_grows_with_heterogeneity():
    tr = _trace()
    reds = []
    for hi in (1, 2, 5, 10, 30):
        costs = assign_costs(N_BLOCKS, hi_frac=0.2, hi=float(hi), lo=1.0)
        reds.append(pinning_benefit(tr, costs, N_SLOTS, PIN)["reduction"])
    assert abs(reds[0]) < 1e-9, "uniform (hi=1) must tie"
    assert all(b >= a - 1e-9 for a, b in zip(reds, reds[1:])), f"should grow: {reds}"
    assert reds[-1] > 0.10, f"diverged cost should give a clear win, got {reds[-1]:.0%}"
    print("cost-aware total-stall reduction by heterogeneity (hi=1,2,5,10,30): "
          + ", ".join(f"{r:.0%}" for r in reds))


if __name__ == "__main__":
    test_trace_skew_realistic()
    test_uniform_cost_ties()
    test_heterogeneous_cost_aware_wins()
    test_reduction_grows_with_heterogeneity()
    print("PASS: cost-aware pinning ties at uniform cost, beats frequency as miss costs diverge "
          "(reproduces the S5.5 shape on a representative trace)")
