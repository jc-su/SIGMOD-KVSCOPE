"""Cost-aware cache pinning replay (paper S5.5). The point eKV makes: whether to PIN a KV block to
GPU memory should depend on its EXPECTED cost = reuse-frequency x per-block MISS COST, and the
miss cost (which tier the block reloads from) is the term only eKV measures. Frequency counting
alone is the standard LRU/reuse heuristic; this harness replays an access trace through an LRU
cache with a fixed pin budget and compares pinning by frequency vs by frequency x miss-cost, as a
function of cost heterogeneity.

Pure, offline, no GPU/engine: it consumes an access trace (a list of block ids -- captured live by
the fuser, or generated representatively here) + a per-block miss-cost vector, and returns the
total KV-stall (sum of miss costs) under each policy. Expectation reproduced by test_pinning:
  * at UNIFORM cost the two policies are identical (frequency IS expected cost);
  * as miss costs diverge, the cost-aware policy avoids the EXPENSIVE misses -> less total stall.
"""
from __future__ import annotations

import random
from collections import Counter, OrderedDict


def gen_zipf_trace(n_blocks: int, n_accesses: int, exponent: float = 1.1, seed: int = 0):
    """A representative skewed access trace: block popularity ~ Zipf(exponent). Returns the access
    list. (A live trace from the fuser is a drop-in replacement -- same shape: a list of block ids.)"""
    rng = random.Random(seed)
    ranks = list(range(1, n_blocks + 1))
    weights = [1.0 / (r ** exponent) for r in ranks]
    # map ranks to block ids in a fixed shuffled order so "popular" isn't trivially id 0..k
    ids = list(range(n_blocks))
    rng.shuffle(ids)
    return rng.choices(ids, weights=weights, k=n_accesses)


def skew(trace) -> float:
    """Fraction of accesses captured by the top 10% of distinct blocks (the paper's skew stat)."""
    c = Counter(trace)
    top = max(1, len(c) // 10)
    hot = sum(n for _, n in c.most_common(top))
    return hot / len(trace)


def assign_costs(n_blocks: int, hi_frac: float, hi: float, lo: float = 1.0, seed: int = 1):
    """Per-block miss cost, assigned INDEPENDENTLY of reuse (a random hi_frac of blocks cost `hi`,
    the rest `lo`). hi==lo gives the uniform-cost baseline. Grounded in eKV's measured tiers
    (host reload ~1, remote park ~30)."""
    rng = random.Random(seed)
    return {b: (hi if rng.random() < hi_frac else lo) for b in range(n_blocks)}


def replay_pin(trace, n_slots: int, pin_budget: int, costs, policy: str):
    """Replay `trace` through an LRU cache of `n_slots`, with `pin_budget` slots holding a fixed
    PIN SET chosen by `policy` ('frequency' = top by reuse count; 'cost' = top by count x miss-cost).
    Pinned blocks never miss; the rest share an LRU of (n_slots - pin_budget). Returns the total
    KV-stall (sum of miss costs) and the per-access stall list (for a tail metric)."""
    cnt = Counter(trace)
    if policy == "frequency":
        score = lambda b: cnt[b]                       # noqa: E731
    elif policy == "cost":
        score = lambda b: cnt[b] * costs[b]            # noqa: E731  (expected cost)
    else:
        raise ValueError(f"unknown policy {policy!r}")
    pin_budget = max(0, min(pin_budget, len(cnt)))
    pinned = set(sorted(cnt, key=score, reverse=True)[:pin_budget])
    lru_cap = max(0, n_slots - pin_budget)
    lru: OrderedDict = OrderedDict()
    total = 0.0
    stalls = []
    for b in trace:
        if b in pinned:
            stalls.append(0.0)
            continue
        if b in lru:
            lru.move_to_end(b)
            stalls.append(0.0)
        else:
            c = costs[b]
            total += c
            stalls.append(c)
            lru[b] = 1
            if len(lru) > lru_cap:
                lru.popitem(last=False)
    return total, stalls


def pinning_benefit(trace, costs, n_slots: int, pin_budget: int):
    """Total KV-stall under each policy + the cost-aware reduction. reduction>0 means the per-block
    miss cost (eKV's signal) helped over the frequency heuristic."""
    freq, fs = replay_pin(trace, n_slots, pin_budget, costs, "frequency")
    cost, cs = replay_pin(trace, n_slots, pin_budget, costs, "cost")
    red = 0.0 if freq == 0 else (freq - cost) / freq
    # tail: P99 of per-access stall (the costliest reloads a request waits on)
    def p99(xs):
        s = sorted(xs)
        return s[min(len(s) - 1, int(len(s) * 0.99))] if s else 0.0
    tail_red = 0.0
    tf, tc = p99(fs), p99(cs)
    if tf:
        tail_red = (tf - tc) / tf
    return {"freq_stall": freq, "cost_stall": cost, "reduction": red,
            "tail_freq": tf, "tail_cost": tc, "tail_reduction": tail_red}
