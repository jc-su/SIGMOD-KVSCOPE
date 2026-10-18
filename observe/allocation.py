"""Allocate a read budget across requests by the attention each block carries.

Each request has a score for each of its blocks: the share of its attention that falls on the
block. Keeping the k highest-scoring blocks of a request retains the sum of those k scores.
That retained share is a concave function of k, because the scores are taken in descending
order, so the best split of a budget is a marginal-utility allocation: spend each unit of budget
on the block with the highest marginal gain, wherever it is. A uniform fraction ignores that
requests differ in how concentrated their attention is, and it is optimal only when they do not.

Three objectives are supported. ``sum`` maximizes the total weighted retained share, which a
global greedy choice solves exactly. ``maxmin`` maximizes the smallest retained share of any
request, which a water level solves: every request keeps the fewest blocks that reach the level,
and the highest level the budget affords is found by bisection. ``uniform`` is the baseline that
ignores how requests differ: every request keeps the same fraction of its blocks, each ranked by
its own scores, as a per-request eviction method applies one fraction to all requests.

Every request keeps its pinned blocks, by default its first and its last. They count toward the
budget, and a budget below the pinned total is an error.
"""

import heapq
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

Objective = Literal["sum", "maxmin", "uniform"]


@dataclass(frozen=True)
class Request:
    """One request: the score of each of its blocks, and how much its retained share counts."""

    scores: np.ndarray
    weight: float = 1.0
    pinned: np.ndarray | None = None

    def pinned_mask(self) -> np.ndarray:
        if self.pinned is not None:
            return np.asarray(self.pinned, dtype=bool)
        mask = np.zeros(len(self.scores), dtype=bool)
        mask[[0, -1]] = len(self.scores) > 0
        return mask

    def shares(self) -> np.ndarray:
        """Scores normalized to sum to 1. A request with no evidence gets equal shares."""
        scores = np.maximum(np.asarray(self.scores, dtype=np.float64), 0.0)
        total = scores.sum()
        if total <= 0.0:
            return np.full(len(scores), 1.0 / max(len(scores), 1))
        return scores / total


def allocate(requests: Sequence[Request], budget: int, objective: Objective = "sum") -> list[np.ndarray]:
    """Return, for each request, a bool mask of the blocks to keep. At most ``budget`` blocks are kept."""
    sizes = [len(r.scores) for r in requests]
    pinned = [r.pinned_mask() for r in requests]
    if any(len(p) != n for p, n in zip(pinned, sizes, strict=True)):
        raise ValueError("a pinned mask must have one entry per block")
    if budget < 0:
        raise ValueError("the budget cannot be negative")
    must = sum(int(p.sum()) for p in pinned)
    if must > budget and budget < sum(sizes):
        raise ValueError(f"the budget {budget} cannot hold the {must} pinned blocks")
    if budget >= sum(sizes):
        return [np.ones(n, dtype=bool) for n in sizes]
    if objective == "sum":
        return _greedy(requests, pinned, budget - must)
    if objective in ("maxmin", "uniform"):
        if any(r.weight != 1.0 for r in requests):
            raise ValueError(f"the {objective} objective does not use request weights")
        if objective == "maxmin":
            return _water_level(requests, pinned, budget - must)
        return _uniform(requests, pinned, budget - must)
    raise ValueError(f"unknown objective {objective!r}")


def _uniform(requests: Sequence[Request], pinned: list[np.ndarray], free: int) -> list[np.ndarray]:
    """Give every request the same fraction of its unpinned blocks, the best by its own scores."""
    open_blocks = [np.flatnonzero(~p) for p in pinned]
    total = sum(len(o) for o in open_blocks)
    exact = [free * len(o) / total if total else 0.0 for o in open_blocks]
    quota = [int(x) for x in exact]
    for r in sorted(range(len(requests)), key=lambda i: (-(exact[i] - quota[i]), i))[: free - sum(quota)]:
        quota[r] += 1
    kept = [p.copy() for p in pinned]
    for r, req in enumerate(requests):
        order = open_blocks[r][np.argsort(-req.shares()[open_blocks[r]], kind="stable")]
        kept[r][order[: quota[r]]] = True
    return kept


def _greedy(requests: Sequence[Request], pinned: list[np.ndarray], free: int) -> list[np.ndarray]:
    """Keep the ``free`` unpinned blocks with the highest weighted share, newer blocks first on ties."""
    kept = [p.copy() for p in pinned]
    if free <= 0 or not requests:
        return kept
    open_blocks = [np.flatnonzero(~p) for p in pinned]
    index = np.concatenate(open_blocks)
    gain = np.concatenate(
        [req.weight * req.shares()[blocks] for req, blocks in zip(requests, open_blocks, strict=True)]
    )
    if free >= len(gain):
        selected = np.ones(len(gain), dtype=bool)
    else:
        # The ``free`` largest gains. A gain equal to the smallest chosen one goes to the newer block first,
        # then to the earlier request, so the result does not depend on how the partition orders equal values.
        worst = -gain
        threshold = np.partition(worst, free - 1)[free - 1]
        selected = worst < threshold
        ties = np.flatnonzero(worst == threshold)
        selected[ties[np.argsort(-index[ties], kind="stable")][: free - int(selected.sum())]] = True
    start = 0
    for r, blocks in enumerate(open_blocks):
        kept[r][blocks[selected[start : start + len(blocks)]]] = True
        start += len(blocks)
    return kept


def _water_level(requests: Sequence[Request], pinned: list[np.ndarray], free: int) -> list[np.ndarray]:
    """Raise the lowest retained share as far as the budget allows, then spend what is left."""
    shares = [req.shares() for req in requests]
    order = [np.flatnonzero(~p)[np.argsort(-shares[r][~p], kind="stable")] for r, p in enumerate(pinned)]
    base = [float(shares[r][p].sum()) for r, p in enumerate(pinned)]
    prefix = [np.cumsum(shares[r][order[r]]) for r in range(len(requests))]
    # One sorted array for every request: request r's running totals, shifted by 2r. A share total is at most 1,
    # so the ranges of different requests do not overlap, and one search answers every request at once.
    stacked = np.concatenate([p + 2.0 * r for r, p in enumerate(prefix)])
    sizes = np.array([len(o) for o in order])
    starts = np.concatenate(([0], np.cumsum(sizes)[:-1]))
    offsets = 2.0 * np.arange(len(requests))
    base_a = np.array(base)

    def extra(level: float) -> np.ndarray:
        """The blocks each request needs beyond its pinned ones to retain ``level`` of its attention."""
        need = level - base_a
        position = np.searchsorted(stacked, need - 1e-12 + offsets) - starts
        return np.where(need <= 1e-12, 0, np.minimum(position + 1, sizes))

    low, high = 0.0, 1.0
    for _ in range(60):
        mid = (low + high) / 2
        if int(extra(mid).sum()) <= free:
            low = mid
        else:
            high = mid
    counts = [int(c) for c in extra(low)]
    kept = [p.copy() for p in pinned]
    for r, k in enumerate(counts):
        kept[r][order[r][:k]] = True
    leftover = free - sum(counts)
    retained = [
        base[r] + (float(prefix[r][counts[r] - 1]) if counts[r] else 0.0) for r in range(len(requests))
    ]
    heap = [(retained[r], r) for r in range(len(requests)) if counts[r] < len(order[r])]
    heapq.heapify(heap)
    while leftover > 0 and heap:
        level, r = heapq.heappop(heap)
        block = order[r][counts[r]]
        kept[r][block] = True
        counts[r] += 1
        leftover -= 1
        if counts[r] < len(order[r]):
            heapq.heappush(heap, (level + float(shares[r][block]), r))
    return kept
