"""Budget allocation across requests: optimality against exhaustive search, and its guarantees."""

import itertools

import numpy as np
import pytest

from observe.allocation import Request, allocate


def retained(request: Request, kept: np.ndarray) -> float:
    return float(request.shares()[kept].sum())


def random_requests(rng: np.random.Generator, n_requests: int, max_blocks: int) -> list[Request]:
    out = []
    for _ in range(n_requests):
        n = int(rng.integers(2, max_blocks + 1))
        scores = rng.random(n) ** float(rng.uniform(1, 6))
        out.append(Request(scores=scores, weight=float(rng.uniform(0.5, 3.0))))
    return out


def total_value(requests: list[Request], masks: list[np.ndarray]) -> float:
    return sum(r.weight * retained(r, m) for r, m in zip(requests, masks, strict=True))


def lowest_share(requests: list[Request], masks: list[np.ndarray]) -> float:
    return min(retained(r, m) for r, m in zip(requests, masks, strict=True))


def best_over_subsets(requests: list[Request], budget: int, score) -> float:
    """The best ``score(requests, masks)`` over every way to keep ``budget`` blocks with pins kept."""
    pins = [r.pinned_mask() for r in requests]
    slots = [(i, b) for i, p in enumerate(pins) for b in np.flatnonzero(~p)]
    free = budget - sum(int(p.sum()) for p in pins)
    best = -1.0
    for chosen in itertools.combinations(slots, free):
        masks = [p.copy() for p in pins]
        for i, b in chosen:
            masks[i][b] = True
        best = max(best, score(requests, masks))
    return best


def test_sum_matches_exhaustive_search():
    rng = np.random.default_rng(0)
    for _ in range(40):
        requests = random_requests(rng, 3, 5)
        total = sum(len(r.scores) for r in requests)
        budget = int(rng.integers(sum(int(r.pinned_mask().sum()) for r in requests), total))
        kept = allocate(requests, budget, "sum")
        best = best_over_subsets(requests, budget, total_value)
        assert sum(int(m.sum()) for m in kept) == budget
        assert total_value(requests, kept) == pytest.approx(best, abs=1e-12)


def test_maxmin_matches_exhaustive_search():
    rng = np.random.default_rng(1)
    for _ in range(40):
        requests = [Request(scores=r.scores) for r in random_requests(rng, 3, 5)]
        total = sum(len(r.scores) for r in requests)
        budget = int(rng.integers(sum(int(r.pinned_mask().sum()) for r in requests), total))
        kept = allocate(requests, budget, "maxmin")
        best = best_over_subsets(requests, budget, lowest_share)
        assert sum(int(m.sum()) for m in kept) <= budget
        assert lowest_share(requests, kept) == pytest.approx(best, abs=1e-9)


def test_uniform_gives_every_request_the_same_fraction_ranked_by_its_own_scores():
    short = Request(scores=np.array([0.1, 0.5, 0.2, 0.1, 0.1]))
    long = Request(scores=np.array([0.05, 0.05, 0.4, 0.1, 0.1, 0.1, 0.1, 0.05, 0.03, 0.02]))
    kept = allocate([short, long], 9, "uniform")
    assert sum(int(m.sum()) for m in kept) == 9
    assert abs(int(kept[0].sum()) / 5 - int(kept[1].sum()) / 10) <= 0.2
    assert kept[0][1] and kept[1][2]  # each request keeps its own highest-scoring block


def test_uniform_gives_up_the_total_share_that_the_marginal_allocation_keeps():
    rng = np.random.default_rng(4)
    gains = 0
    for _ in range(30):
        requests = [Request(scores=r.scores) for r in random_requests(rng, 4, 14)]
        budget = sum(len(r.scores) for r in requests) // 2
        fair = total_value(requests, allocate(requests, budget, "uniform"))
        marginal = total_value(requests, allocate(requests, budget, "sum"))
        assert marginal >= fair - 1e-12
        gains += marginal > fair + 1e-9
    assert gains > 0  # requests differ, so the marginal allocation does strictly better


def test_uniform_rejects_weights_instead_of_ignoring_them():
    with pytest.raises(ValueError):
        allocate([Request(scores=np.ones(4), weight=2.0), Request(scores=np.ones(4))], 5, "uniform")


def test_identical_requests_split_the_budget_evenly():
    scores = np.array([0.4, 0.2, 0.15, 0.1, 0.08, 0.07])
    requests = [Request(scores=scores) for _ in range(4)]
    for objective in ("sum", "maxmin", "uniform"):
        counts = [int(m.sum()) for m in allocate(requests, 12, objective)]
        assert sum(counts) == 12
        assert max(counts) - min(counts) <= 1


def test_concentrated_requests_give_up_blocks_to_diffuse_ones():
    concentrated = Request(scores=np.array([0.9, 0.02, 0.02, 0.02, 0.02, 0.02]))
    diffuse = Request(scores=np.full(6, 1 / 6))
    kept = allocate([concentrated, diffuse], 7, "sum")
    assert int(kept[0].sum()) < int(kept[1].sum())
    uniform_value = retained(concentrated, np.array([1, 1, 1, 0, 0, 0], bool)) + retained(
        diffuse, np.array([1, 1, 1, 1, 0, 0], bool)
    )
    value = sum(retained(r, m) for r, m in zip([concentrated, diffuse], kept, strict=True))
    assert value >= uniform_value


def test_pinned_blocks_are_kept_and_counted():
    requests = [
        Request(scores=np.array([0.0, 1.0, 1.0, 0.0])),
        Request(scores=np.array([0.0, 0.0, 0.0, 1.0])),
    ]
    kept = allocate(requests, 5, "sum")
    for r, m in zip(requests, kept, strict=True):
        assert m[0] and m[-1]
    assert sum(int(m.sum()) for m in kept) == 5


def test_budget_below_the_pinned_blocks_is_an_error():
    requests = [Request(scores=np.ones(5)), Request(scores=np.ones(5))]
    with pytest.raises(ValueError):
        allocate(requests, 3, "sum")


def test_budget_covering_everything_keeps_everything():
    requests = [Request(scores=np.ones(4)), Request(scores=np.ones(6))]
    assert all(m.all() for m in allocate(requests, 10, "sum"))
    assert all(m.all() for m in allocate(requests, 99, "maxmin"))


def test_a_heavier_request_keeps_at_least_as_many_blocks():
    scores = np.array([0.3, 0.25, 0.2, 0.15, 0.1])
    light, heavy = Request(scores=scores, weight=1.0), Request(scores=scores, weight=4.0)
    kept = allocate([light, heavy], 6, "sum")
    assert int(kept[1].sum()) >= int(kept[0].sum())


def test_maxmin_does_not_lower_the_smallest_retained_share():
    rng = np.random.default_rng(2)
    for _ in range(30):
        requests = [Request(scores=r.scores) for r in random_requests(rng, 4, 12)]
        budget = sum(len(r.scores) for r in requests) // 2
        fair = allocate(requests, budget, "maxmin")
        greedy = allocate(requests, budget, "sum")
        low = min(retained(r, m) for r, m in zip(requests, fair, strict=True))
        low_greedy = min(retained(r, m) for r, m in zip(requests, greedy, strict=True))
        assert low >= low_greedy - 1e-12


def test_a_request_without_evidence_is_treated_as_uniform():
    blind = Request(scores=np.zeros(8))
    assert blind.shares().sum() == pytest.approx(1.0)
    kept = allocate([blind, Request(scores=np.ones(8))], 8, "sum")
    assert sum(int(m.sum()) for m in kept) == 8


def test_maxmin_rejects_weights_instead_of_ignoring_them():
    with pytest.raises(ValueError):
        allocate([Request(scores=np.ones(4), weight=2.0), Request(scores=np.ones(4))], 5, "maxmin")


def test_the_allocation_is_deterministic():
    rng = np.random.default_rng(3)
    requests = random_requests(rng, 5, 20)
    budget = sum(len(r.scores) for r in requests) // 3
    first, second = allocate(requests, budget, "sum"), allocate(requests, budget, "sum")
    assert all(np.array_equal(a, b) for a, b in zip(first, second, strict=True))


def reference_sum(requests: list[Request], budget: int) -> list[np.ndarray]:
    """The plain version: sort every unpinned block by weighted share, newer blocks first on ties, keep the top."""
    pinned = [r.pinned_mask() for r in requests]
    kept = [p.copy() for p in pinned]
    cells = [
        (r.weight * r.shares()[b], b, i) for i, r in enumerate(requests) for b in np.flatnonzero(~pinned[i])
    ]
    cells.sort(key=lambda c: (-c[0], -c[1], c[2]))
    for _gain, b, i in cells[: budget - sum(int(p.sum()) for p in pinned)]:
        kept[i][b] = True
    return kept


def reference_maxmin(requests: list[Request], budget: int) -> list[np.ndarray]:
    """The plain version: bisect the level with one search per request, then spend what is left on the lowest."""
    pinned = [r.pinned_mask() for r in requests]
    shares = [r.shares() for r in requests]
    order = [np.flatnonzero(~p)[np.argsort(-shares[i][~p], kind="stable")] for i, p in enumerate(pinned)]
    base = [float(shares[i][p].sum()) for i, p in enumerate(pinned)]
    prefix = [np.cumsum(shares[i][order[i]]) for i in range(len(requests))]
    free = budget - sum(int(p.sum()) for p in pinned)

    def counts_at(level: float) -> list[int]:
        out = []
        for i in range(len(requests)):
            need = level - base[i]
            out.append(
                0 if need <= 1e-12 else min(int(np.searchsorted(prefix[i], need - 1e-12)) + 1, len(order[i]))
            )
        return out

    low, high = 0.0, 1.0
    for _ in range(60):
        mid = (low + high) / 2
        if sum(counts_at(mid)) <= free:
            low = mid
        else:
            high = mid
    counts = counts_at(low)
    kept = [p.copy() for p in pinned]
    for i, k in enumerate(counts):
        kept[i][order[i][:k]] = True
    leftover = free - sum(counts)
    while leftover > 0:
        retained_now = [
            base[i] + (float(prefix[i][counts[i] - 1]) if counts[i] else 0.0)
            for i in range(len(requests))
            if counts[i] < len(order[i])
        ]
        if not retained_now:
            break
        open_rows = [i for i in range(len(requests)) if counts[i] < len(order[i])]
        i = min(
            open_rows,
            key=lambda row: (base[row] + (float(prefix[row][counts[row] - 1]) if counts[row] else 0.0), row),
        )
        kept[i][order[i][counts[i]]] = True
        counts[i] += 1
        leftover -= 1
    return kept


@pytest.mark.parametrize("seed", range(40))
def test_sum_equals_the_plain_version_including_ties(seed):
    rng = np.random.default_rng(seed)
    requests = []
    for _ in range(int(rng.integers(1, 9))):
        n = int(rng.integers(3, 40))
        scores = rng.integers(0, 4, size=n).astype(float)  # many equal scores, so many ties
        requests.append(Request(scores=scores, weight=float(rng.choice([1.0, 2.0, 0.5]))))
    total = sum(len(r.scores) for r in requests)
    must = sum(int(r.pinned_mask().sum()) for r in requests)
    budget = int(rng.integers(must, max(total, must + 1)))
    got, want = allocate(requests, budget, "sum"), reference_sum(requests, budget)
    assert all(np.array_equal(a, b) for a, b in zip(got, want, strict=True))


@pytest.mark.parametrize("seed", range(40))
def test_maxmin_equals_the_plain_version(seed):
    rng = np.random.default_rng(seed)
    requests = random_requests(rng, int(rng.integers(1, 9)), 40)
    requests = [Request(scores=r.scores) for r in requests]  # maxmin takes no weights
    total = sum(len(r.scores) for r in requests)
    must = sum(int(r.pinned_mask().sum()) for r in requests)
    budget = int(rng.integers(must, max(total, must + 1)))
    got, want = allocate(requests, budget, "maxmin"), reference_maxmin(requests, budget)
    assert all(np.array_equal(a, b) for a, b in zip(got, want, strict=True))
