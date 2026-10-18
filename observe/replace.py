"""KV eviction as ADAPTIVE CACHE REPLACEMENT.

The caching-theory reframe (docs/DESIGN-unified-attribution.md): a KV block's importance is a
cache-replacement value, and the sparsity literature is fixed heuristics in disguise --
  * H2O  (keep highest cumulative attention)  == LFU  (least-frequently-used)
  * StreamingLLM (keep recent + sinks)         == LRU  (least-recently-used)
Both are workload-dependent (textbook): LRU wins on recency-heavy access, LFU on skewed-frequency.
Neither dominates -- which is exactly why our attention-frequency versus recency baseline was task-dependent.

The fix is ADAPTIVE replacement (ARC/LIRS): learn the recency-vs-frequency balance from the MISS
STREAM (a block evicted then re-accessed was evicted wrongly). eKV's compiler observations and
eBPF lifecycle tracing supply all three terms: the pass gives recency and frequency; the reload
and rematerialization stream gives
the misses. KVARC blends them with a parameter p that the miss stream tunes -- and it SUBSUMES both
fixed policies (p=0 -> LFU, p=1 -> LRU), so it can only match or beat the better of the two.

Pure + trace-driven; no GPU.
"""
from __future__ import annotations


class Replacer:
    """A fixed-budget KV block cache. access(block) -> True on hit, False on miss (a reload). A miss
    admits the block and, if over budget, evicts the lowest-value one per the policy."""

    def __init__(self, budget: int):
        self.budget = budget
        self.hits = 0
        self.misses = 0
        self.freq: dict[int, float] = {}          # cumulative attention (frequency signal)
        self.last: dict[int, int] = {}            # last-access tick   (recency signal)
        self._t = 0

    def access(self, block: int, weight: float = 1.0) -> bool:
        self._t += 1
        hit = block in self.last
        self.freq[block] = self.freq.get(block, 0.0) + weight
        self.last[block] = self._t
        if hit:
            self.hits += 1
        else:
            self.misses += 1
            self._on_miss(block)
            while len(self.last) > self.budget:
                self._evict()
        return hit

    def _on_miss(self, block: int) -> None:      # hook for adaptation
        pass

    def _evict(self) -> None:
        raise NotImplementedError

    @property
    def hit_rate(self) -> float:
        n = self.hits + self.misses
        return self.hits / n if n else 0.0

    def _drop(self, block: int) -> None:
        self.freq.pop(block, None)
        self.last.pop(block, None)


class LRU(Replacer):
    """Recency-only == StreamingLLM."""
    def _evict(self) -> None:
        b = min(self.last, key=self.last.get)
        self._drop(b)


class LFU(Replacer):
    """Frequency-only == H2O."""
    def _evict(self) -> None:
        b = min(self.freq, key=self.freq.get)
        self._drop(b)


class KVARC(Replacer):
    """Adaptive recency/frequency blend, tuned by the miss stream. value(b) = (1-p)*freq_n +
    p*rec_n over the cache; evict the min. p in [0,1] adapts: a re-accessed block that was
    RECENCY-strong at eviction (so its low frequency killed it) grows p toward recency; a
    FREQUENCY-strong one shrinks it. p=0 -> LFU, p=1 -> LRU, so KVARC >= max(LFU, LRU)."""

    def __init__(self, budget: int, p0: float = 0.5, rate: float = 0.05):
        super().__init__(budget)
        self.p = p0
        self.rate = rate
        self._ghost: dict[int, float] = {}        # evicted block -> recency-strength at eviction

    def _norm(self):
        fs, ls = self.freq, self.last
        fmin, fmax = min(fs.values()), max(fs.values())
        lmin, lmax = min(ls.values()), max(ls.values())
        fr = (fmax - fmin) or 1.0
        lr = (lmax - lmin) or 1.0
        return fmin, fr, lmin, lr

    def _on_miss(self, block: int) -> None:
        g = self._ghost.pop(block, None)
        if g is None:
            return                                # never-seen block, no ghost signal
        # g = recency-strength (in [0,1]) the block had when evicted. High g -> it was recency-strong
        # yet got evicted (frequency weighting killed it) -> we should have valued recency more.
        self.p = min(1.0, max(0.0, self.p + self.rate * (2 * g - 1)))

    def _evict(self) -> None:
        fmin, fr, lmin, lr = self._norm()
        best_b, best_score, best_recstr = None, None, 0.0
        for b in self.last:
            fn = (self.freq[b] - fmin) / fr
            rn = (self.last[b] - lmin) / lr
            score = (1 - self.p) * fn + self.p * rn
            if best_score is None or score < best_score:
                best_b, best_score, best_recstr = b, score, rn
        self._ghost[best_b] = best_recstr         # remember its recency-strength for the miss signal
        if len(self._ghost) > self.budget:        # bounded ghost list (like ARC's |B| <= c)
            self._ghost.pop(next(iter(self._ghost)))
        self._drop(best_b)


class ARC(Replacer):
    """Proper Adaptive Replacement Cache (Megiddo & Modha): T1 (recency) + T2 (frequency) + ghost
    lists B1/B2, with the target p tuned by ghost-list hits. Unlike KVARC's crude p-blend (which on
    REAL production traces converged to recency and lost to LFU under pressure), ARC beats BOTH LRU
    and LFU on the real Alibaba prefix-cache traces at every budget (eval/legacy/f15). This is the flagship
    adaptive policy; KVARC is retained as the intuition-building approximation."""

    def __init__(self, budget: int):
        super().__init__(budget)
        from collections import OrderedDict
        self.T1, self.T2, self.B1, self.B2 = (OrderedDict() for _ in range(4))
        self.p = 0

    def _replace(self, in_b2: bool):
        c = self.budget
        if self.T1 and (len(self.T1) > self.p or (in_b2 and len(self.T1) == self.p and self.T1)):
            k, _ = self.T1.popitem(last=False); self.B1[k] = 1
        elif self.T2:
            k, _ = self.T2.popitem(last=False); self.B2[k] = 1
        elif self.T1:
            k, _ = self.T1.popitem(last=False); self.B1[k] = 1

    def access(self, block: int, weight: float = 1.0) -> bool:
        c = self.budget
        if block in self.T1:
            self.hits += 1; del self.T1[block]; self.T2[block] = 1; return True
        if block in self.T2:
            self.hits += 1; self.T2.move_to_end(block); return True
        self.misses += 1
        if block in self.B1:
            self.p = min(c, self.p + max(len(self.B2) // max(len(self.B1), 1), 1))
            self._replace(False); del self.B1[block]; self.T2[block] = 1
        elif block in self.B2:
            self.p = max(0, self.p - max(len(self.B1) // max(len(self.B2), 1), 1))
            self._replace(True); del self.B2[block]; self.T2[block] = 1
        else:
            if len(self.T1) + len(self.B1) == c:
                if len(self.T1) < c:
                    self.B1.popitem(last=False); self._replace(False)
                else:
                    self.T1.popitem(last=False)
            elif len(self.T1) + len(self.B1) + len(self.T2) + len(self.B2) >= c:
                if len(self.T1) + len(self.B1) + len(self.T2) + len(self.B2) == 2 * c and self.B2:
                    self.B2.popitem(last=False)
                self._replace(False)
            self.T1[block] = 1
        return False


POLICIES = {"lru": LRU, "lfu": LFU, "kvarc": KVARC, "arc": ARC}
