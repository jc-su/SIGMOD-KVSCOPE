"""Per-request KV ledger joining kernel observations with host-side lifecycle tracing.

Both sources resolve to the same live request through physical KV-block ownership:

  * Triton instrumentation: request <-> blocks (block table), execution cycles, and accumulated
    attention weight.
  * eBPF lifecycle tracing: a costed ``KVEdge`` per physical transfer, resolved
    to an anchor by WHICH ANCHOR'S ROW CURRENTLY HOLDS THE EDGE'S BLOCK. The block is the spine;
    no fragile request-ID matching is needed, and the eBPF network request ID stays an
    OPTIONAL label (consistent with content-addressed identity).

The ledger degrades gracefully: with no lifecycle source attached, every lifecycle field is 0 and
the record is exactly the pass-only view -- so a deployment that cannot run eBPF (no CAP_BPF)
loses nothing it had, and one that can gains movement/regeneration attribution for free. This is
what lets the eDSL read consumption AND lifecycle from one vocabulary while remaining general.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from collections.abc import Iterable

from .types import Op, Tier

# tiers a reload can come FROM (a lower tier brought back into HBM) and offload can go TO
_LOWER = frozenset((Tier.DRAM, Tier.SSD, Tier.REMOTE))


@dataclass
class KVRecord:
    """One request's KV record, keyed by its captured request anchor."""
    anchor: int
    # On-device observations from compiler instrumentation.
    n_blocks: int = 0
    cycles: int = 0                     # compute cost this window (timer)
    attention_weight: float = 0.0       # accumulated softmax-numerator statistic
    shared: bool = False                # holds a reused prefix block
    # Off-device lifecycle from eBPF, 0 when no lifecycle source is attached.
    reload_bytes: int = 0               # KV brought back into HBM from a lower tier
    reload_ns: int = 0                  # measured latency of those reloads
    offload_bytes: int = 0              # KV pushed out of HBM
    recompute: int = 0                  # blocks re-derived from tokens (no copy at all)
    born: float = 0.0                   # monotonic first-seen time; age_s = now - born. Lifecycle
                                        # fields accumulate over fusion windows, so a young record
                                        # is UNPRICED, not cheap (hold-until-priced routes on it)

    @property
    def was_offloaded(self) -> bool:
        return self.offload_bytes > 0

    @property
    def reload_latency_ms(self) -> float:
        """Per-request latency attributable to KV reload operations, in milliseconds."""
        return self.reload_ns / 1e6


class KVLedger:
    """Per-request records fed by compiler observations and eBPF lifecycle tracing."""

    def __init__(self) -> None:
        self.records: dict[int, KVRecord] = {}
        self.lifecycle_windows = 0                          # complete lifecycle windows ingested
        self.lifecycle_gaps = 0                             # incomplete windows (coverage holes)
        # Coverage of the MOST RECENT window, distinct from "ever": the counters above say the
        # tracer has worked at some point, which a policy cannot act on. This says whether the
        # movement fields are trustworthy RIGHT NOW, so a policy can route on being blind.
        self.last_window_complete: bool | None = None       # None: no window ingested yet
        self.unowned_edges = 0                              # movement resolved to a block no live
                                                            # request holds (see ingest_ebpf)

    def _rec(self, anchor: int) -> KVRecord:
        r = self.records.get(anchor)
        if r is None:
            r = self.records[anchor] = KVRecord(anchor, born=time.monotonic())
        return r

    # Compiler-inserted observations, refreshed every policy-loop update.
    def ingest_pass(self, blocks_by_anchor: dict[int, set[int]],
                    cycles_by_anchor: dict[int, int] | None = None,
                    attention_weight_by_request: dict[int, float] | None = None,
                    shared_anchors: Iterable[int] = ()) -> None:
        """Refresh the consumption view. Records for anchors no longer present are dropped so the
        ledger tracks the live request set (movement is folded in the SAME tick, below)."""
        live = set(blocks_by_anchor)
        for gone in [a for a in self.records if a not in live]:
            del self.records[gone]
        shared = set(shared_anchors)
        for a, blocks in blocks_by_anchor.items():
            r = self._rec(a)
            r.n_blocks = len(blocks)
            r.shared = a in shared
            if cycles_by_anchor is not None:
                r.cycles = cycles_by_anchor.get(a, 0)
            if attention_weight_by_request is not None:
                r.attention_weight = attention_weight_by_request.get(a, 0.0)

    # eBPF lifecycle events, resolved to requests through the captured block table.
    def ingest_ebpf(self, edges: Iterable, block_to_anchors: dict[int, set[int]],
                    complete: bool = True) -> None:
        """Fold each costed KVEdge into the record(s) of the anchor(s) that own its block. A
        shared prefix block resolves to every owning anchor -- its movement is a shared cost, and
        attributing it to each sharer is correct for a per-request retention/trim decision.

        ``complete`` is the window's coverage (ObservationBatch.complete): only a COMPLETE window
        counts as a lifecycle observation, so features() can distinguish known zero (the
        tracer watched and saw nothing) from unknown (no source, or a window it may have missed)."""
        self.last_window_complete = bool(complete)
        if complete:
            self.lifecycle_windows += 1
        else:
            self.lifecycle_gaps += 1
        for e in edges:
            b = getattr(e, "block_id", None)
            if b is None:
                continue
            anchors = block_to_anchors.get(b)
            if not anchors:
                # Resolved to a real block, but no LIVE request holds it -- usually legitimate
                # (an offload of a block whose request just finished), so it does not make the
                # window incomplete. Counted all the same: this is the last drop in the fusion
                # path, and an unaccounted drop is how the extent bug stayed invisible one layer
                # up. A rising ratio means captured rows are lagging lifecycle traffic.
                self.unowned_edges += 1
                continue
            src, dst = getattr(e, "src_tier", None), getattr(e, "dst_tier", None)
            nbytes = getattr(e, "nbytes", 0) or 0
            cost = getattr(e, "cost", 0) or 0
            for a in anchors:
                r = self._rec(a)
                if src == Tier.RECOMPUTE:                   # re-derived from tokens (no copy)
                    r.recompute += 1
                elif dst == Tier.HBM and src in _LOWER:     # reload: lower tier -> HBM
                    r.reload_bytes += nbytes
                    r.reload_ns += cost
                elif src == Tier.HBM and dst in _LOWER:     # offload: HBM -> lower tier
                    r.offload_bytes += nbytes

    # Lifecycle feature view consumed by policies.
    def features(self, anchor: int) -> dict:
        observed = self.lifecycle_windows > 0               # unknown != known-zero
        # ...and "was the LAST window incomplete", which `observed` cannot express: once one
        # window lands, `observed` is True forever, so a policy could not tell that tracing
        # has gone blind since. Stale means the movement fields are a floor, not a measurement.
        stale = self.last_window_complete is False
        r = self.records.get(anchor)
        if r is None:
            return {"reload_bytes": 0, "recompute": 0, "reload_latency_ms": 0.0,
                    "was_offloaded": False,
                    "age_s": 0.0,                           # unseen == unpriced == young
                    "lifecycle_observed": observed, "lifecycle_stale": stale}
        return {"reload_bytes": r.reload_bytes, "recompute": r.recompute,
                "reload_latency_ms": r.reload_latency_ms, "was_offloaded": r.was_offloaded,
                "age_s": time.monotonic() - r.born,
                "lifecycle_observed": observed, "lifecycle_stale": stale}
