"""The KV-induced latency model — eKV's attribution contribution, as executable spec.

The question no existing surface answers: of a user-visible latency, how much was
*caused* by KV-cache decisions, and which decisions? This module defines that latency
per request, folds it into comparable causal paths, and attributes the tail.

WELL-DEFINEDNESS UNDER CONTINUOUS BATCHING (the crux). Naively, "this prefetch cost
380ms" is ill-posed: under continuous batching the prefetch overlaps other requests'
compute, so a wall-clock figure has to decide "whose compute hid this" — and there is
no sound answer. eKV sidesteps that entirely by attributing each latency signal AT ITS
SOURCE, where it is already request-scoped, so no overlap arbitration is ever needed:

  • async_load  — the request was parked in WAITING_FOR_REMOTE_KVS and not scheduled
                  until its KV finished loading (BlockingEvent.duration). This is the
                  request's OWN un-schedulable interval — ground truth from the
                  scheduler, not an estimate. The many-to-many of a shared block is
                  resolved by the scheduler itself: only the request that had to WAIT
                  is parked; others that later reuse the now-resident block pay nothing.
  • inline_load — a connector that fills synchronously (no park) stalls the request's
                  step in place; the latency is the host-paid load latency on that step
                  (the H2D Copy cost attributed to the request by the fuser). A given
                  reload is EITHER parked (async) OR inline (sync), never both, so the
                  two never double-count.
  • recompute   — an evicted block recomputed instead of reloaded: GPU cycles on the
                  request's own tokens (a COMPUTE KVEdge with a request). Supplied
                  only when the evict→recompute linkage is available; never inferred.

Each term is a wall-clock (or cycle→ns) quantity charged to exactly one request by
construction. Per-request latency is their sum; the fold groups it into causal paths and
reports each path's share of the tail (the requests with the highest attributed latency), because
the tail —
not the mean — is what SLOs are written against.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Literal

from .events import BlockingEvent, KVEdge
from .types import Nanoseconds, Op, RequestId, Tier

CausalPath = Literal["async_load", "inline_load", "recompute"]
_PATHS: tuple[CausalPath, ...] = ("async_load", "inline_load", "recompute")


@dataclass(frozen=True, slots=True)
class RequestLatency:
    """KV-induced latency charged to one request, split by causal path. Every
    field is a request-scoped wall-clock (ns); ``total`` is their sum."""
    request_id: RequestId
    async_load: Nanoseconds = Nanoseconds(0)
    inline_load: Nanoseconds = Nanoseconds(0)
    recompute: Nanoseconds = Nanoseconds(0)
    n_blocks_loaded: int = 0
    n_block_events: int = 0

    @property
    def total(self) -> Nanoseconds:
        return Nanoseconds(self.async_load + self.inline_load + self.recompute)

    @property
    def by_path(self) -> dict[CausalPath, Nanoseconds]:
        return {"async_load": self.async_load, "inline_load": self.inline_load,
                "recompute": self.recompute}


@dataclass(frozen=True, slots=True)
class PathLatency:
    """One causal path's contribution across a run: total latency it caused, how many
    requests it touched, and — the load-bearing number — its share of the tail."""
    path: CausalPath
    total_ns: Nanoseconds
    n_requests: int
    mean_ns: float
    tail_share: float        # fraction of tail-request latency due to this path
    overall_share: float     # fraction of all requests' total latency that is this path


def blocking_from_loads(loads: Iterable[tuple], block_to_request: dict
                        ) -> list[BlockingEvent]:
    """Reconstruct async_load BlockingEvents from the eBPF-captured KV-LOAD transfers —
    so the park is recovered EXACTLY from eBPF lifecycle tracing, with NO scheduler hook.

    The scheduler's WAITING_FOR_REMOTE_KVS interval is not a hidden scheduler-state secret:
    it ENTERS when the scheduler kicks off the async KV load and EXITS on finished_recving
    — i.e. it *is* the KV-load transfer. eBPF (kv_copy) times that transfer's
    [enqueue, completion] exactly; attributing it to the request by the blocks it loaded
    (the request<->block tie) yields the request's park. A request parked until ALL its
    loads finished, so per request we take [first enqueue, last completion].

    loads: iterable of (blocks, entered_ns, exited_ns) — one finished KV-load transfer.
    block_to_request: {physical block id -> request id} (from the request<->block tie).

    (The completion->promotion lag is <= one schedule step; for sub-step exactness pass
    the request's first-decode-step time as exited_ns instead of the load completion.)"""
    span: dict = {}   # request -> [min_entered, max_exited, set(blocks), n_loads]
    for blocks, entered, exited in loads:
        reqs = {block_to_request[int(b)] for b in blocks if int(b) in block_to_request}
        for r in reqs:
            s = span.get(r)
            if s is None:
                span[r] = [entered, exited, set(int(b) for b in blocks), 1]
            else:
                s[0] = min(s[0], entered)
                s[1] = max(s[1], exited)
                s[2].update(int(b) for b in blocks)
                s[3] += 1
    return [BlockingEvent(request_id=RequestId(str(r)), entered_at=Nanoseconds(s[0]),
                          exited_at=Nanoseconds(s[1]), n_steps=s[3],
                          blocks=tuple(sorted(s[2])))
            for r, s in span.items()]


def parks_from_schedule(loads: Iterable[tuple], request_blocks: dict, enter_time: dict,
                        window_ns: int = 500_000_000) -> list[BlockingEvent]:
    """Finer, still-zero-touch async-load park attribution. ``blocking_from_loads`` keys
    loads on a static block->request tie and takes [first, last] — which over-aggregates
    every reuse of a SHARED block across the whole run into one giant span (the live diag
    run showed 25s "parks"). Here each reload is instead attributed to the request whose
    ADMISSION it gated: a request R is parked from when its KV reload kicks off until R
    first enters the running batch. Per request we take the reloads of R's blocks that
    COMPLETE in a window just before R's entry (the loads that had to finish for R to be
    scheduled), and set park = [earliest such enqueue, R's entry] — bounding the park to
    one admission episode, not the run.

    This is an APPROXIMATION: the exact un-schedulable interval is the scheduler's
    WAITING_FOR_REMOTE_KVS state; R's batch-entry time stands in for un-park, and the
    window separates distinct admission episodes of a shared block.

    loads: (blocks, enqueue_ns, completion_ns) finished KV-load transfers.
    request_blocks: {request_id -> physical block ids} (the compiler-captured ownership join).
    enter_time: {request_id -> ns it first appeared in a running batch} (input_ids/network
                timeline). window_ns: reloads completing within this window before entry gate it."""
    loads = list(loads)
    evs = []
    for r, blocks in request_blocks.items():
        te = enter_time.get(r)
        bset = {int(b) for b in blocks}
        if te is None or not bset:
            continue
        burst = [(enq, comp, lb) for (lb, enq, comp) in loads
                 if comp is not None and te - window_ns <= comp <= te and bset & {int(b) for b in lb}]
        if not burst:
            continue
        start = min(enq for enq, _, _ in burst)
        used = {int(b) for _, _, lb in burst for b in lb if int(b) in bset}
        evs.append(BlockingEvent(request_id=RequestId(str(r)), entered_at=Nanoseconds(start),
                                 exited_at=Nanoseconds(te), n_steps=len(burst),
                                 blocks=tuple(sorted(used))))
    return evs


# Attribution provenance — a first-class confidence tag on every reconstructed quantity, split
# by ROOT. EXACT: uniquely determined. BOUNDED: a temporal proxy with bounded error. SHARED: a
# cost a KNOWN SET of requests jointly incurred — a shared block reloaded once that gated a
# multi-request admission step blocked ALL of them, so the wait is concurrent, not a coin-flip
# over one owner; per request it is the full span, and the aggregate splits it 1/N to avoid
# double counting. IDENTITY_AMBIGUOUS: which request<->slot, when two requests share their whole
# observable timeline AND no content signal is available — aggregate-HARMLESS (same path).
# ATTRIBUTION_AMBIGUOUS: kept for genuinely undeterminable control-plane ownership (the SHARED
# reclassification removes it from the reload tie).
EXACT = "exact"
BOUNDED = "bounded"
SHARED = "shared"
IDENTITY_AMBIGUOUS = "identity-ambiguous"
ATTRIBUTION_AMBIGUOUS = "attribution-ambiguous"


def attribute_reload_parks(reloads, admission: dict, arrival: dict, request_blocks: dict,
                           step_ns: int = 5_000_000) -> dict:
    """ZERO-TOUCH reload-park attribution by the reload -> NEXT-ADMITTED-USER proxy. No scheduler
    or connector hook: each reload is owned by the request that (a) uses its block and (b) is the
    EARLIEST admitted AFTER the reload completed — the request that actually waited for it. A
    request reusing an already-resident shared block (admitted long after its reload, which an
    earlier request owns) owns ZERO reloads -> KV-park 0. This is the fix for the shared-reload
    over-attribution (req 36): one reload has ONE owner, by admission order, not "every user of
    the block" (charge-by-request over-attributes) nor "max of its completions" (grabs a copy
    meant for someone else). Inputs are all off-process: ``reloads`` = eBPF [(block_id,
    completion_ns)]; ``admission``/``arrival`` = {req -> ns} from OTel (the engine's own export);
    ``request_blocks`` = {req -> block ids} (compiler-captured request ownership). All monotonic.

    ``reloads`` = [(block_id, enqueue_ns, completion_ns)]. Per request: kv_park = the SPAN its
    KV was loading = max(completion) - min(enqueue) over its OWNED reloads (the time it was blocked
    in WAITING_FOR_REMOTE_KVS — NOT completion-minus-arrival, which would fold in the slot wait),
    else 0. provenance: EXACT (no owned reload -> 0, certain), BOUNDED (owned via a clean next-
    after = a temporal proxy, bounded error), SHARED (>1 request admitted within step_ns after the
    reload -> the step was blocked until it, so ALL of them CO-WAITED: each gets the full span, the
    aggregate splits 1/N — a known shared cost, not a coin-flip over one owner). Returns
    {req -> {kv_park_ns, provenance, n_owned}}."""
    block_users: dict = {}
    for r, blks in request_blocks.items():
        for b in blks:
            block_users.setdefault(int(b), []).append(r)
    owned: dict = {}          # req -> [(enqueue_ns, completion_ns) of the reloads it owns]
    shared: set = set()       # co-waiters of a shared reload (a known set, not ambiguous)
    for blk, enq, comp in reloads:
        enq, comp = int(enq), int(comp)
        cands = sorted((admission[r], r) for r in block_users.get(int(blk), [])
                       if r in admission and admission[r] >= comp)
        if not cands:
            continue          # no subsequent user -> reload not on any request's critical path
        adm0, owner = cands[0]
        tied = [r for adm, r in cands if adm - adm0 <= step_ns]   # admitted together => co-waiters
        if len(tied) > 1:
            for r in tied:                # the step was blocked until the reload: ALL of them waited
                owned.setdefault(r, []).append((enq, comp))
                shared.add(r)
        else:
            owned.setdefault(owner, []).append((enq, comp))
    out: dict = {}
    for r in admission:
        ev = owned.get(r, [])
        # the span its KV was loading (bounded by its queue wait), NOT arrival->completion
        park = (min(max(c for _, c in ev) - min(e for e, _ in ev),
                    int(admission[r]) - int(arrival.get(r, admission[r]))) if ev else 0)
        if r in shared:
            prov = SHARED                  # co-waiters of a shared reload: concurrent, known set
        elif ev:
            prov = BOUNDED                 # owned via a clean next-after: temporal proxy
        else:
            prov = EXACT                   # no owned reload -> waited 0 for KV (cache-hit/none)
        out[r] = {"kv_park_ns": park, "provenance": prov, "n_owned": len(ev)}
    return out


def tier_miss_costs(reads, h2d, window_ns: int = 50_000_000) -> dict:
    """Recover the FULL lower-tier miss cost the cudaMemcpy tracing alone cannot see, by
    fusing the eBPF storage-read probe with the H2D copy stream. A tiered KV backend serving a
    block from a LOWER tier (local disk / network file) first READS it into a host buffer
    (``read()``/``pread64``, timed by storage_read) and THEN issues the host->device copy that
    kv_copy times. The tracer sees only the H2D, so a disk-tier miss looks as cheap as a
    DRAM-tier hit. Here each read is joined to the H2D it feeds — the earliest copy enqueued at/
    after the read COMPLETES, SAME-THREAD when the backend loads synchronously (exact pairing),
    else nearest-following within ``window_ns`` (the async-executor path: read on a worker thread,
    H2D on the engine thread — a temporal proxy) — so tier miss cost = disk_read + H2D.

    The disk-read latency itself is EXACT (the syscall duration); only its PAIRING to one specific
    H2D is bounded, so each pair's tier_miss is provenance BOUNDED. The AGGREGATE (count, total and
    percentile read time) is exact regardless of pairing. H2D copies with NO preceding read are
    upper-tier (CPU/pool) hits — cheap, H2D-only — and the contrast IS the tier heterogeneity that
    the copy-only view (uniform H2D) hides. ``reads`` = [{tid, ts_ns, bytes, dur_ns}] (ts_ns = read
    completion); ``h2d`` = [{tid, enqueued_at, cost_ns}] (H2D copies; cost = finished-enqueued).
    Returns {pairs:[{read_dur_ns,h2d_cost_ns,total_ns,join,provenance}], aggregate:{...}}."""
    rds = sorted(reads, key=lambda r: int(r["ts_ns"]))
    copies = sorted(h2d, key=lambda c: int(c["enqueued_at"]))
    taken: set = set()
    pairs: list = []
    for rd in rds:
        rt, rtid = int(rd["ts_ns"]), rd.get("tid")
        best_same = best_any = None
        for i, c in enumerate(copies):
            if i in taken:
                continue
            dt = int(c["enqueued_at"]) - rt
            if dt < 0:
                continue                       # H2D before the read completed: not the one it feeds
            if dt > window_ns:
                break                          # sorted by enqueue -> nothing later is closer
            if best_any is None:
                best_any = i
            if rtid is not None and c.get("tid") == rtid:
                best_same = i                  # synchronous backend: read+H2D on one thread -> exact
                break
        idx = best_same if best_same is not None else best_any
        if idx is None:                        # a read with no following H2D in window (rare)
            pairs.append({"tid": rtid, "read_dur_ns": int(rd["dur_ns"]), "h2d_cost_ns": None,
                          "total_ns": int(rd["dur_ns"]), "join": "unpaired", "provenance": BOUNDED})
            continue
        taken.add(idx)
        c = copies[idx]
        pairs.append({"tid": rtid, "read_dur_ns": int(rd["dur_ns"]), "h2d_cost_ns": int(c["cost_ns"]),
                      "total_ns": int(rd["dur_ns"]) + int(c["cost_ns"]),
                      "join": "same-thread" if idx == best_same else "time-proximity",
                      "provenance": BOUNDED})

    def _pct(xs, q):
        return xs[min(len(xs) - 1, int(len(xs) * q))] if xs else 0
    durs = sorted(int(rd["dur_ns"]) for rd in rds)
    h2d_costs = sorted(int(c["cost_ns"]) for c in copies)
    totals = sorted(p["total_ns"] for p in pairs if p["h2d_cost_ns"] is not None)
    med_read, med_h2d = _pct(durs, 0.5), _pct(h2d_costs, 0.5)
    aggregate = {
        "n_reads": len(rds), "total_read_ns": sum(durs),
        "median_read_ns": med_read, "p99_read_ns": _pct(durs, 0.99),
        "max_read_ns": durs[-1] if durs else 0,
        "n_h2d": len(copies), "median_h2d_ns": med_h2d, "p99_h2d_ns": _pct(h2d_costs, 0.99),
        "n_paired": len(totals),
        "n_same_thread": sum(1 for p in pairs if p["join"] == "same-thread"),
        "n_h2d_cpu_tier": len(copies) - len(taken),   # H2D with no preceding read = upper-tier hit
        "median_miss_ns": _pct(totals, 0.5), "p99_miss_ns": _pct(totals, 0.99),
        "max_miss_ns": totals[-1] if totals else 0,
        # how many x the INVISIBLE disk read is vs the H2D copy that was the only visible part
        "read_dominance": (med_read / med_h2d) if (med_h2d and durs) else None,
    }
    return {"pairs": pairs, "aggregate": aggregate}


def split_admission_delay_pass(otel_reqs, kind: dict, reload_done: dict,
                               clock_offset_ns: int, base=None) -> dict:
    """Cause-split using a per-request KV-acquisition CLASS — no time window.

    EXACT for the no-wait classes: cache_hit / recompute / none -> KV-park = 0 (the KV was already
    resident, or re-derived by the request's OWN compute — neither is a queue wait). The pass
    query length supplies recompute (large) vs read (small) zero-touch, so this side is exact.

    The reload side is only as exact as its ``reload_done`` input, and that input is the catch:
    distinguishing a RELOAD (this request waited for a copy) from a CACHE-HIT on a SHARED block
    (the copy was for another request) is NOT derivable from the transfer stream — both READ the
    KV (small query length); the only difference is a copy that can't be attributed to a specific
    request. So ``reload_done`` must be a TRUE per-request gating-reload completion (e.g. from a
    per-request load tag, not max-of-its-shared-blocks'-completions, which over-attributes a late
    reload done for someone else). With a correct reload_done: reload -> KV-park = (reload_done
    clamped to admission) - arrival; queue_other = total - kv_park.

    ``kind[r]`` in {'cache_hit','recompute','reload','none'}; ``reload_done`` is {request_id ->
    completion_ns (monotonic)}; ``clock_offset_ns`` = CLOCK_REALTIME - CLOCK_MONOTONIC. Returns
    {request_id -> {kv_park_s, queue_total_s, queue_other_s, kind, kv_share}}."""
    norm = base or (lambda x: str(x))
    kd = {norm(k): v for k, v in kind.items()}
    rd = {norm(k): int(v) for k, v in reload_done.items()}
    out: dict = {}
    for r in otel_reqs:
        rid = norm(r.request_id)
        total = float(r.time_in_queue_s)
        k = kd.get(rid, "none")
        if k == "reload" and rid in rd:
            arrival = int(r.t_start_ns) - int(clock_offset_ns)
            park = max(0.0, min(rd[rid] - arrival, int(total * 1e9)) / 1e9)
        else:
            park = 0.0                       # cache-hit / recompute / none -> KV already there / own compute
        out[rid] = {"kv_park_s": park, "queue_total_s": total,
                    "queue_other_s": max(0.0, total - park), "kind": k,
                    "kv_share": (min(park, total) / total) if total > 0 else 0.0}
    return out


def split_admission_delay(blocking, queue_total: dict, base=None) -> dict:
    """Split each request's admission delay into the KV-load PARK (eKV-attributed, from the
    eBPF KV-load reconstruction) and the RESIDUAL queue/priority delay (capacity/scheduling
    — NOT a KV-cache problem). This is the cause-split a raw queue number can't give:

      kv_park[r]     = sum of its BlockingEvent durations (parks_from_schedule / from_loads)
      queue_total[r] = the engine's total queue wait (OTel gen_ai.latency.time_in_queue, s)
      queue_other[r] = max(0, queue_total - kv_park)   # capacity / priority, not KV

    So a request slow because its KV reloaded (high kv_share) is separated from one merely
    queued behind others (high queue_other) — only eBPF (which loads ran during the wait)
    can tell them apart; OTel gives the total, eKV carves out the KV part. ``queue_total``
    is {request_id -> seconds}; ``base`` optionally normalizes ids (OTel 'N' vs engine
    'N-<hash>'). Returns {request_id -> {kv_park_s, queue_total_s, queue_other_s, kv_share}}."""
    norm = base or (lambda x: str(x))
    kv: dict = {}
    for ev in blocking:
        kv[norm(ev.request_id)] = kv.get(norm(ev.request_id), 0.0) + ev.duration / 1e9
    qt = {norm(k): float(v) for k, v in queue_total.items()}
    out: dict = {}
    for r in set(kv) | set(qt):
        park, total = kv.get(r, 0.0), qt.get(r, 0.0)
        other = max(0.0, total - park)
        out[r] = {"kv_park_s": park, "queue_total_s": total, "queue_other_s": other,
                  "kv_share": (min(park, total) / total) if total > 0 else 0.0}
    return out


def attribute_latency(
    blocking: Iterable[BlockingEvent] = (),
    edges: Iterable[KVEdge] = (),
    *,
    recompute_edges: Iterable[KVEdge] | None = None,
) -> dict[RequestId, RequestLatency]:
    """Fold the per-source signals into per-request latency.

    ``blocking``        → async_load (BlockingEvent.duration, by request).
    ``edges``           → inline_load: COPY edges that move CPU→GPU (a reload) carrying
                          a request and a cost. (Edges whose request already parked are
                          excluded automatically — a parked request's reload has no
                          inline cost charged; see module docstring.)
    ``recompute_edges`` → recompute: COMPUTE edges with a request and cost. Optional and
                          never synthesized; supply only real evict→recompute linkage."""
    a: dict[RequestId, Nanoseconds] = defaultdict(lambda: Nanoseconds(0))
    nb: dict[RequestId, int] = defaultdict(int)
    ne: dict[RequestId, int] = defaultdict(int)
    parked: set[RequestId] = set()
    for ev in blocking:
        r = ev.request_id
        a[r] = Nanoseconds(a[r] + ev.duration)
        nb[r] += len(ev.blocks) or ev.n_tokens and 1 or 0
        ne[r] += 1
        parked.add(r)

    il: dict[RequestId, Nanoseconds] = defaultdict(lambda: Nanoseconds(0))
    for e in edges:
        if e.op != Op.COPY or e.request is None or not e.cost:
            continue
        # a reload that stalls the step in place: CPU(DRAM) -> GPU(HBM), and the request
        # did NOT park for it (else the park already counted the wait).
        if (e.src_tier == Tier.DRAM and e.dst_tier == Tier.HBM
                and RequestId(str(e.request)) not in parked):
            r = RequestId(str(e.request))
            il[r] = Nanoseconds(il[r] + e.cost)

    rc: dict[RequestId, Nanoseconds] = defaultdict(lambda: Nanoseconds(0))
    for e in (recompute_edges or ()):
        if e.op == Op.COMPUTE and e.request is not None and e.cost:
            r = RequestId(str(e.request))
            rc[r] = Nanoseconds(rc[r] + e.cost)

    out: dict[RequestId, RequestLatency] = {}
    for r in set(a) | set(il) | set(rc):
        out[r] = RequestLatency(
            request_id=r, async_load=a.get(r, Nanoseconds(0)),
            inline_load=il.get(r, Nanoseconds(0)), recompute=rc.get(r, Nanoseconds(0)),
            n_blocks_loaded=nb.get(r, 0), n_block_events=ne.get(r, 0))
    return out


def fold_paths(latencies: Iterable[RequestLatency], *, tail_pct: float = 0.05) -> dict[CausalPath, PathLatency]:
    """Collapse per-request latency into per-path contributions, including each path's
    share of the TAIL — the ``tail_pct`` fraction of requests with the highest total
    latency. tail_share answers 'what is hurting the worst-off requests' (the SLO question)
    rather than 'what is biggest on average'."""
    hs = list(latencies)
    totals: dict[CausalPath, int] = {p: 0 for p in _PATHS}
    touched: dict[CausalPath, int] = {p: 0 for p in _PATHS}
    grand = 0
    for h in hs:
        grand += h.total
        for p, v in h.by_path.items():
            totals[p] += v
            if v > 0:
                touched[p] += 1

    # Tail requests ranked by total attributed latency.
    ranked = sorted(hs, key=lambda h: h.total, reverse=True)
    k = max(1, int(round(len(ranked) * tail_pct))) if ranked else 0
    tail = ranked[:k]
    tail_total = sum(h.total for h in tail) or 1
    tail_by_path: dict[CausalPath, int] = {p: 0 for p in _PATHS}
    for h in tail:
        for p, v in h.by_path.items():
            tail_by_path[p] += v

    out: dict[CausalPath, PathLatency] = {}
    for p in _PATHS:
        n = touched[p]
        out[p] = PathLatency(
            path=p, total_ns=Nanoseconds(totals[p]), n_requests=n,
            mean_ns=(totals[p] / n) if n else 0.0,
            tail_share=tail_by_path[p] / tail_total,
            overall_share=(totals[p] / grand) if grand else 0.0)
    return out


def summarize(latencies: dict[RequestId, RequestLatency], *, tail_pct: float = 0.05) -> dict:
    """A compact, printable verdict: per-path totals/shares + the tail it dominates."""
    hs = list(latencies.values())
    paths = fold_paths(hs, tail_pct=tail_pct)
    totals = sorted((h.total for h in hs), reverse=True)
    n = len(totals)
    p95 = totals[int(n * 0.05)] if n else 0
    return {
        "n_requests_with_latency": n,
        "total_latency_ms": sum(totals) / 1e6,
        "p95_request_latency_ms": p95 / 1e6,
        "paths": {p: {"total_ms": ph.total_ns / 1e6, "n_requests": ph.n_requests,
                      "mean_ms": ph.mean_ns / 1e6, "tail_share": round(ph.tail_share, 3),
                      "overall_share": round(ph.overall_share, 3)}
                  for p, ph in paths.items() if ph.total_ns > 0},
    }
