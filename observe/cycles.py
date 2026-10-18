"""Per-request MEASURED attention cycles -- the analysis layer over the pass timer.

The pass accumulates, per instrumented kernel VARIANT v (eager prefill, each CUDA-graph decode
bucket), a monotone u64 vector T_v[row]: every CTA atomic-adds its [entry, return] clock64 delta
into the row of the request it served. This module turns those raw accumulators into per-request
attribution with an explicit contract.

The algebra (each is asserted by tests/test_cycles.py):
  * MONOTONE: T_v[row] never decreases, so a window delta d_r(w) = T_v[r](w_end) - T_v[r](w_start)
    is >= 0; a negative delta means the buffers were restarted or snapshots were swapped -- an
    error, never clamped silently.
  * ADDITIVE: atomic add is associative and commutative, so a request's cycles compose exactly
    across its CTAs (q-blocks x head-splits), layers, graph replays, and kernel variants:
    delta-then-sum == sum-then-delta. A variant born inside the window contributes its full
    accumulation (it started at zero inside the window).
  * SHARES partition unity: s_r = d_r / sum(d); invariant to any constant per-CTA instrumentation
    multiplier and to the layer count (both scale every request identically).
  * EXACT PARTITION: attribute_window rescales shares onto the measured window time, so the
    per-request times sum to the window exactly -- only the split carries measurement error, the
    total never does.
  * SLOT STABILITY: rows are batch SLOTS; across compaction a slot's delta mixes two requests.
    Attribution is valid only on windows where the slot -> request map is constant, checked per
    slot by the first-block anchor + grow-only row prefix (stable_slots); unstable slots are
    EXCLUDED loudly, never guessed.

The GPU-architecture contract (why the signal is sound, and what it is NOT):
  * clock64 is per-SM and a CTA never migrates SMs, so its entry->return difference is a valid
    duration on one clock; there is no cross-SM skew inside a CTA's measurement.
  * The duration is SM-RESIDENCY (occupancy), not pure arithmetic work: co-resident CTAs share an
    SM, inflating each other's durations by the contention factor. In homogeneous DECODE -- the
    production-dominant regime, one query token per sequence -- contention is symmetric and the
    SHARES track work; heterogeneous prefill mixes lengths, and a long CTA's uncontended tail
    deflates its marginal cycles (measured live: corr(cycles, causal area) ~= 0.95). Residency is
    itself a defensible COST unit: it is the accelerator time the request occupied, the GPU
    analogue of CPU core-seconds.
  * Cycles are DVFS-invariant only for compute-bound phases; a memory stall is fixed nanoseconds,
    so its cycle count scales with the clock. The production path is therefore shares x measured
    window time (the shares see one clock; the wall time carries the seconds), never raw
    cross-window cycle sums read as time.
  * VISIBILITY: the accumulators are RELAXED GPU-scope atomics to host-mapped memory; the host
    sees a consistent value only after a kernel-completion happens-before. Snapshot only at
    observed completion syncs (the eBPF tracer's synchronize/event-query-ready bracket).
  * Scope: only instrumented (attention) kernels are counted. Non-attention step time (MLP,
    norms, scheduling) is outside this signal and out of scope for KV attribution.
"""
from __future__ import annotations

import array
from dataclasses import dataclass, field

from observe import zero_touch as zt

__all__ = ["CycleSnapshot", "attention_entries", "snapshot", "delta", "total",
           "request_cycles", "shares", "attribute_window", "stable_slots", "rows_match",
           "window_series"]


@dataclass(frozen=True)
class CycleSnapshot:
    """Raw per-variant cumulative cycle vectors. The variant key is kernel@timer_va, so the same
    buffer read twice keys identically and two variants of one kernel stay distinct."""
    rows: dict[str, list[int]] = field(default_factory=dict)


def rows_match(parsed: dict[int, list[int]], reference: dict[int, list[int]]) -> bool:
    """Content-addressed variant selection: a pass entry belongs to the ATTENTION kernel iff the
    block-table rows it captured agree with a reference table (e.g. a prior capture's) on at
    least one common ACTIVE row -- one row's positive prefix equal or prefix-nested (rows only
    grow). Dummy profile tables and foreign kernels cannot reproduce the engine's physical block
    ids. Name-free."""
    for s, ref in reference.items():
        got = parsed.get(s)
        if not got or not ref:
            continue
        gp = [x for x in got if x > 0]
        rp = [x for x in ref if x > 0]
        if gp and rp and (gp[:len(rp)] == rp or rp[:len(gp)] == gp):
            return True
    return False


def attention_entries(pid: int | None = None, kernel: str | None = None,
                      reference_slots: dict[int, list[int]] | None = None) -> list:
    """The pass entries carrying the attention kernel's buffers (capture + timer). Selection is
    content-addressed when a reference table is supplied (rows_match against e.g. a prior
    capture), else by kernel-name substring -- reader-side metadata only; structural detection
    itself stays name-free."""
    out = []
    for b in zt.find_published(pid):
        if not (b.host_va and (b.timer_host_va or b.timer_dev)):
            continue
        if reference_slots is not None:
            parsed = zt.parse_slots(zt.read_proc_mem(b.pid, b.host_va, b.bytes),
                                    b.ncap, b.max_rows)
            if rows_match(parsed, reference_slots):
                out.append(b)
        elif kernel is None or kernel in b.kernel:
            out.append(b)
    return out


def snapshot(entries: list) -> CycleSnapshot:
    """Read every entry's FULL timer vector (zeros included -- a row at 0 now and 5 later needs
    delta 5). Snapshot only after an observed completion sync (see the visibility contract)."""
    rows: dict[str, list[int]] = {}
    for b in entries:
        a = array.array("q")
        raw = (zt.read_dev_mem(b.timer_dev, b.timer_bytes) if b.timer_dev
               else zt.read_proc_mem(b.pid, b.timer_host_va, b.timer_bytes))
        a.frombytes(raw)
        key = b.timer_dev or b.timer_host_va               # device-resident since the PCIe-storm fix
        rows[f"{b.kernel}@{key:#x}"] = list(a[:b.max_rows])
    return CycleSnapshot(rows)


def delta(start: CycleSnapshot, end: CycleSnapshot) -> dict[int, int]:
    """{row -> cycles} spent in the window, summed across variants (delta-then-sum ==
    sum-then-delta by atomic additivity). A variant only in `end` was born inside the window and
    contributes its full accumulation. A NEGATIVE per-row delta violates monotonicity (restarted
    buffers / swapped snapshots) and raises -- it is never clamped silently."""
    out: dict[int, int] = {}
    for key, endv in end.rows.items():
        startv = start.rows.get(key, [])
        for r, e in enumerate(endv):
            s = startv[r] if r < len(startv) else 0
            d = int(e) - int(s)
            if d < 0:
                raise ValueError(f"negative cycle delta ({d}) at row {r} of {key}: "
                                 "snapshots swapped or buffers restarted")
            if d:
                out[r] = out.get(r, 0) + d
    return out


def total(end: CycleSnapshot) -> dict[int, int]:
    """Cumulative rows since birth: delta from the empty snapshot."""
    return delta(CycleSnapshot(), end)


def stable_slots(slots_start: dict[int, list[int]],
                 slots_end: dict[int, list[int]]) -> dict[int, bool]:
    """slot -> did the SAME request hold it across the window? True iff the slot's first block is
    unchanged and the start row is a prefix of the end row (a request's row only grows; a first-
    block change or a non-prefix rewrite is compaction/reuse). Slots absent at either edge are
    unstable. Unstable slots' deltas mix requests -- exclude them, do not apportion."""
    out: dict[int, bool] = {}
    for s in set(slots_start) | set(slots_end):
        a = [x for x in slots_start.get(s, []) if x > 0]
        b = [x for x in slots_end.get(s, []) if x > 0]
        out[s] = bool(a) and bool(b) and a[0] == b[0] and b[:len(a)] == a
    return out


def request_cycles(row_cycles: dict[int, int], slot_blocks: dict[int, list[int]],
                   fb_to_request: dict[int, str] | None = None,
                   stable: dict[int, bool] | None = None) -> dict:
    """Join {row -> cycles} to requests. A row participates only if it holds a REAL block-table
    row (any positive id) -- profile-dummy and graph-padding rows carry cycles but no table and
    are excluded. Identity goes through the content-addressed anchor (zero_touch's
    slot_to_request_via_anchor) when fb_to_request is given, else the slot's own first block
    names the entity (the intrinsic key). `stable` (from stable_slots) drops mixed windows."""
    real = {s: row for s, row in slot_blocks.items() if any(x > 0 for x in row)}
    if stable is not None:
        real = {s: row for s, row in real.items() if stable.get(s)}
    if fb_to_request is not None:
        s2r = zt.slot_to_request_via_anchor(real, fb_to_request)
    else:
        s2r = {s: row[0] for s, row in real.items() if row and row[0] > 0}
    out: dict = {}
    for s, rid in s2r.items():
        c = row_cycles.get(s, 0)
        if c:
            out[rid] = out.get(rid, 0) + c
    return out


def shares(req_cycles: dict) -> dict:
    """Partition of unity over requests; {} stays {} (never divide by zero)."""
    tot = sum(req_cycles.values())
    if tot <= 0:
        return {}
    return {r: c / tot for r, c in req_cycles.items()}


def attribute_window(req_cycles: dict, t_window: float) -> dict:
    """Per-request time over a measured window: s_r * t_window. Sums to t_window EXACTLY (the
    total is the measurement; only the split carries the signal's error). The window must be
    sync-bounded and slot-stable (see delta / stable_slots)."""
    sh = shares(req_cycles)
    return {r: s * float(t_window) for r, s in sh.items()}


def window_series(stream: list, wall_times: list | None = None,
                  fb_to_request: dict[int, str] | None = None) -> dict:
    """The production streaming reduction: fold a captured sequence of sync-bounded snapshots into
    per-window and accumulated per-request cycle attribution.

    ``stream`` is [(CycleSnapshot, slot_blocks), ...] read at successive OBSERVED completion syncs,
    oldest first -- exactly what eBPF tracing brackets (the kv_copy synchronize / event-query
    ready points give the boundaries; the block-table copy at each gives slot_blocks; the timer
    buffer read at each gives the snapshot). Window i spans (stream[i-1], stream[i]):
      * cycles d = delta(prev, cur) -- monotone, so d >= 0 (a violation raises, never clamped);
      * a window is CLEAN iff every slot holding a real row at either edge is stable across it
        (stable_slots: same first-block anchor + grow-only prefix). An unclean window is a
        compaction boundary: the whole batch shifted, so its cycles are reported for the stable
        slots but it is NOT used for wall-time attribution (attributing its wall time only to the
        survivors would misprice them);
      * request_cycles joins d to requests by the content-addressed anchor (own first block, or
        fb_to_request), excluding dummy/pad rows and unstable slots.
    Accumulates per-request cycles over all windows and, when ``wall_times`` (per-window seconds)
    is given, per-request time over CLEAN windows only (each clean window partitions its wall time
    exactly, so the accumulated time is exact over the clean fraction). Returns
    {"windows": [{"cycles": {req: c}, "clean": bool}], "total_cycles": {req: c},
     "total_time": {req: s}, "clean": n_clean, "n": n_windows}."""
    total_cyc: dict = {}
    total_time: dict = {}
    windows: list = []
    n_clean = 0
    for i in range(1, len(stream)):
        (snap0, sb0), (snap1, sb1) = stream[i - 1], stream[i]
        d = delta(snap0, snap1)
        stable = stable_slots(sb0, sb1)
        real = {s for s, r in sb0.items() if any(x > 0 for x in r)} \
            | {s for s, r in sb1.items() if any(x > 0 for x in r)}
        clean = bool(real) and all(stable.get(s) for s in real)
        rc = request_cycles(d, sb1, fb_to_request, stable)
        windows.append({"cycles": rc, "clean": clean})
        for r, c in rc.items():
            total_cyc[r] = total_cyc.get(r, 0) + c
        if clean:
            n_clean += 1
            if wall_times is not None:
                for r, t in attribute_window(rc, wall_times[i - 1]).items():
                    total_time[r] = total_time.get(r, 0.0) + t
    out = {"windows": windows, "total_cycles": total_cyc, "clean": n_clean,
           "n": max(0, len(stream) - 1)}
    if wall_times is not None:
        out["total_time"] = total_time
    return out
