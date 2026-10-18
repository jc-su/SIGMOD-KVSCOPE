"""The live eBPF ``kv_copy`` stream, fused into ``KVEdge``s that
``observe.ledger`` joins to requests on block identity.

Two constructions:

``make_ebpf_source(path, base, blocks, stride)``
    manual geometry (the caller knows the pool), returns a zero-arg callable -> ObservationBatch.

``AutoEbpfSource(pid, workdir)``
    attaches the collectors, identifies the KV pool as the eBPF-observed allocation containing
    the compiler-published kv_ptr (an independent signal;
    never "the largest allocation"), and calibrates the block stride as the modal size of
    pool-internal transfers (engines move KV a block at a time). Zero geometry configuration.

Coverage is first-class: every call returns an ObservationBatch whose ``complete`` flag and
``reason`` distinguish "the tracer observed nothing" (known zero) from "the tracer could not
observe" (unknown). A failed window still never raises into the control loop (automatic keep-all), but it
is no longer silent: the batch says so, and the loop surfaces it.
"""
from __future__ import annotations

import os
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .blockmap import BlockAddressMap, identify_kv_pool
from .ebpf_reader import Correlator, read_copies_report
from .fuse import fuse_copies

# Widening bound (below): a defensive upper bound. 2^20 units is far past any single-GPU KV pool
# at supported KV block sizes -- 64 GiB at vLLM's 64 KiB/block, 16 M tokens at SGLang's B=1 -- so it never
# binds in practice; it stops one stray lattice-aligned address from materializing a huge map.
MAX_UNITS = int(os.environ.get("EKV_MAX_UNITS", 1 << 20))

_ADDR_RE = re.compile(r'"(?:dst|src)":(\d+)')
_ENQUEUE = '"type":"enqueue"'


def scan_max_unit(lines, base: int, stride: int) -> int:
    """Highest lattice-aligned unit index appearing in a raw batch, without a full parse.
    Used to widen the extent BEFORE fusing, so the very window that first reaches a high unit
    is attributed rather than dropped. Returns -1 when the batch touches the lattice nowhere.

    ENQUEUES ONLY. The same stream carries request-identity `meta` records (the input_ids
    H2D copy, EKV_META), whose `dst` is not a KV address at all; one of those landing on the
    lattice by coincidence would widen the extent to a unit that does not exist. The ledger's
    join would reject the bogus ids downstream, so this was never a correctness risk -- but
    widening is the one place where reading a record we do not understand costs something, so
    read only the records that are by definition KV traffic."""
    top = -1
    for ln in lines:
        if _ENQUEUE not in ln:
            continue
        for m in _ADDR_RE.finditer(ln):
            off = int(m.group(1)) - base
            if off >= 0 and off % stride == 0:
                u = off // stride
                if u > top:
                    top = u
    return top


@dataclass
class ObservationBatch:
    """One eBPF lifecycle-observation window, with its coverage."""
    edges: list = field(default_factory=list)
    complete: bool = True          # False: this window may have missed events (see reason)
    reason: str = ""               # why incomplete: "stream not up", "parse error", "calibrating"
    parse_errors: int = 0
    # ── drop accounting (why an observed copy produced no edge) ───────────────────────────
    unresolved: int = 0            # not this pool's traffic: correctly ignored
    beyond_extent: int = 0         # ON the pool lattice but past the map's extent: a REAL hole
    max_unit: int = -1             # highest lattice unit seen this window (-1: none)

    def __iter__(self):            # ledger.ingest_ebpf iterates edges; keep that working
        return iter(self.edges)


class _TailFuser:
    """Tail complete JSONL lines from a growing file; correlate, resolve, fuse.

    Correlation state is HELD ACROSS WINDOWS (`Correlator`): an enqueue completing in the next
    window used to report there as an unmatched completion and mark an otherwise healthy window
    incomplete."""

    def __init__(self, copy_path: str | Path):
        self.path = str(copy_path)
        self.offset = 0
        self.correlator = Correlator()

    def lines(self) -> list[str] | None:
        """New complete lines since last call; None when the stream is not readable yet."""
        try:
            with open(self.path, "rb") as f:
                f.seek(self.offset)
                buf = f.read()
        except OSError:
            return None
        if not buf:
            return []
        # only consume COMPLETE lines; leave a partial trailing line for the next call
        nl = buf.rfind(b"\n")
        if nl < 0:
            return []
        self.offset += nl + 1
        return buf[:nl].decode("utf-8", "replace").splitlines()

    def fuse(self, lines: list[str], addr_map: BlockAddressMap) -> ObservationBatch:
        if not lines:
            return ObservationBatch()
        try:
            rep = read_copies_report(lines, self.correlator)  # copies + integrity counters
            diag: dict = {}
            edges = fuse_copies(rep.copies, addr_map, diag=diag)
            beyond = diag.get("beyond_extent", 0)
            reasons = []
            if not rep.complete:
                reasons.append(f"capture integrity: {rep.unmatched_completions} unmatched, "
                               f"{rep.overflowed_completions} overflowed, {rep.gaps} gaps")
            if beyond:
                # NOT an integrity failure -- an ATTRIBUTION failure, and the one that used to be
                # invisible: real pool traffic we saw and could not name, because the extent was
                # short. Incomplete is the honest verdict; AutoEbpfSource then widens.
                reasons.append(f"{beyond} copies beyond the derived extent "
                               f"(map covers {addr_map.extent}, saw unit "
                               f"{diag.get('max_unit', -1)})")
            return ObservationBatch(edges=edges,
                                    complete=rep.complete and not beyond,
                                    reason="; ".join(reasons),
                                    unresolved=diag.get("unresolved", 0),
                                    beyond_extent=beyond,
                                    max_unit=diag.get("max_unit", -1))
        except Exception as e:                              # keep all, but report the failed batch
            return ObservationBatch(complete=False, reason=f"parse error: {e}", parse_errors=1)


def make_ebpf_source(copy_path: str, base: int, num_blocks: int, block_stride: int):
    """Manual geometry: block i occupies [base + i*stride, base + (i+1)*stride)."""
    addr_map = BlockAddressMap.from_alloc(base, num_blocks, block_stride)
    tail = _TailFuser(copy_path)

    def source() -> ObservationBatch:
        lines = tail.lines()
        if lines is None:
            return ObservationBatch(complete=False, reason="stream not up")
        return tail.fuse(lines, addr_map)

    return source


def merge_regions(allocs):
    """VA-adjacent captured allocations merge into regions: pooled allocators split one logical
    reservation across many cuMemAlloc calls, and the tensor layout is only legible at region
    granularity. Returns sorted (base, end) pairs."""
    norm = sorted(((int(a["addr"]), int(a["size"])) if isinstance(a, dict)
                   else (int(a[0]), int(a[1]))) for a in allocs)
    regions: list[list[int]] = []
    for a, sz in norm:
        if regions and a <= regions[-1][1]:
            regions[-1][1] = max(regions[-1][1], a + sz)
        else:
            regions.append([a, a + sz])
    return [(a, b) for a, b in regions]


def derive_geometry(allocs, kv_ptr: int | None, copy_records: list[dict],
                    seed_stride: int | None = None):
    """The pure derivation (unit-testable, no attach). Facts, each from an independent signal:

    * REGIONS: VA-adjacent captured allocations merge (merge_regions); the KV pool is the merged
      region containing the pass-published kv_ptr, plus every region of the same size (per-layer
      tensors pack into same-sized regions).
    * STRIDE: the modal size of pool-internal transfers (engines move KV a block at a time).
    * IDS: exact ONLY on the kv_ptr lattice -- kv_ptr IS a tensor base the kernel was handed, so
      id = (addr - kv_ptr)/stride for lattice-aligned addresses after it. Region bases are NOT
      tensor bases (measured: tensors sit at aligned offsets inside pooled regions), so dividing
      region offsets mis-numbers blocks. A block moved in any layer moves in kv_ptr's layer too,
      so per-block coverage survives; bytes are one layer's share (stated, not scaled).

    The lattice is also the CHECK on the stride heuristic: a modal size that no pool transfer
    lands on (relative to kv_ptr) is rejected loudly rather than mis-deriving identity. Validated
    live: derived ids span exactly the engine's block space, with id 0 absent because the engine
    reserves block 0 as the never-moved null block.

    Returns ((regions, stride, kv_ptr), "") where regions is a tuple of (base, end) pairs with
    the kv_ptr region first, or (None, reason)."""
    if not kv_ptr:
        return None, "pool not identified (no kv_ptr published)"
    regions = merge_regions(allocs)
    anchor = next(((a, b) for a, b in regions if a <= int(kv_ptr) < b), None)
    if anchor is None:
        # LATE-ATTACH FALLBACK: the collectors missed the pool's cudaMalloc (attached after
        # engine init -- the daemon's normal situation). The lattice needs no allocation record:
        # candidate strides are the modal transfer sizes, and the winning stride is the one the
        # most endpoints land on RELATIVE TO kv_ptr; the region is the observed lattice extent.
        # Early attach still yields the full region family (better containment); this fallback
        # is lattice-scoped by construction and self-limits to the published tensor.
        kvp = int(kv_ptr)
        pts = [int(x) for c in copy_records if c.get("bytes")
               for x in (c.get("dst"), c.get("src"))
               if x is not None and 0 <= int(x) - kvp < (1 << 31)]
        # Candidate strides: modal transfer sizes THAT COULD BE A BLOCK. Tiny scalar copies
        # (4/8/16-byte flags and lengths) divide every address, so a raw most-hits vote always
        # elects them over the true stride (measured live: an 8B run with 95K genuine 64KiB
        # reloads derived stride=8 and went geometry-blind). A per-block KV span is KBs at any
        # KV block size and head geometry; nothing below 4KiB is a block move.
        cands = [sz for sz, _ in Counter(int(c["bytes"]) for c in copy_records
                                         if c.get("bytes")).most_common(8)
                 if sz >= 4096][:3]
        best, hits = None, []
        for sz in cands:
            h = [x for x in pts if (x - kvp) % sz == 0]
            if len(h) > len(hits):
                best, hits = sz, h
        if best is None or len(hits) < 4:
            return None, ("pool not identified (kv_ptr in no captured region; too little "
                          "lattice-landing traffic to bootstrap without the allocation)")
        span = max(hits) - kvp + best
        return ((( kvp, kvp + span),), best, kvp), ""
    span = anchor[1] - anchor[0]
    family = [anchor] + [r for r in regions if r != anchor and r[1] - r[0] == span]
    # TRAFFIC-BEARING widening: same-size membership assumes one allocation granularity, but a
    # foreign mover's restores land wherever the engine's layer tensors actually live
    # (measured: LMCache torch-path chunks 0.97 GiB above kv_ptr, in regions of another size).
    # A captured region receiving >=2 bulk (>=4KiB) copy endpoints holds KV by evidence; it
    # joins the family. Election/ids still bind on the lattice, so a false member can only
    # produce ids the ledger's join rejects -- never a wrong attribution.
    fam = set(family)
    bulk = [int(x) for c in copy_records if c.get("bytes") and int(c["bytes"]) >= 4096
            for x in (c.get("dst"), c.get("src")) if x is not None]
    for r in regions:
        if r in fam:
            continue
        n = sum(1 for x in bulk if r[0] <= x < r[1])
        if n >= 2:
            family.append(r)
            fam.add(r)

    def inside(x):
        if x is None:
            return False
        x = int(x)
        return any(a <= x < b for a, b in family)

    # A compiler-supplied stride takes precedence over election outright: the pass KNOWS
    # bytes-per-block, election only ever infers it. Ids bind by floor division downstream.
    if seed_stride and sum(1 for c in copy_records if c.get("bytes")
                           for x in (c.get("dst"), c.get("src")) if inside(x)) >= 4:
        return (tuple(tuple(r) for r in family), int(seed_stride), int(kv_ptr)), ""

    endpoints = [int(x) for c in copy_records if c.get("bytes")
                 for x in (c.get("dst"), c.get("src")) if inside(x)]
    sizes = Counter(int(c["bytes"]) for c in copy_records
                    if c.get("bytes") and (inside(c.get("dst")) or inside(c.get("src"))))
    if not sizes:
        # With a compiler-supplied stride, ids come
        # from FLOOR DIVISION on the kv_ptr lattice -- no alignment requirement. Movers that
        # write at sub-block (token) offsets (measured: LMCache torch path, 2KiB steps) are
        # attributable exactly this way; election-by-modulo can never accept them.
        kvp0 = int(kv_ptr)
        pts0 = [int(x) for c in copy_records if c.get("bytes")
                for x in (c.get("dst"), c.get("src"))
                if x is not None and 0 <= int(x) - kvp0 < (1 << 31)]
        if seed_stride and len(pts0) >= 4:
            return (tuple(tuple(r) for r in family), int(seed_stride), kvp0), ""
        # No copy touches the captured region family -- but a mover that only writes INTO the
        # pool (measured: LMCache's torch path, chunk-granular H2D restores) still exposes the
        # stride as the lattice its into-pool endpoints land on relative to kv_ptr. Same
        # election as the late-attach fallback, same >=4KiB block floor, same loud rejection.
        kvp = int(kv_ptr)
        pts = [int(x) for c in copy_records if c.get("bytes")
               for x in (c.get("dst"), c.get("src"))
               if x is not None and 0 <= int(x) - kvp < (1 << 31)]
        cands = [sz for sz, _ in Counter(int(c["bytes"]) for c in copy_records
                                         if c.get("bytes")).most_common(8)
                 if sz >= 4096][:3]
        best, hits = None, []
        for sz in cands:
            h = [x for x in pts if (x - kvp) % sz == 0]
            if len(h) > len(hits):
                best, hits = sz, h
        if best is not None and len(hits) >= 4:
            return (tuple(tuple(r) for r in family), best, kvp), ""
        return None, "calibrating (no pool-internal transfers observed yet)"
    stride = sizes.most_common(1)[0][0]
    on_lattice = [x for x in endpoints
                  if int(kv_ptr) <= x < anchor[1] and (x - int(kv_ptr)) % stride == 0]
    if not on_lattice:
        return None, (f"stride {stride} rejected: no pool transfer lands on the published "
                      f"kv_ptr lattice")
    return (tuple(tuple(r) for r in family), stride, int(kv_ptr)), ""


class AutoEbpfSource:
    """Self-configuring eBPF lifecycle source. Attaches on construction; call for batches; close() detaches.
    Until the pool is identified and the stride calibrated, batches are complete=False with the
    reason, and NO line is consumed (the first fused window includes the traffic that calibrated
    it)."""

    def __init__(self, pid: int, workdir: str | Path, kv_ptr_reader=None, units: int = 0):
        from .ebpf_attach import EbpfSession
        from .zero_touch import read_pass_kv_ptr
        self.pid = pid
        self.work = Path(workdir)
        self.work.mkdir(parents=True, exist_ok=True)
        self.session = EbpfSession(pid, self.work)
        self.session.__enter__()
        self._kv_ptr = kv_ptr_reader or (lambda: read_pass_kv_ptr(pid))
        self.tail = _TailFuser(self.session.kv_copy)
        self.addr_map: BlockAddressMap | None = None
        self.geometry: tuple | None = None                  # (regions, stride, kv_ptr) once calibrated
        self.units_hint = int(units)                        # operator-known pool size (exact from tick 0)
        self._kvp = self._stride = 0
        self._units = 0
        self.widenings = 0                                  # times the extent grew (health signal)
        self.over_cap = 0                                   # widen requests refused by MAX_UNITS

    def _allocs(self) -> list[dict]:
        import json
        try:
            with open(self.session.kv_alloc) as f:
                return [json.loads(ln) for ln in f if '"addr"' in ln]
        except OSError:
            return []

    def _try_calibrate(self) -> str:
        import json
        recs = []
        try:
            with open(self.session.kv_copy) as f:           # peek; the tail offset is NOT consumed
                for ln in f:
                    if ln.startswith("{"):
                        try:
                            recs.append(json.loads(ln))
                        except ValueError:                  # partial trailing line mid-write
                            pass
        except OSError:
            pass
        try:
            kvp = self._kv_ptr()
        except OSError as e:                                # cross-process read needs privileges
            return f"kv_ptr unreadable ({e}); run the daemon privileged or grant CAP_SYS_PTRACE"
        geo, why = derive_geometry(self._allocs(), kvp, recs)
        if geo is None:
            return why
        regions, stride, kv_ptr = geo
        self.geometry = geo
        # ids are exact ONLY on the published tensor's lattice (see derive_geometry): tile from
        # kv_ptr to the end of its region. Pool traffic off this lattice never becomes a block id.
        nblk = (regions[0][1] - kv_ptr) // stride
        self._kvp, self._stride = int(kv_ptr), int(stride)
        self._set_extent(max(nblk, self.units_hint))
        return ""

    def _set_extent(self, units: int) -> None:
        """(Re)build the address map at `units` KV units and keep the published geometry honest."""
        self._units = int(units)
        self.addr_map = BlockAddressMap.from_alloc(self._kvp, self._units, self._stride)
        if self.geometry:                                    # health must report the REAL extent
            _, stride, kvp = self.geometry
            self.geometry = (((kvp, kvp + self._units * stride),), stride, kvp)

    def _widen_for(self, lines) -> None:
        """The derived extent is a LOWER BOUND, not a fact — so keep it honest as traffic arrives.

        `derive_geometry`'s late-attach fallback sizes the lattice from whatever traffic had
        landed when calibration ran (the daemon's normal case: it attaches after the engine has
        already allocated its pool, so there is no cudaMalloc record to read the true size from).
        Freezing that first guess is a silent, unbounded attribution loss: on a committed 8B run
        the extent locked at 82 units while the pool used 300, so 218 of 299 units — 73% of the
        movement — resolved to nothing for the rest of the run, with every self-check still
        passing. An address landing ON the lattice past the current extent is itself proof the
        pool is bigger, so widen to it. Bounded by MAX_UNITS; refusals are counted, not silent."""
        if not lines or not self._units or not self._stride:
            return
        top = scan_max_unit(lines, self._kvp, self._stride)
        if top < self._units:
            return
        if top + 1 > MAX_UNITS:
            self.over_cap += 1
            return
        self._set_extent(top + 1)
        self.widenings += 1

    def __call__(self) -> ObservationBatch:
        if self.addr_map is None:
            why = self._try_calibrate()
            if self.addr_map is None:
                return ObservationBatch(complete=False, reason=why)
        lines = self.tail.lines()
        if lines is None:
            return ObservationBatch(complete=False, reason="stream not up")
        self._widen_for(lines)              # widen BEFORE fusing: this window is attributed too
        return self.tail.fuse(lines, self.addr_map)

    def close(self):
        self.session.__exit__(None, None, None)
