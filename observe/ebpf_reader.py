"""Turn the eBPF ``kv_copy`` JSONL stream into correlated ``Copy`` records.

Two pure steps:

    parse_records(lines)  : raw JSONL  ->  CopyRecord stream  (Enqueue | Bind | Completion)
    correlate(records)    : records    ->  CaptureReport       (copies + integrity diagnostics)

The reader is strict where strictness means correctness, and lenient where being
strict would lie — three distinct axes:

  * SYNTAX / SCHEMA  → raise. Bad JSON, a non-object line, an unknown record *type*
    or completion *mode*, a missing/non-int field, a non-positive size/count: the
    emitter changed or the capture is corrupt, and continuing would pollute results.
  * RELEVANCE        → drop. ``H2H`` is host↔host and never a KV tier movement, so
    it is ignored (not fatal). ``DEFAULT`` (cudaMemcpyDefault) carries no declared
    direction: BOTH addresses are kept and the *fuser* infers offload/load from
    which side is a KV block — the reader never guesses a direction.
  * COMPLETENESS     → tolerate + flag. A completion for an unknown handle
    (attach-mid-flight) or one claiming more copies than were captured (ringbuf
    drops) marks the window incomplete and leaves the affected copies UNCOSTED.
    It never assigns a completion to the oldest unmatched copy — a wrong cost is
    worse than a missing one.
"""
from __future__ import annotations

import json
import struct
from collections import deque
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import assert_never, cast

from .events import Bind, Completion, Copy, CopyRecord, Enqueue, Gap
from .types import (Address, CompletionVia, Direction, EventHandle, Nanoseconds,
                    Pid, SizeBytes, StreamHandle)

_VIA: frozenset[str] = frozenset(("stream-sync", "event-sync", "event-query"))
_KNOWN_DIR: frozenset[str] = frozenset(("H2D", "D2H", "D2D"))     # declared single direction
_AMBIGUOUS_DIR: frozenset[str] = frozenset(("BATCH", "DEFAULT"))  # carry both, fuser resolves
_IGNORED_DIR: frozenset[str] = frozenset(("H2H",))               # never a KV tier movement


# ── parse: raw JSONL -> typed records (strict on syntax/schema) ──────────────
def parse_records(lines: Iterable[str]) -> Iterator[CopyRecord]:
    for lineno, raw in enumerate(lines, 1):
        line = raw.strip()
        if not line:
            continue
        try:
            o = json.loads(line)
        except ValueError as exc:
            raise ValueError(f"invalid copy JSONL at line {lineno}") from exc
        if not isinstance(o, dict):
            raise ValueError(f"copy JSONL line {lineno} is not an object")
        match o.get("type"):
            case "enqueue":
                rec = _enqueue(o, lineno)
                if rec is not None:            # None == ignored direction (H2H)
                    yield rec
            case "bind":
                yield Bind(event=EventHandle(_int(o, "event", lineno)),
                           stream=StreamHandle(_int(o, "stream", lineno)),
                           at=Nanoseconds(_int(o, "t_ns", lineno)))
            case "completion":
                count = _int(o, "n", lineno)
                if count <= 0:
                    raise ValueError(f"completion count must be >0 at line {lineno}")
                yield Completion(handle=_int(o, "handle", lineno),
                                 via=_via(o.get("via"), lineno),
                                 count=count,
                                 at=Nanoseconds(_int(o, "t_ns", lineno)))
            case "gap":
                # COMPLETENESS marker from the collector's supervisor (a restart): capture was
                # interrupted. Tolerate + flag (do not raise) -- t_ns is best-effort.
                yield Gap(at=Nanoseconds(_int(o, "t_ns", lineno)) if "t_ns" in o else Nanoseconds(0),
                          reason=str(o.get("reason", "")))
            case "meta":
                # The request-identity record (EKV_META=1: host->device boundary metadata, the
                # request-identity input) legitimately sharing the lifecycle stream. Known and not
                # ours to correlate: skip, never a parse error -- one meta line must not void a
                # window of copies. Strictness stays for types the collector does not emit.
                continue
            case other:
                raise ValueError(f"unknown copy record type {other!r} at line {lineno}")


def _int(o: dict, key: str, lineno: int) -> int:
    if key not in o or o[key] is None:
        raise ValueError(f"missing {key!r} at line {lineno}")
    try:
        return int(o[key])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid integer {key!r} at line {lineno}: {o[key]!r}") from exc


def _via(v: object, lineno: int) -> CompletionVia:
    if v not in _VIA:
        raise ValueError(f"unknown completion via {v!r} at line {lineno}")
    return cast(CompletionVia, v)


def _enqueue_from(direction, *, src: int, dst: int, nbytes: int, at: Nanoseconds,
                  stream: StreamHandle, pid: Pid | None) -> Enqueue | None:
    """Build an Enqueue applying the relevance policy — SHARED by the JSON and binary
    parsers so they can't drift: ignore H2H, carry both addresses for BATCH/DEFAULT
    (no direction guess), fix the device side for H2D/D2H/D2D, raise on truly-unknown.
    The collector always emits both src and dst, so both are available either way."""
    if nbytes <= 0:
        raise ValueError(f"enqueue bytes must be >0 (got {nbytes})")
    if direction in _IGNORED_DIR:
        return None                                   # host↔host: not a KV tier move
    d = cast(Direction, direction)
    if direction in _AMBIGUOUS_DIR:
        return Enqueue(Address(src), SizeBytes(nbytes), d, at, stream, pid, Address(dst))
    if direction in _KNOWN_DIR:
        return Enqueue(Address(src if direction == "D2H" else dst), SizeBytes(nbytes),
                       d, at, stream, pid)
    raise ValueError(f"unknown copy direction {direction!r}")


def _enqueue(o: dict, lineno: int) -> Enqueue | None:
    """JSON enqueue -> shared policy. A truly unknown direction is a schema error."""
    return _enqueue_from(
        o.get("kind"), src=_int(o, "src", lineno), dst=_int(o, "dst", lineno),
        nbytes=_int(o, "bytes", lineno), at=Nanoseconds(_int(o, "t_ns", lineno)),
        stream=StreamHandle(_int(o, "stream", lineno)),
        pid=Pid(_int(o, "pid", lineno)) if o.get("pid") is not None else None)


# ── correlate: records -> CaptureReport (tolerant of incomplete windows) ─────
@dataclass(slots=True)
class _Pending:
    """A mutable copy builder used only during correlation; frozen into a ``Copy``
    once its completion (if any) is known."""
    addr: Address
    nbytes: SizeBytes
    direction: Direction
    enqueued_at: Nanoseconds
    pid: Pid | None
    other_addr: Address | None
    finished_at: Nanoseconds | None = None
    via: CompletionVia | None = None

    def settle(self, at: Nanoseconds, via: CompletionVia) -> None:
        if self.finished_at is None:
            self.finished_at, self.via = at, via

    def freeze(self) -> Copy:
        return Copy(self.addr, self.nbytes, self.direction, self.enqueued_at,
                    self.finished_at, self.via, self.pid, self.other_addr)


@dataclass(frozen=True, slots=True)
class CaptureReport:
    """Correlation result: the costed copies plus capture-integrity diagnostics.

    ``complete`` is False when the window saw completions it could not match —
    attach-mid-flight (unknown handle) or ringbuf drops (more completed than were
    captured). Those copies are left UNCOSTED, never mis-attributed, so a downstream
    flame can mark the window partial rather than trust a fabricated latency."""
    copies: list[Copy] = field(default_factory=list)
    ambiguous_default: int = 0       # cudaMemcpyDefault copies carried (direction inferred by membership)
    unmatched_completions: int = 0   # completion for a handle never enqueued/bound (mid-flight)
    overflowed_completions: int = 0  # completion claimed more copies than captured (drops)
    gaps: int = 0                    # collector restarts seen in the window (records lost across each)

    @property
    def complete(self) -> bool:
        """No capture-integrity ERRORS — no unmatched/overflowed completions, and no collector
        restart (gap) interrupted the window. This does NOT mean every copy is costed: a copy
        still in flight at window end is pending, not an error. For "all copies costed" use
        `fully_costed`."""
        return (self.unmatched_completions == 0 and self.overflowed_completions == 0
                and self.gaps == 0)

    @property
    def uncosted(self) -> int:
        """Copies with no completion yet (pending at window end) — expected for a
        live window, not in itself an integrity failure."""
        return sum(1 for c in self.copies if not c.is_complete)

    @property
    def fully_costed(self) -> bool:
        """Every copy has a measured completion AND there were no integrity errors —
        the right check for an offline/post-hoc capture that should be whole."""
        return self.complete and self.uncosted == 0


class Correlator:
    """Correlation state, so a LIVE tailer can carry it across windows.

    A stateless correlator is right for a whole-file read and wrong for a tailed stream: a
    copy enqueued in window N routinely completes in window N+1, where its handle looks like
    one that was never enqueued — counted `unmatched`, marking a perfectly healthy window
    incomplete. (Measured on a committed 8B run: 808 of 1150 reported lifecycle gaps were
    exactly this shape, `1 unmatched, 1 overflowed`.) Keeping `pend` across windows removes
    the false signal. The copy itself was already emitted, uncosted, in its own window, so a
    late settle changes nothing downstream — only the spurious incompleteness goes away.

    Drained handles stay as tombstones on purpose: a second sync on an idle stream must not
    read as unmatched either. `max_handles` bounds the state (streams/events number in the
    tens; the cap is a defensive bound, oldest-first)."""

    def __init__(self, max_handles: int = 4096) -> None:
        self._pend: dict[int, deque[_Pending]] = {}
        self._max_handles = max_handles

    def feed(self, records: Iterable[CopyRecord]) -> CaptureReport:
        pend = self._pend                       # stream/event handle -> builders
        order: list[_Pending] = []              # THIS window's copies only
        ambiguous_default = unmatched = overflowed = gaps = 0
        for r in records:
            match r:
                case Gap():
                    gaps += 1               # collector restarted: this window lost records
                case Enqueue(addr, nbytes, direction, at, stream, pid, other):
                    if direction == "DEFAULT":
                        ambiguous_default += 1
                    b = _Pending(addr, nbytes, direction, at, pid, other)
                    order.append(b)
                    pend.setdefault(stream, deque()).append(b)
                case Bind(event, stream, _):
                    q = pend.get(stream)
                    if q:                   # re-key the stream's copies onto the event
                        pend.setdefault(event, deque()).extend(q)
                        q.clear()
                case Completion(handle, via, count, at):
                    q = pend.get(handle)
                    if q is None:           # handle never seen: attach-mid-flight / drop
                        unmatched += 1      # window incomplete; affected copies stay uncosted
                        continue
                    if not q:               # known handle already drained (e.g. moved to event)
                        continue
                    if count > len(q):      # more completed than captured: dropped enqueues
                        overflowed += 1
                    for _ in range(min(count, len(q))):
                        q.popleft().settle(at, via)
                case _:
                    assert_never(r)
        if len(pend) > self._max_handles:       # defensive bound, oldest handles first
            for k in list(pend)[:len(pend) - self._max_handles]:
                del pend[k]
        return CaptureReport(
            copies=[b.freeze() for b in order],
            ambiguous_default=ambiguous_default,
            unmatched_completions=unmatched,
            overflowed_completions=overflowed,
            gaps=gaps)


def correlate(records: Iterable[CopyRecord]) -> CaptureReport:
    """Fold the record stream into immutable Copies (enqueue order preserved),
    pairing completions to EXACTLY the copies they finished, keyed by handle.

    One-shot: a fresh `Correlator`, which is what an offline/whole-file read wants. A live
    tailer must hold a `Correlator` instead — see its docstring."""
    return Correlator().feed(records)


# ── convenience entry points ────────────────────────────────────────────────
def read_copies_report(lines: Iterable[str],
                       correlator: Correlator | None = None) -> CaptureReport:
    """Parse + correlate, returning copies AND capture-integrity diagnostics. Pass a
    `correlator` to correlate ACROSS calls (a live tailer); omit it for a one-shot read."""
    return (correlator or Correlator()).feed(parse_records(lines))


def read_copies(lines: Iterable[str]) -> list[Copy]:
    """Parse + correlate into just the ``Copy`` list (diagnostics discarded)."""
    return read_copies_report(lines).copies


def read_copies_file(path: str) -> list[Copy]:
    with open(path) as f:
        return read_copies(f)


def read_copies_file_report(path: str) -> CaptureReport:
    with open(path) as f:
        return read_copies_report(f)


# ── binary transport (EKV_BINARY) — no JSON in the hot path ──────────────────
# Wire: a 12-byte header [magic "eKV1", u32 version, u32 record_size] then fixed
# kv_copy_evt records. The record layout (kv_copy.h): 6×u32 then 7×u64, 80 bytes,
# naturally packed. The relevance/correlation policy is shared with the JSON path
# via _enqueue_from + correlate, so the two formats can't diverge.
_BIN_HDR = struct.Struct("<4sII")
_BIN_REC = struct.Struct("<6I7Q")    # type,kind,api,pid,tid,n | ts,bytes,first,wait,handle,dst,src
_KIND_DIR = {0: "H2H", 1: "H2D", 2: "D2H", 3: "D2D", 4: "DEFAULT", 9: "BATCH"}
_VIA_BIN = {0: "stream-sync", 1: "event-sync", 2: "event-query"}


def parse_records_binary(stream) -> Iterator[CopyRecord]:
    """Parse the binary transport from a binary stream (file 'rb' / sys.stdin.buffer).
    Validates the self-describing header (catches struct drift); a short final read
    (collector stopped mid-record) ends cleanly."""
    hdr = stream.read(_BIN_HDR.size)
    if len(hdr) < _BIN_HDR.size:
        return
    magic, _ver, rec_size = _BIN_HDR.unpack(hdr)
    if magic != b"eKV1":
        raise ValueError(f"bad eKV binary header magic {magic!r}")
    if rec_size != _BIN_REC.size:
        raise ValueError(f"record size {rec_size} != expected {_BIN_REC.size} "
                         "(kv_copy_evt layout drift)")
    while True:
        buf = stream.read(rec_size)
        if len(buf) < rec_size:
            break
        (typ, kind, _api, pid_, _tid, n,
         ts, nbytes, _first, _wait, handle, dst, src) = _BIN_REC.unpack(buf)
        if typ == 0:                        # KV_ENQUEUE
            direction = _KIND_DIR.get(kind)
            if direction is None:
                raise ValueError(f"unknown copy kind {kind} in binary record")
            rec = _enqueue_from(direction, src=src, dst=dst, nbytes=nbytes,
                                at=Nanoseconds(ts), stream=StreamHandle(handle),
                                pid=Pid(pid_) if pid_ else None)
            if rec is not None:
                yield rec
        elif typ == 2:                      # KV_BIND
            yield Bind(event=EventHandle(handle), stream=StreamHandle(dst),
                       at=Nanoseconds(ts))
        elif typ == 1:                      # KV_COMPLETION
            via = _VIA_BIN.get(kind)
            if via is None:
                raise ValueError(f"unknown completion kind {kind} in binary record")
            if n <= 0:
                raise ValueError(f"completion count must be >0 (got {n})")
            yield Completion(handle=handle, via=cast(CompletionVia, via), count=n,
                             at=Nanoseconds(ts))
        else:
            raise ValueError(f"unknown record type {typ} in binary stream")


def read_copies_binary(stream) -> list[Copy]:
    return correlate(parse_records_binary(stream)).copies


def read_copies_binary_file(path: str) -> list[Copy]:
    with open(path, "rb") as f:
        return correlate(parse_records_binary(f)).copies
