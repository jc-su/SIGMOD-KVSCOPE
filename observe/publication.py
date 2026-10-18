"""Versioned policy commit and safe-point application.

The policy controller runs outside the serving process. It writes complete policy generations to a
shared-memory mailbox and never touches serving memory. A small hook in the serving process
applies one complete generation between engine steps: it remaps the generation from the rows the
controller observed to the rows of the live batch by request key, and hands it to the block-table
view (``runtime.block_view.DecodeView``) that the next decode step reads.

A generation carries a keep mask over each request's blocks, an audit request for each row, the
request key of each row, and a lease deadline. An audited row is read in full on the next decode
step, so that the instrumented kernel can measure its attention share. The mailbox also carries
what the controller needs from the serving process: the request key and length of each scheduler
row, and a record of the last completed audit, which says when the attention-share statistic is
complete and for which requests.
"""

from __future__ import annotations

import atexit
import fcntl
import hashlib
import mmap
import os
import re
import struct
import threading
import time
from pathlib import Path

import numpy as np

from .reloads import kv_cache_path, kv_cache_ranges, publish_kv_cache

MAGIC = b"EKVCTR11"
VERSION = 11
HEADER = 128
# magic, version, header bytes, seqlock, generation, lease deadline (host monotonic ns), flags,
# rows, width, applied generation, application timestamp, request-context generation, completed
# audit steps, blocks the view has read, blocks that reading every block would have read, started audit steps,
# and audit steps the controller has read. An audit step writes the statistic of every row from the time it
# starts until it completes. While a controller holds a lease, the view starts no audit step until the
# controller has read every completed one (``audit_read``), and the controller copies the statistic of audit
# ``audit_done`` only while ``audit_started`` equals ``audit_done``, before and after its copy (a sequence lock).
_HEADER = struct.Struct("<8sIIQQQIIIQQQQQQQQ")
READ_COUNT_PERIOD = 32  # safe points between two updates of the read counters (each needs a device sync)

F_ROW_MASK = 1 << 0
F_LEASE = 1 << 1
F_AUDIT_ROWS = 1 << 2
F_POLICY_KEYS = 1 << 3
F_RELEASE = 1 << 4


_VLLM_SUFFIX = re.compile(r"-[0-9a-f]{8}$")


def external_id(request_id: str) -> str:
    """The id the client gave a request. vLLM appends a random suffix of 8 hex digits to it."""
    return _VLLM_SUFFIX.sub("", str(request_id))


def request_key(client_id: str) -> int:
    """Stable nonzero 64-bit key of the id a client gave a request.

    A front end that knows its own ids computes the key of each one with this function and attaches its
    metadata to it, such as a weight. The publisher computes the key of an engine id with
    ``engine_request_key``, which removes the suffix the engine appended, so both agree."""
    value = int.from_bytes(hashlib.blake2b(str(client_id).encode(), digest_size=8).digest(), "little")
    return value or 1


def engine_request_key(engine_id: str) -> int:
    """The key of the client's id, from the id the engine uses internally."""
    return request_key(external_id(engine_id))


def mailbox_path(pid: int) -> Path:
    directory = Path(os.environ.get("EKV_CONTROL_DIR", "/dev/shm"))
    return directory / f"ekv_control_{int(pid)}.shm"


def _layout(max_rows: int, row_width: int) -> tuple[dict[str, int], int]:
    sizes = {
        "row_mask": max_rows * row_width,
        "release_mask": max_rows * row_width,
        "audit_rows": max_rows,
        "request_keys": max_rows * 8,
        "request_lengths": max_rows * 4,
        "policy_keys": max_rows * 8,
        "audit_keys": max_rows * 8,
    }
    offsets = {}
    cursor = HEADER
    for name, size in sizes.items():
        offsets[name] = cursor
        cursor += size
    return offsets, cursor


class ControlMailbox:
    """Fixed-size shared-memory mailbox with nonblocking safe-point reads."""

    def __init__(
        self, path: str | os.PathLike, *, create: bool = False, max_rows: int = 0, row_width: int = 0
    ):
        self.path = Path(path)
        if create:
            if min(max_rows, row_width) <= 0:
                raise ValueError("mailbox dimensions must be positive")
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
            self.offsets, total = _layout(max_rows, row_width)
            os.ftruncate(self.fd, total)
            self.mm = mmap.mmap(self.fd, total, access=mmap.ACCESS_WRITE)
            self.max_rows, self.row_width = int(max_rows), int(row_width)
            self._pack(
                seq=0,
                generation=0,
                lease=0,
                flags=0,
                applied_generation=0,
                applied_ns=0,
                context_generation=0,
                audit_done=0,
                read_blocks=0,
                full_blocks=0,
                audit_started=0,
                audit_read=0,
            )
        else:
            self.fd = os.open(self.path, os.O_RDWR)
            total = os.fstat(self.fd).st_size
            self.mm = mmap.mmap(self.fd, total, access=mmap.ACCESS_WRITE)
            # A writer packs the header under the exclusive lock, and struct.pack_into zero-fills the header before
            # it writes the fields, so a read without the lock can see rows 0 (measured: a run on SGLang failed with
            # "control mailbox size 18512 != expected 128" while the controller renewed its lease).
            fcntl.flock(self.fd, fcntl.LOCK_SH)
            try:
                magic, version, header, _seq, _gen, _lease, _flags, rows, width, *_ = _HEADER.unpack_from(
                    self.mm, 0
                )
            finally:
                fcntl.flock(self.fd, fcntl.LOCK_UN)
            if magic != MAGIC or version != VERSION or header != HEADER:
                raise RuntimeError(f"invalid control mailbox {self.path}")
            self.max_rows, self.row_width = rows, width
            self.offsets, expected = _layout(rows, width)
            if total != expected:
                raise RuntimeError(f"control mailbox size {total} != expected {expected}")

    @classmethod
    def create(cls, path, max_rows: int, row_width: int):
        return cls(path, create=True, max_rows=max_rows, row_width=row_width)

    def _pack(
        self,
        *,
        seq: int,
        generation: int,
        lease: int,
        flags: int,
        applied_generation: int,
        applied_ns: int,
        context_generation: int,
        audit_done: int,
        read_blocks: int,
        full_blocks: int,
        audit_started: int,
        audit_read: int,
    ) -> None:
        _HEADER.pack_into(
            self.mm,
            0,
            MAGIC,
            VERSION,
            HEADER,
            int(seq),
            int(generation),
            int(lease),
            int(flags),
            self.max_rows,
            self.row_width,
            int(applied_generation),
            int(applied_ns),
            int(context_generation),
            int(audit_done),
            int(read_blocks),
            int(full_blocks),
            int(audit_started),
            int(audit_read),
        )

    def header(self):
        return _HEADER.unpack_from(self.mm, 0)

    def _counters(self) -> dict[str, int]:
        """Every header counter that a writer must carry over when it changes only some of them."""
        (
            _m,
            _v,
            _h,
            seq,
            generation,
            lease,
            flags,
            _rows,
            _width,
            applied_generation,
            applied_ns,
            context_generation,
            audit_done,
            read_blocks,
            full_blocks,
            audit_started,
            audit_read,
        ) = self.header()
        return {
            "seq": seq,
            "generation": generation,
            "lease": lease,
            "flags": flags,
            "applied_generation": applied_generation,
            "applied_ns": applied_ns,
            "context_generation": context_generation,
            "audit_done": audit_done,
            "read_blocks": read_blocks,
            "full_blocks": full_blocks,
            "audit_started": audit_started,
            "audit_read": audit_read,
        }

    def _read(self, name: str, dtype, count: int) -> np.ndarray:
        start = self.offsets[name]
        return np.frombuffer(self.mm[start : start + count * np.dtype(dtype).itemsize], dtype=dtype).copy()

    def _write(self, name: str, array: np.ndarray) -> None:
        raw = np.ascontiguousarray(array).view(np.uint8)
        start = self.offsets[name]
        self.mm[start : start + raw.nbytes] = raw.tobytes()

    def status(self) -> dict[str, int]:
        """Return a coherent snapshot of the commit and application counters."""
        fcntl.flock(self.fd, fcntl.LOCK_SH)
        try:
            return {key: int(value) for key, value in self._counters().items()}
        finally:
            fcntl.flock(self.fd, fcntl.LOCK_UN)

    def request_context(self) -> tuple[int, np.ndarray, np.ndarray]:
        """Return the context generation, and the request key and length in blocks of each scheduler row."""
        fcntl.flock(self.fd, fcntl.LOCK_SH)
        try:
            return (
                int(self._counters()["context_generation"]),
                self._read("request_keys", np.uint64, self.max_rows),
                self._read("request_lengths", np.uint32, self.max_rows),
            )
        finally:
            fcntl.flock(self.fd, fcntl.LOCK_UN)

    def audit_record(self) -> tuple[int, int, np.ndarray]:
        """Return the numbers of completed and of started audit steps, and the key of each row audited at the
        last completed one."""
        fcntl.flock(self.fd, fcntl.LOCK_SH)
        try:
            counters = self._counters()
            return (
                int(counters["audit_done"]),
                int(counters["audit_started"]),
                self._read("audit_keys", np.uint64, self.max_rows),
            )
        finally:
            fcntl.flock(self.fd, fcntl.LOCK_UN)

    def close(self, unlink: bool = False) -> None:
        self.mm.close()
        os.close(self.fd)
        if unlink:
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass


class MailboxWriter:
    """Controller-side transaction builder for atomically committed policy generations."""

    def __init__(self, path):
        self.box = ControlMailbox(path)
        self.pending: dict[str, np.ndarray | int] = {}
        self.flags = 0
        self.last_generation = 0

    def ack_audit(self, done: int) -> None:
        """Tell the serving process that the controller has read the audits up to ``done``."""
        box = self.box
        fcntl.flock(box.fd, fcntl.LOCK_EX)
        try:
            box._pack(**{**box._counters(), "audit_read": int(done)})
        finally:
            fcntl.flock(box.fd, fcntl.LOCK_UN)

    def stage_rows(
        self, keep: np.ndarray, audit: np.ndarray, policy_keys: np.ndarray, release: np.ndarray
    ) -> None:
        """Stage the keep mask, the audit request, and the blocks to release of every row, and the
        request key each row was computed for. A release is permanent: the engine frees the block."""
        rows, width = self.box.max_rows, self.box.row_width
        keep = np.ascontiguousarray(keep, dtype=np.uint8)
        audit = np.ascontiguousarray(audit, dtype=np.uint8)
        keys = np.ascontiguousarray(policy_keys, dtype=np.uint64)
        release = np.ascontiguousarray(release, dtype=np.uint8)
        if (
            keep.shape != (rows, width)
            or release.shape != (rows, width)
            or audit.shape != (rows,)
            or keys.shape != (rows,)
        ):
            raise ValueError(
                f"keep {keep.shape}, release {release.shape}, audit {audit.shape}, keys {keys.shape}"
                " do not match the mailbox"
            )
        self.pending.update(
            {"row_mask": keep, "release_mask": release, "audit_rows": audit, "policy_keys": keys}
        )
        self.flags |= F_ROW_MASK | F_AUDIT_ROWS | F_POLICY_KEYS | F_RELEASE

    def stage_lease(self, deadline_ns: int) -> None:
        self.pending["lease"] = int(deadline_ns)
        self.flags |= F_LEASE

    def commit(self) -> int:
        if not self.flags:
            return 0
        box = self.box
        fcntl.flock(box.fd, fcntl.LOCK_EX)
        try:
            counters = box._counters()
            box._pack(**{**counters, "seq": counters["seq"] + 1, "flags": 0})
            for name in ("row_mask", "release_mask", "audit_rows", "policy_keys"):
                staged = self.pending.get(name)
                if isinstance(staged, np.ndarray):
                    box._write(name, staged)
            generation = counters["generation"] + 1
            lease = int(self.pending.get("lease", counters["lease"]))
            box._pack(
                **{
                    **counters,
                    "seq": counters["seq"] + 2,
                    "generation": generation,
                    "lease": lease,
                    "flags": self.flags,
                }
            )
        finally:
            fcntl.flock(box.fd, fcntl.LOCK_UN)
        self.pending.clear()
        self.flags = 0
        self.last_generation = int(generation)
        return 1

    def close(self) -> None:
        self.box.close()


class ScheduledPublisher:
    """Serving-process side: applies committed generations to a ``DecodeView`` between steps."""

    def __init__(self, path, view):
        self.box = ControlMailbox(path)
        self.view = view
        self.generation = 0
        self.applications = 0
        self._request_ids: tuple[str, ...] = ()
        self._request_lengths: tuple[int, ...] = ()
        self._audits_marked = 0
        self._safe_points = 0
        view.on_audit_start = self.mark_audit_start  # the view starts an audit step in the attention build
        view.audit_gate = self.audit_allowed  # and asks first whether it may
        view.before_build = self.drain_before_build  # and takes the newest generation right before it builds

    def drain_before_build(self) -> None:
        """Apply the newest generation in the attention build, where the view's lease is checked."""
        global _publisher_cpu_s
        start = time.thread_time()
        with _publisher_lock:
            self.drain()
        _publisher_cpu_s += time.thread_time() - start

    def _current_keys(self, rows: int) -> np.ndarray:
        keys = np.zeros(rows, dtype=np.uint64)
        for index, request_id in enumerate(self._request_ids[:rows]):
            if request_id:
                keys[index] = engine_request_key(request_id)
        return keys

    @staticmethod
    def _row_remap(policy_keys: np.ndarray, current_keys: np.ndarray) -> dict[int, int]:
        """Return current-row -> policy-row for request identities present in both batches."""
        source = {int(key): index for index, key in enumerate(policy_keys) if int(key)}
        return {
            index: source[int(key)]
            for index, key in enumerate(current_keys)
            if int(key) and int(key) in source
        }

    def drain(self) -> bool:
        """Apply the newest complete generation, if there is one. Returns whether one was applied."""
        box = self.box
        try:
            fcntl.flock(box.fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        try:
            counters = box._counters()
            seq1, generation, flags = counters["seq"], counters["generation"], counters["flags"]
            if seq1 & 1 or generation == self.generation:
                return False
            rows, width = box.max_rows, box.row_width
            mask = (
                box._read("row_mask", np.uint8, rows * width).reshape(rows, width)
                if flags & F_ROW_MASK
                else None
            )
            release = (
                box._read("release_mask", np.uint8, rows * width).reshape(rows, width)
                if flags & F_RELEASE
                else None
            )
            audit = box._read("audit_rows", np.uint8, rows) if flags & F_AUDIT_ROWS else None
            policy_keys = box._read("policy_keys", np.uint64, rows) if flags & F_POLICY_KEYS else None
            lease = counters["lease"]
            if box._counters()["seq"] != seq1:
                return False
        finally:
            fcntl.flock(box.fd, fcntl.LOCK_UN)
        if (mask is not None or audit is not None or release is not None) and policy_keys is None:
            raise RuntimeError("row state without the request keys it was computed for cannot be attributed")
        if policy_keys is not None:
            mapping = self._row_remap(policy_keys, self._current_keys(rows))
            if mask is not None:
                remapped = np.ones((rows, width), dtype=np.uint8)  # rows with no counterpart read every block
                for current, policy in mapping.items():
                    remapped[current] = mask[policy]
                self.view.write_row_controls(remapped)
            if release is not None:
                to_release = np.zeros(
                    (rows, width), dtype=np.uint8
                )  # rows with no counterpart release nothing
                for current, policy in mapping.items():
                    to_release[current] = release[policy]
                self._release(self.view.write_release(to_release))
            if audit is not None:
                requested = np.zeros(rows, dtype=bool)
                for current, policy in mapping.items():
                    requested[current] = bool(audit[policy])
                self.view.write_audit_rows(requested)
        if flags & F_LEASE:
            self.view.write_lease_deadline(int(lease))
        applied_ns = time.monotonic_ns()
        # Acknowledge only after the view holds the generation. Keep a newer controller generation
        # that arrived meanwhile.
        fcntl.flock(box.fd, fcntl.LOCK_EX)
        try:
            current = box._counters()
            box._pack(**{**current, "applied_generation": generation, "applied_ns": applied_ns})
        finally:
            fcntl.flock(box.fd, fcntl.LOCK_UN)
        self.generation = generation
        self.applications += 1
        return True

    def _release(self, released: list[tuple[int, int]]) -> None:
        """Ask the engine to free the blocks the view has just stopped reading."""
        if not released:
            return
        from runtime.release import queue_release

        by_request: dict[str, list[int]] = {}
        for row, block in released:
            by_request.setdefault(self._request_ids[row], []).append(block)
        for request_id, blocks in by_request.items():
            queue_release(request_id, blocks)

    def update_requests(self, request_ids, request_lengths=None, *, advance: bool = False) -> bool:
        """Publish the runtime's current scheduler-row to request-key association.

        The comparison occurs before hashing or locking, so a steady decode step pays only for
        constructing and comparing a short tuple. Controllers join these opaque keys with front-end
        policy metadata; no request text enters the mailbox.
        """
        current = tuple("" if value is None else str(value) for value in request_ids)
        lengths_current = tuple(
            max(0, int(value)) for value in (() if request_lengths is None else request_lengths)
        )
        unchanged = current == self._request_ids and (
            not lengths_current or lengths_current == self._request_lengths
        )
        if unchanged and not advance:
            return False
        keys = lengths = None
        if not unchanged:
            keys = np.zeros(self.box.max_rows, dtype=np.uint64)
            for index, request_id in enumerate(current[: self.box.max_rows]):
                if request_id:
                    keys[index] = engine_request_key(request_id)
            lengths = np.zeros(self.box.max_rows, dtype=np.uint32)
            for index, value in enumerate(lengths_current[: self.box.max_rows]):
                lengths[index] = value
        box = self.box
        fcntl.flock(box.fd, fcntl.LOCK_EX)
        try:
            counters = box._counters()
            for name, array in (("request_keys", keys), ("request_lengths", lengths)):
                if array is not None:
                    box._write(name, array)
            box._pack(**{**counters, "context_generation": counters["context_generation"] + 1})
        finally:
            fcntl.flock(box.fd, fcntl.LOCK_UN)
        self._request_ids = current
        if lengths_current:
            self._request_lengths = lengths_current
        return True

    def reassign_changed_requests(self, request_ids) -> bool:
        """The scheduler changed which request some rows hold: carry each request's state to its row.

        Call before ``update_requests``, while this publisher still holds the previous batch. An audit
        that the previous batch completed is first finished and published in that batch's rows and keys:
        its statistic was written in those rows, and once rows move, a later publication would give the
        statistic of one request to another."""
        self.view.finish_audit()
        self.mark_audit()
        current = tuple("" if value is None else str(value) for value in request_ids)
        width = min(max(len(current), len(self._request_ids)), self.box.max_rows)
        before = {request_id: row for row, request_id in enumerate(self._request_ids) if request_id}
        moved: dict[int, int] = {}
        new: list[int] = []
        empty: list[int] = []
        for row in range(width):
            now = current[row] if row < len(current) else ""
            if now == (self._request_ids[row] if row < len(self._request_ids) else ""):
                continue
            if not now:
                empty.append(row)
            elif now in before:
                moved[row] = before[now]
            else:
                new.append(row)
        if not (moved or new or empty):
            return False
        self.view.reassign_rows(moved, new, empty)
        return True

    def audit_allowed(self) -> bool:
        """Whether the view may start an audit step now: when no controller holds a lease, or when the controller
        has read every completed audit, whose statistic the new step would overwrite."""
        if not self.view.active():
            return True
        fcntl.flock(self.box.fd, fcntl.LOCK_SH)
        try:
            read = self.box._counters()["audit_read"]
        finally:
            fcntl.flock(self.box.fd, fcntl.LOCK_UN)
        return read >= self.view.audits_completed

    def mark_audit_start(self) -> None:
        """Publish that an audit step has started, before the device runs it and writes the statistic."""
        box = self.box
        fcntl.flock(box.fd, fcntl.LOCK_EX)
        try:
            box._pack(**{**box._counters(), "audit_started": self.view.audits_started})
        finally:
            fcntl.flock(box.fd, fcntl.LOCK_UN)

    def mark_audit(self) -> bool:
        """Record the request key of each row of the last completed audit, if a new one finished."""
        done = self.view.audits_completed
        if done == self._audits_marked:
            return False
        keys = np.where(self.view.last_audit_rows, self._current_keys(self.box.max_rows), 0).astype(np.uint64)
        box = self.box
        fcntl.flock(box.fd, fcntl.LOCK_EX)
        try:
            box._write("audit_keys", keys)
            box._pack(**{**box._counters(), "audit_done": done})
        finally:
            fcntl.flock(box.fd, fcntl.LOCK_UN)
        self._audits_marked = done
        return True

    def mark_reads(self) -> bool:
        """Publish the view's read counters every ``READ_COUNT_PERIOD`` safe points.

        The counters live on the device, so reading them waits for the step in flight. The period
        spreads that wait, and a reader sees counts that are at most that many steps old."""
        self._safe_points += 1
        if self._safe_points % READ_COUNT_PERIOD:
            return False
        read, full = self.view.reads()
        box = self.box
        fcntl.flock(box.fd, fcntl.LOCK_EX)
        try:
            box._pack(**{**box._counters(), "read_blocks": read, "full_blocks": full})
        finally:
            fcntl.flock(box.fd, fcntl.LOCK_UN)
        return True

    def close(self, unlink: bool = False) -> None:
        self.box.close(unlink=unlink)


_publisher_lock = threading.Lock()
_publisher: ScheduledPublisher | None = None
_publisher_cpu_s = 0.0
_publisher_calls = 0


def _ensure_current_publisher(runner) -> ScheduledPublisher:
    """Create the process's view and mailbox from the engine's own dimensions, once."""
    global _publisher
    if _publisher is not None:
        return _publisher
    from runtime.block_view import attach_audit_plane

    view = attach_audit_plane(runner)
    path = mailbox_path(os.getpid())
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        ControlMailbox.create(path, view.rows, view.width).close()
    except FileExistsError:
        # Reuse only a compatible mailbox for this live process. A recycled pid can otherwise
        # inherit a dead worker's dimensions or an older layout.
        try:
            existing = ControlMailbox(path)
            compatible = (existing.max_rows, existing.row_width) == (view.rows, view.width)
            existing.close()
        except (OSError, RuntimeError):
            compatible = False
        if not compatible:
            path.unlink(missing_ok=True)
            ControlMailbox.create(path, view.rows, view.width).close()
    # where the GPU KV cache lives, so that a controller can tell a reload from host memory in the eBPF copies
    publish_kv_cache(os.getpid(), kv_cache_ranges(runner.kv_caches))
    _publisher = ScheduledPublisher(path, view)
    return _publisher


def drain_current_process(runner) -> bool:
    """Apply one generation at the vLLM worker's model-execution safe point."""
    global _publisher_cpu_s, _publisher_calls
    start = time.thread_time()
    with _publisher_lock:
        publisher = _ensure_current_publisher(runner)
        applied = publisher.drain()
        publisher.mark_audit()
        publisher.mark_reads()
        _publisher_calls += 1
    _publisher_cpu_s += time.thread_time() - start
    return applied


def publish_current_requests(runner, request_ids, request_lengths=None, *, advance: bool = False) -> bool:
    """Update the row-to-request association after vLLM updates its input batch."""
    global _publisher_cpu_s
    start = time.thread_time()
    with _publisher_lock:
        changed = _ensure_current_publisher(runner).update_requests(
            request_ids, request_lengths, advance=advance
        )
    _publisher_cpu_s += time.thread_time() - start
    return changed


def reassign_changed_requests(runner, request_ids) -> bool:
    """Carry each request's policy to the scheduler row it now holds."""
    global _publisher_cpu_s
    start = time.thread_time()
    with _publisher_lock:
        changed = _ensure_current_publisher(runner).reassign_changed_requests(request_ids)
    _publisher_cpu_s += time.thread_time() - start
    return changed


def policy_update_stats() -> dict[str, float | int | str]:
    applied = 0 if _publisher is None else _publisher.applications
    return {
        "calls": _publisher_calls,
        "applied_policy_updates": applied,
        "cpu_seconds": _publisher_cpu_s,
        "mailbox": str(mailbox_path(os.getpid())),
    }


def _close_current_publisher() -> None:
    global _publisher
    if _publisher is not None:
        _publisher.close(unlink=True)
        kv_cache_path(os.getpid()).unlink(missing_ok=True)
        _publisher = None


atexit.register(_close_current_publisher)


def _publish_updated_batch(runner, scheduler_output, *, advance: bool) -> bool:
    """Export the persistent batch that the next attention launch will consume."""
    input_batch = getattr(runner, "input_batch", None)
    request_ids = getattr(input_batch, "req_ids", None)
    if request_ids is None:
        return False
    num_requests = int(getattr(input_batch, "num_reqs", len(request_ids)))
    current_ids = request_ids[:num_requests]
    reassign_changed_requests(runner, current_ids)
    computed = getattr(input_batch, "num_computed_tokens_cpu", None)
    block_size = max(1, min((int(value) for value in getattr(runner, "_kernel_block_sizes", ())), default=1))
    scheduled = getattr(scheduler_output, "num_scheduled_tokens", {})
    lengths = (
        None
        if computed is None
        else [
            (int(computed[index]) + int(scheduled.get(request_id, 0)) + block_size - 1) // block_size
            for index, request_id in enumerate(current_ids)
        ]
    )
    mark_final_prefill(runner, current_ids, computed, scheduled)
    return publish_current_requests(runner, current_ids, lengths, advance=advance)


def mark_final_prefill(runner, request_ids, computed, scheduled) -> None:
    """With a prefill audit, mark the rows whose next step computes the last tokens of their prompt."""
    with _publisher_lock:
        view = _ensure_current_publisher(runner).view
        if not view.prefill_audit:
            return
        if computed is None:
            raise RuntimeError("a prefill audit needs the batch's computed-token counts")
        prompts = runner.input_batch.num_prompt_tokens
        final = [
            int(computed[index])
            < int(prompts[index])
            <= int(computed[index]) + int(scheduled.get(request_id, 0))
            for index, request_id in enumerate(request_ids)
        ]
        view.write_final_prefill(np.array(final, dtype=bool))


def install_vllm_safe_point() -> int:
    """Install policy application at supported vLLM model-execution boundaries.

    The hook runs immediately after vLLM updates its persistent batch and before input preparation
    or attention: it resets rows whose request changed, exports the exact request order and logical
    KV length used by the upcoming launch, then remaps and applies a completed controller
    generation by request identity.

    It also installs the decode view, which must exist before the engine captures CUDA graphs.
    """
    from runtime.block_view import install_decode_view

    install_decode_view()
    patched = 0
    classes = []
    for module_name in ("vllm.v1.worker.gpu_model_runner", "vllm.v1.worker.gpu.model_runner"):
        try:
            module = __import__(module_name, fromlist=["GPUModelRunner"])
            classes.append(module.GPUModelRunner)
        except (ImportError, AttributeError):
            continue
    advance_context_each_step = os.environ.get("EKV_REQUEST_CONTEXT_EVERY_STEP") == "1"
    for cls in classes:
        original_update = getattr(cls, "_update_states", None)
        if original_update is None or getattr(original_update, "_ekv_scheduled_publication", False):
            continue

        def wrapped_update(self, *args, __original_update=original_update, **kwargs):
            result = __original_update(self, *args, **kwargs)
            scheduler_output = args[0] if args else kwargs.get("scheduler_output")
            _publish_updated_batch(self, scheduler_output, advance=advance_context_each_step)
            drain_current_process(self)
            return result

        wrapped_update._ekv_scheduled_publication = True  # ty: ignore[unresolved-attribute]  # marks the hook
        cls._update_states = wrapped_update
        patched += 1
    return patched
