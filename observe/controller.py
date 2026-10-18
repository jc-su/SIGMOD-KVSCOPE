"""The decide stage: turn measured attention shares into a keep mask for every live request.

Each tick the controller reads which request occupies each scheduler row and how many blocks it
holds, folds in the attention shares of the latest audit step, splits a read budget across the
requests by marginal attention share (``observe.allocation``), and commits one complete generation
to the mailbox. A request with no audit evidence yet keeps all its blocks and does not count
against the budget, so the controller never trims a request it has not measured.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from observe.allocation import Objective, Request, allocate
from observe.publication import MailboxWriter
from observe.tile_shares import Aggregate, normalize_tile_lse, read_tile_rows


@dataclass(frozen=True)
class Settings:
    """What the controller is asked to do."""

    budget_fraction: float  # fraction of the measured requests' blocks to keep, in (0, 1]
    objective: Objective = "sum"
    recent_blocks: int = 4  # newest blocks always kept, in addition to the first block
    lease_seconds: float = 0.5  # how long a committed generation stays valid without renewal
    refresh_blocks: int = 0  # audit a request again when it has this many unmeasured blocks; 0 audits once
    audit_hold_ticks: int = 10  # wait this many ticks before asking again for an audit not yet seen
    share_decay: float = 0.5  # weight of the older shares when a new audit arrives
    # Hold the median engine step time near this many milliseconds by moving the budget fraction between
    # ``min_budget_fraction`` and 1, starting from ``budget_fraction``. None keeps the fraction fixed.
    target_step_ms: float | None = None
    min_budget_fraction: float = 0.15
    # Leave out of the step-time target every step during which a KV reload from host memory was in flight
    # (measured by the eBPF copy collector). The read fraction cannot shorten a reload, so such a step is not
    # a sample of what the fraction controls. Needs a step-time target and a reload source.
    reload_aware: bool = False
    # Ask the engine to free a block once it has been left out in this many consecutive audits of its
    # request. 0 never frees. A freed block cannot be read again, so this must be more than 1.
    release_after: int = 0
    # Apply the budget only to requests longer than this many blocks; a shorter request reads in full and does
    # not count against the budget. With a step-time target, this holds the step time of a shared batch by
    # trimming only its long requests. 0 applies the budget to every request.
    cap_above_blocks: int = 0

    def __post_init__(self) -> None:
        if self.release_after < 0 or self.release_after == 1:
            raise ValueError("release_after must be 0 (never) or at least 2 audits")
        if not 0.0 < self.budget_fraction <= 1.0:
            raise ValueError("the budget fraction must be in (0, 1]")
        if not 0.0 <= self.share_decay < 1.0 or self.recent_blocks < 0:
            raise ValueError("invalid controller settings")
        if self.refresh_blocks < 0 or self.audit_hold_ticks <= 0 or self.cap_above_blocks < 0:
            raise ValueError("invalid controller settings")
        if self.target_step_ms is not None and self.target_step_ms <= 0.0:
            raise ValueError("the step-time target must be positive")
        if not 0.0 < self.min_budget_fraction <= 1.0:
            raise ValueError("the smallest budget fraction must be in (0, 1]")
        if self.reload_aware and self.target_step_ms is None:
            raise ValueError("a reload-aware target needs a step-time target")


class ShareSource(Protocol):
    """Anything that reports the attention share per (row, block) of the live rows.

    ``row_blocks`` is the length in blocks of each live row."""

    def read(self, live_rows: int, row_blocks: np.ndarray) -> np.ndarray: ...


class ReloadSource(Protocol):
    """Anything that reports the KV reloads from host memory of the engine (``observe.reloads.ReloadTracker``).

    ``watermark_ns`` is the time up to which every reload that started has been read. Times are
    ``time.monotonic_ns``, the clock of the mailbox's application stamps."""

    watermark_ns: int

    def poll(self) -> None: ...

    def in_flight_ns(self, start_ns: int, end_ns: int) -> int: ...

    def forget_before(self, ns: int) -> None: ...


class TileShareSource:
    """Reads the attention shares of the live rows from the pass's statistic block."""

    def __init__(
        self, cuda: object, entry: dict[str, int | str], pid: int, aggregate: Aggregate = "mean"
    ) -> None:
        import os

        self.aggregate = aggregate
        self.cuda = cuda
        self.rows = int(entry["max_rows"])
        self.width = int(entry["ncap"])
        self.group_rows = int(entry["tile_lse_group_rows"])
        self.max_layers = int(entry["tile_lse_layers"])
        self.tile_blocks = int(entry["attn_share_tile_blocks"])
        local = pid == os.getpid()
        self.base = int(entry["tile_lse_dev"]) if local else cuda.ipc_open(str(entry["tile_lse_ipc"]))  # ty: ignore[unresolved-attribute]
        self.table = int(entry["layer_table_dev"]) if local else cuda.ipc_open(str(entry["layer_table_ipc"]))  # ty: ignore[unresolved-attribute]

    def read(self, live_rows: int, row_blocks: np.ndarray) -> np.ndarray:
        """Attention share per (row, block) for the first ``live_rows`` rows; NaN without evidence.

        A row of 0 blocks has no evidence and is not copied from the GPU, so a caller that needs a few rows
        of a large batch pays for those rows only."""
        raw = self.cuda.d2h(self.table, self.max_layers * 8)  # ty: ignore[unresolved-attribute]
        layers = int(np.count_nonzero(np.frombuffer(raw, dtype=np.uint64)))
        if layers == 0:
            raise RuntimeError("the layer table is empty: the serving process has not filled it")
        shares = np.full((live_rows, self.width), np.nan)
        wanted = np.flatnonzero(row_blocks[:live_rows] > 0)
        if wanted.size == 0:
            return shares
        lse = read_tile_rows(
            self.cuda,
            self.base,
            self.max_layers,
            layers,
            self.rows,
            self.width,
            self.group_rows,
            wanted,
            self.tile_blocks,
        )
        shares[wanted] = normalize_tile_lse(lse, row_blocks[wanted], self.tile_blocks, self.aggregate)
        return shares


class Controller:
    """Commits a budgeted keep mask for the live requests of one serving process."""

    def __init__(
        self,
        writer: MailboxWriter,
        source: ShareSource | None,
        settings: Settings,
        weight: Callable[[int], float] | None = None,
        scores: Callable[[int], np.ndarray] | None = None,
        reloads: ReloadSource | None = None,
    ) -> None:
        """``source`` reports measured shares. ``scores`` replaces them with a fixed rule, used by
        baselines that need no measurement: it maps a request's block count to one score per block.
        Exactly one of the two is given. ``reloads`` is given exactly when the target is reload-aware."""
        if (source is None) == (scores is None):
            raise ValueError("give a share source or a fixed scoring rule, not both and not neither")
        if settings.reload_aware != (reloads is not None):
            raise ValueError("a reload-aware target needs a reload source, and only it uses one")
        if scores is not None and settings.release_after:
            raise ValueError(
                "releasing blocks needs measured shares, which a fixed scoring rule does not have"
            )
        self.writer = writer
        self.source = source
        self.fixed_scores = scores
        self.settings = settings
        self.weight = weight or (lambda _key: 1.0)
        self.shares: dict[int, np.ndarray] = {}  # request key -> smoothed share per block
        self.measured: dict[int, int] = {}  # request key -> blocks the latest audit covered
        self.allocation: list[
            tuple[int, int, int]
        ] = []  # (request key, blocks, blocks kept) of the last tick
        self.streak: dict[int, np.ndarray] = {}  # request key -> audits in a row that left each block out
        self.release: dict[int, np.ndarray] = {}  # request key -> blocks asked to be freed (permanent)
        self._fresh: set[int] = set()  # requests whose evidence arrived since the last tick
        self._audits_seen = 0
        self._asked_tick: dict[int, int] = {}  # request key -> tick of its latest audit request
        self.ticks = 0
        self._evidence_version = 0  # counts the audits that changed some request's shares
        self._signature: tuple[bytes, bytes, int, float] | None = None  # what the cached allocation is for
        self._cached: tuple[np.ndarray, np.ndarray] | None = None  # its keep mask and release mask
        self.fraction = (
            settings.budget_fraction
        )  # the budget as a fraction of all blocks; moves if a target is set
        self._last_applied: tuple[int, int] | None = None  # the last (generation, time) the engine applied
        self._steps_ms: list[float] = []  # recent time between engine steps, since the fraction last changed
        self.reloads = reloads
        # (start, end) of the steps not yet compared with the reloads: a step is compared once the reload
        # source has read past its end
        self._unclassified: deque[tuple[int, int]] = deque()
        self.steps_classified = 0  # steps compared with the reloads
        self.steps_with_reload = 0  # of them, the steps during which a reload was in flight
        self._classified_ns = 0  # the time of the steps compared, added up
        self._reload_ns = 0  # the time during them that a reload was in flight, added up
        self._commit_ns: dict[int, int] = {}  # generation -> when it was committed, until it is applied
        self._seen_applied = 0
        self._fraction_sum = 0.0  # the budget fraction at each tick, added up: its mean is reported
        self._audit_asked_ns: dict[int, int] = {}  # request key -> when its latest audit was requested
        self.audits_discarded = 0  # audits whose statistic a later audit step overwrote before it was read
        # request key -> when the controller first saw it without evidence, or last asked for its audit
        self._unmeasured: dict[int, int] = {}
        self.reaction_ms: deque[float] = deque(maxlen=100_000)  # commit to application by the engine
        self.audit_ms: deque[float] = deque(maxlen=100_000)  # request for an audit to its evidence

    # An engine step applies the newest generation, so the time between two applications is a step time.
    # A gap longer than this many targets is an idle engine, not a step.
    IDLE_GAP_TARGETS = 10
    STEPS_PER_ADJUSTMENT = 8  # steps measured, with the current fraction, before the fraction moves again
    SLACK = 0.9  # raise the fraction only when the step time is below this share of the target
    RAISE = 1.08  # factor by which the fraction rises when there is slack
    # Steps that may wait for the reload source to read past their end. More means the copy collector has
    # stopped reporting, and the reload-aware target cannot tell which steps carried a reload.
    MAX_UNCLASSIFIED = 64
    # A request above the length cap that has had no evidence for this long is asked for an audit: its first
    # audit can be lost when a later audit step overwrites the statistic before the controller reads it.
    UNMEASURED_RETRY_NS = 1_000_000_000

    def _note_applied(self, status: dict[str, int]) -> None:
        """When the engine has applied a new generation, record how long it took from the commit."""
        generation = status["applied_generation"]
        if generation == self._seen_applied:
            return
        self._seen_applied = generation
        committed = self._commit_ns.pop(generation, None)
        if committed is not None:
            self.reaction_ms.append((status["applied_ns"] - committed) / 1e6)
        for older in [g for g in self._commit_ns if g < generation]:
            del self._commit_ns[older]  # a generation that was skipped is never applied

    def summary(self) -> dict[str, float]:
        """Percentiles, in milliseconds, of the time from a commit to the engine applying it, and of an audit
        from the request for it to the evidence reaching the controller."""
        out: dict[str, float] = {
            "ticks": float(self.ticks),
            "audits_discarded": float(self.audits_discarded),
            "fraction_final": self.fraction,
            "fraction_mean": self._fraction_sum / max(self.ticks, 1),
        }
        if self.reloads is not None:
            out["steps_classified"] = float(self.steps_classified)
            out["steps_with_reload"] = float(self.steps_with_reload)
            out["reload_share"] = self._reload_ns / max(self._classified_ns, 1)
        for name, samples in (("reaction_ms", self.reaction_ms), ("audit_ms", self.audit_ms)):
            if samples:
                values = np.asarray(samples)
                out[f"{name}_p50"] = float(np.percentile(values, 50))
                out[f"{name}_p99"] = float(np.percentile(values, 99))
                out[f"{name}_count"] = float(len(values))
        return out

    def _observe_step_time(self, status: dict[str, int]) -> None:
        """Record the time since the engine last applied a generation, and move the fraction toward the target."""
        target = self.settings.target_step_ms
        if target is None:
            return
        applied = (status["applied_generation"], status["applied_ns"])
        if self._last_applied is not None and applied[0] != self._last_applied[0]:
            gap_ms = (applied[1] - self._last_applied[1]) / 1e6
            if 0.0 < gap_ms <= self.IDLE_GAP_TARGETS * target:
                if self.reloads is None:
                    self._steps_ms.append(gap_ms)
                else:
                    self._unclassified.append((self._last_applied[1], applied[1]))
        if self._last_applied is None or applied[0] != self._last_applied[0]:
            self._last_applied = applied
        if self.reloads is not None:
            self._classify_steps(self.reloads)
        if len(self._steps_ms) < self.STEPS_PER_ADJUSTMENT:
            return
        median = float(np.median(self._steps_ms))
        if median > target:
            # step time is about F + A k, so a smaller fraction lowers it; the square root damps the move
            self.fraction = max(self.settings.min_budget_fraction, self.fraction * (target / median) ** 0.5)
        elif median < self.SLACK * target:
            self.fraction = min(1.0, self.fraction * self.RAISE)
        self._steps_ms.clear()  # measure the next steps with the new fraction
        self._unclassified.clear()

    def _classify_steps(self, reloads: ReloadSource) -> None:
        """Compare each step whose end the reload source has read past with the reloads, and keep as a sample
        of the step time only a step during which no reload was in flight."""
        reloads.poll()
        while self._unclassified and self._unclassified[0][1] <= reloads.watermark_ns:
            start, end = self._unclassified.popleft()
            overlap = reloads.in_flight_ns(start, end)
            self.steps_classified += 1
            self._classified_ns += end - start
            if overlap:
                self.steps_with_reload += 1
                self._reload_ns += overlap
            else:
                self._steps_ms.append((end - start) / 1e6)
        if len(self._unclassified) > self.MAX_UNCLASSIFIED:
            raise RuntimeError(
                f"{len(self._unclassified)} engine steps wait for the copy collector, whose newest record is "
                f"from {(self._unclassified[-1][1] - reloads.watermark_ns) / 1e6:.0f} ms before the last step "
                "ended: the reload-aware target cannot tell which steps carried a reload"
            )
        # a reload that completed before the oldest step not yet compared began, or before the step that runs
        # now began, can overlap no step still to compare
        assert self._last_applied is not None  # set before the first step is recorded
        reloads.forget_before(self._unclassified[0][0] if self._unclassified else self._last_applied[1])

    def _absorb_audit(self, keys: np.ndarray, lengths: np.ndarray) -> None:
        """Fold the latest completed audit into the per-request shares.

        The statistic is stored per scheduler row. The length of each audited request is its
        current length in blocks, looked up by request key, because the batch may have been
        reordered since the audit. A request that has finished has no blocks and no evidence. A request at or
        below the length cap reads in full whatever it measures, so its row is not read either: with many short
        requests in the batch, reading only the long rows keeps the cost of an audit small.
        """
        done, started, audit_keys = self.writer.box.audit_record()
        if done == self._audits_seen:
            return
        self._audits_seen = done  # an audit is read once or not at all
        if self.source is None or started != done:
            # nothing to read, or a later audit step is already writing the statistic of every row
            self.audits_discarded += self.source is not None
            self.writer.ack_audit(done)
            return
        audited = np.flatnonzero(audit_keys)
        live = int(audited[-1]) + 1 if audited.size else 0
        current = {int(k): int(n) for k, n in zip(keys, lengths) if int(k)}
        row_blocks = np.array([current.get(int(audit_keys[row]), 0) for row in range(live)], dtype=np.int64)
        row_blocks[row_blocks <= self.settings.cap_above_blocks] = 0
        fresh = self.source.read(live, row_blocks) if live else np.empty((0, 0))
        overwritten = self.writer.box.audit_record()[1] != done  # an audit step started during the copy
        self.writer.ack_audit(done)  # the serving process may start the next audit step
        if overwritten:
            self.audits_discarded += 1
            return
        decay = self.settings.share_decay
        for row in range(live):
            key = int(audit_keys[row])
            if key == 0 or np.isnan(fresh[row]).all():
                continue
            new = np.nan_to_num(fresh[row], nan=0.0)
            old = self.shares.get(key)
            if old is not None:
                grown = np.zeros(max(len(old), len(new)))
                grown[: len(old)] = old
                shorter = np.zeros_like(grown)
                shorter[: len(new)] = new
                new = decay * grown + (1.0 - decay) * shorter
            self.shares[key] = new
            self.measured[key] = int(row_blocks[row])
            self._unmeasured.pop(key, None)
            self._fresh.add(key)
            self._evidence_version += 1
            asked = self._audit_asked_ns.pop(key, None)
            if (
                asked is not None
            ):  # an audit that the controller requested, not the first one of a new request
                self.audit_ms.append((time.monotonic_ns() - asked) / 1e6)

    def _forget_departed(self, keys: np.ndarray) -> None:
        """Drop what the controller knows about requests that are no longer in the batch.

        A finished request's evidence is no longer needed. A request that left the batch and came back,
        for example after a preemption, starts again: it has new blocks and nothing released. This runs
        on every tick, so the request is forgotten before it can come back."""
        present = {int(k) for k in keys if int(k)}
        for key in set(self.shares) - present:
            del self.shares[key], self.measured[key]
        for table in (self.streak, self.release, self._unmeasured):
            for key in set(table) - present:
                del table[key]

    def tick(self) -> bool:
        """Commit one generation. Returns whether the mailbox accepted it.

        The allocation depends on which requests are present, how long they are, and what the audits
        measured. When none of these changed since the last tick, the last allocation still holds and the
        tick only renews the lease and the audit requests. That keeps the cost of a tick, which runs every
        few milliseconds, proportional to what changed.
        """
        box = self.writer.box
        _context, keys, lengths = box.request_context()
        self._absorb_audit(keys, lengths)
        self._forget_departed(keys)
        status = box.status()
        self._note_applied(status)
        self._observe_step_time(status)
        signature = (keys.tobytes(), lengths.tobytes(), self._evidence_version, round(self.fraction, 4))
        if self._cached is None or signature != self._signature:
            self._cached = self._allocate(keys, lengths)
            self._signature = signature
        keep, release = self._cached
        self.writer.stage_rows(keep, self._audit_requests(keys, lengths), keys, release)
        self.writer.stage_lease(time.monotonic_ns() + int(self.settings.lease_seconds * 1e9))
        self.ticks += 1
        self._fraction_sum += self.fraction
        accepted = bool(self.writer.commit())
        if accepted:
            self._commit_ns[self.writer.last_generation] = time.monotonic_ns()
        return accepted

    def _allocate(self, keys: np.ndarray, lengths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the keep mask and the release mask of every row for the requests that are present."""
        box = self.writer.box
        keep = np.ones((box.max_rows, box.row_width), dtype=np.uint8)
        measured = [
            row
            for row in range(box.max_rows)
            if lengths[row] > self.settings.cap_above_blocks
            and int(keys[row])
            and (self.fixed_scores or int(keys[row]) in self.shares)
        ]
        requests: list[Request] = []
        resident: list[np.ndarray] = []  # the blocks of each request that still exist
        for row in measured:
            key = int(keys[row])
            n = min(int(lengths[row]), box.row_width)
            if self.fixed_scores is not None:
                scores = np.asarray(self.fixed_scores(n), dtype=np.float64)
            else:
                scores = np.zeros(n)
                have = self.shares[key][:n]
                scores[: len(have)] = have
            pinned = np.zeros(n, dtype=bool)
            pinned[0] = True
            pinned[max(0, n - 1 - self.settings.recent_blocks) :] = True
            if self.fixed_scores is None:
                pinned[self.measured[key] :] = True  # added since the audit: no evidence, so read
            exists = np.ones(n, dtype=bool)
            if key in self.release:
                exists = ~self.release[key][:n]  # a freed block is gone, so it takes no part in the budget
            resident.append(exists)
            requests.append(
                Request(scores=scores[exists], weight=float(self.weight(key)), pinned=pinned[exists])
            )
        self.allocation = []
        release = np.zeros_like(keep)
        if requests:
            total = sum(len(e) for e in resident)  # the budget is a fraction of every block, freed or not
            floor = sum(int(r.pinned_mask().sum()) for r in requests)
            budget = max(floor, int(np.ceil(self.fraction * total)))
            for row, exists, kept in zip(
                measured, resident, allocate(requests, budget, self.settings.objective), strict=True
            ):
                key, n = int(keys[row]), len(exists)
                full = np.zeros(n, dtype=bool)
                full[exists] = kept
                keep[row, :n] = full
                self.allocation.append((key, n, int(np.count_nonzero(full))))
                if self.settings.release_after and key in self._fresh:
                    self._count_exclusions(key, full)
                if key in self.release:
                    release[row, : len(self.release[key])] = self.release[key]
        self._fresh.clear()
        return keep, release

    def _count_exclusions(self, key: int, kept: np.ndarray) -> None:
        """Update, after an audit of request ``key``, how many audits in a row left each block out.

        A block that reaches ``release_after`` is asked to be freed, and stays asked. A block that was
        kept in any audit starts again from zero, so a request whose needs change does not lose blocks."""
        n = len(kept)
        streak = self.streak.setdefault(key, np.zeros(self.writer.box.row_width, dtype=np.int64))
        streak[:n] = np.where(kept, 0, streak[:n] + 1)
        marked = self.release.setdefault(key, np.zeros(self.writer.box.row_width, dtype=bool))
        marked[:n] |= streak[:n] >= self.settings.release_after

    def _audit_requests(self, keys: np.ndarray, lengths: np.ndarray) -> np.ndarray:
        """Ask for a new audit of each live request that has grown by ``refresh_blocks`` unmeasured blocks.

        The serving process audits a new request on its own, so a request with no evidence yet is asked for an
        audit here only when it has had none for ``UNMEASURED_RETRY_NS``: its first audit was lost. A request at
        or below the length cap reads in full, so it is never asked."""
        audit = np.zeros(self.writer.box.max_rows, dtype=np.uint8)
        if self.source is None:
            return audit
        now = time.monotonic_ns()
        for row in range(self.writer.box.max_rows):
            key = int(keys[row])
            if not key or int(lengths[row]) <= self.settings.cap_above_blocks:
                continue
            if key not in self.shares:
                if now - self._unmeasured.setdefault(key, now) >= self.UNMEASURED_RETRY_NS:
                    audit[row] = 1
                    self._unmeasured[key] = now
                continue
            if self.settings.refresh_blocks == 0:
                continue
            held = (
                key in self._asked_tick
                and self.ticks - self._asked_tick[key] < self.settings.audit_hold_ticks
            )
            unmeasured = int(lengths[row]) - self.measured[key]
            if unmeasured >= self.settings.refresh_blocks and not held:
                audit[row] = 1
                self._asked_tick[key] = self.ticks
                self._audit_asked_ns[key] = now
        return audit
