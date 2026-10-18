"""The records eKV speaks in — the typed output of each observation source, the raw
eBPF copy-record sum type, and the fused ``KVEdge`` they collapse into.

    RequestKVUse   ← the request↔block bridge: request R touched block(s) at time t
    BlockingEvent  ← scheduler ground truth: R was parked on an async KV load
    Copy           ← eBPF lifecycle tracing (eBPF): a physical transfer + its measured cost,
                     by address (correlated from the Enqueue|Bind|Completion stream)
    KVEdge         ← fuser output: a σ edge {block, hash, tiers, bytes, cost, request}

Every record is **frozen**: an observation is an immutable fact. Correlation and
fusion produce new records rather than mutating old ones — derived quantities
(``Copy.cost``) are computed, never stored, so they can never disagree with the
fields they come from.
"""
from __future__ import annotations

from dataclasses import dataclass

from .types import (Address, BlockId, CompletionVia, ContentHash,
                    Direction, EventHandle, KVRole, Nanoseconds, Op, Pid,
                    RequestId, SizeBytes, StreamHandle, Tier)


# ── the request↔block bridge (the relation no public surface provides) ──────
@dataclass(frozen=True, slots=True)
class RequestKVUse:
    """Request R touched these KV block(s) in `role` at `at`. The request-to-block relation for
    SOUND per-request attribution: content_hash is many-to-many over requests
    (shared system prompts / repeated prefixes), so the request↔block relation
    can't be inferred from hashes — it's read in-process where vLLM's KVCacheManager
    assigns blocks. `at` is CLOCK_MONOTONIC (== the eBPF collector's clock)."""
    request_id: RequestId
    role: KVRole
    block_ids: tuple[int, ...]
    at: Nanoseconds
    content_hashes: tuple[ContentHash, ...] = ()
    step: int | None = None


# ── the request-blocking relation (scheduler ground truth for KV-load latency) ──
@dataclass(frozen=True, slots=True)
class BlockingEvent:
    """Request R was scheduler-blocked on an async KV load for [entered, exited):
    vLLM parks a request needing offloaded/remote KV in WAITING_FOR_REMOTE_KVS and
    does NOT schedule it until the load (an H2D the eBPF tracer captures) finishes.

    ``duration`` is per-request, ground-truth latency: the wall-clock the request's own
    progress was stalled because its KV was not resident. It is WELL-DEFINED under
    continuous-batching overlap precisely because it is the request's own
    un-schedulable interval — not a wall-clock-overlap estimate that has to decide
    'whose compute hid this'. ``n_steps`` is how many schedule() steps it spanned;
    ``blocks``/``n_tokens`` are what was being loaded (joins to RequestKVUse)."""
    request_id: RequestId
    entered_at: Nanoseconds
    exited_at: Nanoseconds
    n_steps: int = 0
    n_tokens: int = 0
    blocks: tuple[int, ...] = ()

    @property
    def duration(self) -> Nanoseconds:
        return Nanoseconds(max(0, self.exited_at - self.entered_at))


# ── eBPF lifecycle tracing: the raw eBPF copy-record stream is a SUM TYPE ─────────────
@dataclass(frozen=True, slots=True)
class Enqueue:
    """A copy was issued (cudaMemcpyAsync / a cuMemcpyBatchAsync element)."""
    addr: Address                # device side: src for D2H, dst for H2D, src for BATCH
    nbytes: SizeBytes
    direction: Direction
    at: Nanoseconds
    stream: StreamHandle
    pid: Pid | None = None
    other_addr: Address | None = None   # the non-device side (BATCH: the counterpart)


@dataclass(frozen=True, slots=True)
class Bind:
    """cudaEventRecord(event, stream): the stream's pending copies now belong to
    ``event`` — so an event completion can be paired to exactly those copies, even
    across concurrent offload streams."""
    event: EventHandle
    stream: StreamHandle
    at: Nanoseconds


@dataclass(frozen=True, slots=True)
class Completion:
    """A sync/query returned: ``count`` copies on ``handle`` (a stream, or a bound
    event) finished at ``at``, observed ``via`` this mechanism."""
    handle: int                  # a StreamHandle or EventHandle (the reader keys on it)
    via: CompletionVia
    count: int
    at: Nanoseconds


@dataclass(frozen=True, slots=True)
class Gap:
    """The collector restarted (its supervisor emits this into the stream): capture was
    interrupted, so an unknown number of records in this window were lost. Not an error to
    correlate — a COMPLETENESS marker, so attribution over a window that contains one is
    flagged incomplete rather than silently trusted as whole."""
    at: Nanoseconds
    reason: str = ""


CopyRecord = Enqueue | Bind | Completion | Gap   # the eBPF copy stream, as a tagged union


# ── eBPF lifecycle tracing: a correlated copy (enqueue + optional completion) ─────────
@dataclass(frozen=True, slots=True)
class Copy:
    """An enqueued transfer with its completion once correlated. ``finished_at is
    None`` ⇔ still pending — a pending copy simply has no cost yet, so the illegal
    "completed but no time" / "pending but has cost" states cannot be represented."""
    addr: Address
    nbytes: SizeBytes
    direction: Direction
    enqueued_at: Nanoseconds
    finished_at: Nanoseconds | None = None
    via: CompletionVia | None = None
    pid: Pid | None = None
    other_addr: Address | None = None

    @property
    def cost(self) -> Nanoseconds | None:
        """Completion latency — the σ-edge weight — or None while pending."""
        if self.finished_at is None:
            return None
        return Nanoseconds(self.finished_at - self.enqueued_at)

    @property
    def is_complete(self) -> bool:
        return self.finished_at is not None


# ── fuser output ────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class KVEdge:
    """A fused causal edge in σ: a KV block moved/served, with its measured cost —
    the single currency everything downstream (flame, DSL) consumes. ``content_hash``
    is the cross-rank aggregation key (addresses differ per TP rank)."""
    op: Op
    block_id: BlockId | None
    content_hash: ContentHash | None
    prefix_hash: ContentHash | None
    src_tier: Tier | None
    dst_tier: Tier | None
    nbytes: SizeBytes | None
    cost: Nanoseconds | None         # the measured edge weight ĉ
    step: int | None = None
    request: RequestId | None = None
    note: str = ""                   # provenance: which faces contributed
