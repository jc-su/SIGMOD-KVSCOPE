"""A per-request view of the block table for decode attention.

A paged attention kernel reads a request's KV cache through its row of the block table and its
sequence length. A shorter row with a shorter length makes the same kernel read fewer blocks, so
a view of the table that keeps only some blocks is a read budget the kernel enforces without any
change to it. The view is exact for full causal attention: keys carry their positional encoding
in the cache, and a decode query attends every key it is given.

``DecodeView`` holds the policy the controller committed and gives each batch its view.
``install_decode_view`` points the attention metadata of a batch at view buffers that are
refilled before every step. Captured CUDA graphs read those buffers, so the view needs no change
to the graph. Only the rows that decode in this step are trimmed. A row that prefills in the same
step, an expired policy, and an audit step give the identity view, a copy of the engine's table.
A batch with a shared-prefix cascade reads the engine's own table.
"""

import os
import time
from collections.abc import Callable
from typing import Protocol

import numpy as np
import torch

from runtime.audit import AuditPlane


class StatisticSwitch(Protocol):
    """What the view needs to turn the attention-share statistic on for a step and off after it."""

    def set(self, on: bool) -> None: ...

    def select(self, audited: np.ndarray, token_start: np.ndarray) -> None: ...


def compact_block_table(
    table: torch.Tensor, seq_lens: torch.Tensor, keep: torch.Tensor, block_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the view of each request row that keeps the blocks marked in ``keep``.

    ``table`` is the int32 block table ``[rows, width]``, ``seq_lens`` the int32 token count of
    each row, and ``keep`` a bool ``[rows, width]`` that marks the blocks to read. The first block
    and the last block of every request are always kept: the first is the minimum the request
    needs, and the last holds its newest token. Kept blocks stay in their original order. The view
    table holds the kept blocks first; entries after them are not read, because the view length
    ends at the last kept block. The view length is the kept blocks times ``block_size``, less
    the unfilled part of the last block. A row with no tokens gets length 0.
    """
    blocks = (seq_lens + block_size - 1) // block_size
    pos = torch.arange(table.shape[1], device=table.device)[None, :]
    keep = (keep | (pos == 0) | (pos == (blocks - 1)[:, None])) & (pos < blocks[:, None])
    order = torch.argsort((~keep).to(torch.int8), dim=1, stable=True)
    kept = keep.sum(dim=1).to(seq_lens.dtype)
    last = seq_lens - (blocks - 1) * block_size
    view_lens = torch.where(blocks > 0, (kept - 1) * block_size + last, torch.zeros_like(kept))
    return table.gather(1, order), view_lens


class DecodeView:
    """The committed read policy of one serving process, and the view it gives each decode batch.

    The controller's generations reach this object through ``ScheduledPublisher``, which calls
    the ``write_*`` methods below with a generation already remapped to the current scheduler
    rows. ``deadline_ns`` is a ``time.monotonic_ns`` instant, the same clock in every process on
    the host. A policy whose deadline has passed, or that was never committed, gives the identity
    view, so a stopped or killed controller returns every request to full reads without any
    cleanup. A row the controller asks to audit gets the identity view for one decode step, so that
    the instrumented kernel sees every block of that request and can measure its attention share.
    The request is consumed by that step.
    """

    def __init__(
        self,
        rows: int,
        width: int,
        block_size: int,
        audit: StatisticSwitch | None = None,
        prefill_audit: bool = False,
    ) -> None:
        if min(rows, width, block_size) <= 0:
            raise ValueError("view dimensions must be positive")
        if prefill_audit and audit is None:
            raise ValueError("an audit in the last prefill step needs an audit plane")
        self.rows, self.width, self.block_size = rows, width, block_size
        self.audit = audit
        # Serve a new request's audit in the step that computes the last tokens of its prompt. That step reads
        # every block of the request anyway, so the audit costs no read of its own, and the evidence arrives one
        # step earlier. It needs a recording kernel in which only the program with a request's newest query tokens
        # records (the pass's ``attn_share_newest_rows``).
        self.prefill_audit = prefill_audit
        self.final_prefill = np.zeros(
            rows, dtype=bool
        )  # rows that compute the last tokens of their prompt next
        # called before a step audits a prefilling row; it fails if the recording kernel cannot (the plane's check)
        self.on_prefill_audit: Callable[[], None] | None = None
        self.keep = np.ones((rows, width), dtype=bool)
        # Blocks the engine has taken back. A released block is never read, under any policy, in an audit,
        # or after the lease has expired, because the engine may have given its memory to another request.
        self.released = np.zeros((rows, width), dtype=bool)
        self.released_version = 0
        self.row_audited = np.zeros(rows, dtype=bool)  # the row's request has had a completed audit
        self.audited_blocks = np.zeros(rows, dtype=np.int64)  # its length in blocks at that audit
        self._pending_blocks = np.zeros(rows, dtype=np.int64)
        self.audit_rows = np.zeros(rows, dtype=bool)  # rows whose next decode step is an audit
        self.last_audit_rows = np.zeros(rows, dtype=bool)  # rows read in full at the last completed audit
        self.deadline_ns = 0
        self.version = 0
        self.audit_step = False
        self.audits_completed = 0
        self.audits_started = 0
        # called when an audit step starts, before the device runs it (the publisher publishes the start)
        self.on_audit_start: Callable[[], None] | None = None
        # asked before an audit step starts; False keeps the audit requests for a later step (the publisher's gate)
        self.audit_gate: Callable[[], bool] | None = None
        # called first in every build: the publisher applies the newest generation there. The engine can wait
        # for the GPU for seconds between updating its batch and building the view (a long prefill chunk in
        # flight), longer than a lease, so a generation applied only at the batch update can expire in between.
        self.before_build: Callable[[], None] | None = None
        self._audit_pending = False
        self._pending_rows = np.zeros(rows, dtype=bool)
        # read, full, audited, and untrimmed unaudited blocks of every decode row, over all steps
        self._counts: torch.Tensor | None = None
        self._device: torch.device | None = None
        self._device_keep: torch.Tensor | None = None
        self._device_version = -1
        self._device_released: torch.Tensor | None = None
        self._released_device_version = -1
        self._table: torch.Tensor | None = None
        self._lens: torch.Tensor | None = None

    # -- the controller-facing writes, called by ScheduledPublisher ------------------------------

    def write_row_controls(self, mask: np.ndarray, log_weight: np.ndarray | None = None) -> int:
        """Commit one generation: ``mask[row, block]`` nonzero means read that block."""
        if log_weight is not None:
            raise ValueError("a block-table view enforces hard keep decisions only")
        mask = np.asarray(mask)
        if mask.shape != self.keep.shape:
            raise ValueError(f"mask shape {mask.shape} != view shape {self.keep.shape}")
        self.keep = mask != 0
        self.version += 1
        return 1

    def write_lease_deadline(self, deadline_ns: int) -> int:
        self.deadline_ns = int(deadline_ns)
        return 1

    def write_audit_rows(self, requested: np.ndarray) -> int:
        """Ask for an audit of the rows marked in ``requested``; each is consumed by one decode step."""
        requested = np.asarray(requested, dtype=bool)
        if requested.shape != self.audit_rows.shape:
            raise ValueError(f"audit shape {requested.shape} != {self.audit_rows.shape}")
        if requested.any() and self.audit is None:
            raise ValueError(
                "an audit was requested but the view has no audit plane to switch the statistic on"
            )
        self.audit_rows |= requested
        return 1

    def write_final_prefill(self, rows: np.ndarray) -> None:
        """Mark the rows that compute the last tokens of their prompt in the next step."""
        rows = np.asarray(rows, dtype=bool)
        if rows.shape[0] > self.rows:
            raise ValueError(f"{rows.shape[0]} rows exceed the view's {self.rows}")
        self.final_prefill[:] = False
        self.final_prefill[: rows.shape[0]] = rows

    def write_release(self, mask: np.ndarray) -> list[tuple[int, int]]:
        """Mark blocks as released and return the ``(row, block)`` pairs that are new.

        A block can be released only after its request has had a completed audit, and only if the
        audit covered it: a mark from a controller that has not seen this request yet, or that saw an
        earlier request in the same row, is refused. The first block, and the last block at the time of
        that audit, are never released, because the view always reads them.
        """
        mask = np.asarray(mask) != 0
        if mask.shape != self.released.shape:
            raise ValueError(f"release shape {mask.shape} != view shape {self.released.shape}")
        eligible = np.zeros_like(mask)
        for row in np.flatnonzero(self.row_audited):
            eligible[row, : max(int(self.audited_blocks[row]) - 1, 0)] = True
        eligible[:, 0] = False
        new = mask & eligible & ~self.released
        rows, blocks = np.nonzero(new)
        if rows.size:
            self.released |= new
            self.released_version += 1
        return list(zip(rows.tolist(), blocks.tolist(), strict=True))

    def reassign_rows(self, moved: dict[int, int], new: list[int], empty: list[int]) -> None:
        """The scheduler changed which request each row holds.

        ``moved`` maps a row to the row its request held in the previous batch, and the request
        keeps its keep decisions and its pending audit. A row in ``new`` holds a request that was
        not in the previous batch: it starts with full reads and, with an audit plane, is audited
        at its first decode step. The serving process asks for that audit itself, so a request gets
        its evidence without waiting for a controller round trip. A row in ``empty`` holds no request.
        """
        keep, audit = self.keep.copy(), self.audit_rows.copy()
        released, audited, blocks = self.released.copy(), self.row_audited.copy(), self.audited_blocks.copy()
        for row, old in moved.items():
            keep[row], audit[row] = self.keep[old], self.audit_rows[old]
            released[row], audited[row], blocks[row] = (
                self.released[old],
                self.row_audited[old],
                self.audited_blocks[old],
            )
        for row in new:
            keep[row], audit[row] = True, self.audit is not None
            released[row], audited[row], blocks[row] = False, False, 0
        for row in empty:
            keep[row], audit[row] = True, False
            released[row], audited[row], blocks[row] = False, False, 0
        self.keep, self.audit_rows = keep, audit
        if (released != self.released).any():
            self.released_version += 1
        self.released, self.row_audited, self.audited_blocks = released, audited, blocks
        self.version += 1

    # -- the engine-facing view -----------------------------------------------------------------

    def active(self) -> bool:
        return self.deadline_ns > time.monotonic_ns()

    def build(
        self, table: torch.Tensor, seq_lens: torch.Tensor, query_lens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the view buffers (table, lengths) for a batch of ``len(query_lens)`` rows.

        ``query_lens`` is the CPU tensor of tokens each row computes in this step. A row that
        computes one token is decoding and gets its view. A row that computes more is prefilling,
        and it reads its whole table, so a step that mixes prefill and decode still trims its
        decode rows. An audit request waits for a step in which its row decodes, or, with
        ``prefill_audit``, computes the last tokens of its prompt.
        """
        if self.before_build is not None:
            self.before_build()
        # One reading of the lease, before any device work, decides the step and labels its reads. A copy to
        # the device below waits for the GPU's earlier work, which can outlast a lease.
        live = self.active()
        self.finish_audit()
        self._device = table.device
        num_reqs = len(query_lens)
        width = table.shape[1]
        if width != self.width or num_reqs > self.rows:
            raise ValueError(
                f"batch ({num_reqs} rows, {width} blocks) exceeds view ({self.rows}, {self.width})"
            )
        if self._table is None:
            self._table = torch.zeros(self.rows, width, dtype=torch.int32, device=table.device)
            self._lens = torch.zeros(self.rows, dtype=torch.int32, device=table.device)
        assert self._lens is not None
        decode = query_lens.numpy() == 1
        all_decode = bool(decode.all())
        servable = decode | (self.final_prefill[:num_reqs] & self.prefill_audit)
        self.final_prefill[:] = False  # the mark holds for this step only
        audited = np.zeros(self.rows, dtype=bool)
        audited[:num_reqs] = self.audit_rows[:num_reqs] & servable
        if audited.any() and not self.may_audit():
            audited[:] = False  # the requests wait for a step after the controller has read the last audit
        if (audited[:num_reqs] & ~decode).any() and self.on_prefill_audit is not None:
            self.on_prefill_audit()
        self.audit_rows[audited] = False  # an audit request is consumed by the step that serves it
        self.audit_step = bool(audited.any())
        self._audit_pending = self.audit_step
        self._pending_rows = audited
        if self.audit_step:
            self.start_audit()
        if self.audit_step:  # the length of each audited request, for the release rule
            size = self.block_size
            self._pending_blocks[:num_reqs] = ((seq_lens[:num_reqs] + size - 1) // size).cpu().numpy()
        keep = self._keep_on(table.device)[:num_reqs]
        if not live:
            keep = torch.ones_like(keep)
        elif self.audit_step:
            keep = keep | torch.from_numpy(audited[:num_reqs]).to(table.device)[:, None]
        decoding = None if all_decode else torch.from_numpy(decode).to(table.device)
        if decoding is not None:
            keep = keep | ~decoding[:, None]
        if self.released.any():  # last, so nothing above can bring a released block back
            keep = keep & ~self._released_on(table.device)[:num_reqs]
        if self.audit is not None:
            starts = np.concatenate(
                ([0], np.cumsum(query_lens.numpy())[:-1])
            )  # token index of each row's query
            self.audit.select(audited[:num_reqs], starts)
            self.audit.set(self.audit_step)
        view_table, view_lens = compact_block_table(
            table[:num_reqs], seq_lens[:num_reqs], keep, self.block_size
        )
        self._table[:num_reqs] = view_table
        self._lens[:num_reqs] = view_lens
        audited_rows = torch.from_numpy(audited[:num_reqs]).to(table.device)
        self._count(view_lens, seq_lens[:num_reqs], decoding, audited_rows, expired=not live)
        return self._table[:num_reqs], self._lens[:num_reqs]

    def _count(
        self,
        read_lens: torch.Tensor,
        full_lens: torch.Tensor,
        decoding: torch.Tensor | None,
        audited: torch.Tensor | None = None,
        expired: bool = False,
    ) -> None:
        """Add the blocks the kernel is given, and the blocks a full read would use, for decode rows.

        Three parts of the blocks read are counted apart: those of audited rows, those read while
        no policy is live (never committed, or its lease expired), and those of rows that read
        their whole table in a live policy without being audited (no evidence for the request yet).
        """
        if self._counts is None:
            self._counts = torch.zeros(5, dtype=torch.int64, device=read_lens.device)
        size = self.block_size
        read = (read_lens + size - 1) // size
        full = (full_lens + size - 1) // size
        whole = read_lens == full_lens
        if decoding is not None:
            read, full = read * decoding, full * decoding
        in_audit = torch.zeros_like(whole) if audited is None else audited
        live_whole = whole & ~in_audit & (not expired)
        self._counts += torch.stack(
            [
                read.sum(),
                full.sum(),
                (read * in_audit).sum(),
                (read * live_whole).sum(),
                read.sum() if expired else torch.zeros((), dtype=torch.int64, device=read.device),
            ]
        )

    @property
    def buffers(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The view's block table and lengths. They are the buffers the engine's kernels read, and CUDA graphs capture."""
        if self._table is None or self._lens is None:
            raise RuntimeError("the view has built no batch yet")
        return self._table, self._lens

    def reads(self) -> tuple[int, int]:
        """Return the blocks read and the blocks full reads would have read, over every decode step."""
        if self._counts is None:
            return 0, 0
        read, full = (int(x) for x in self._counts.tolist()[:2])
        return read, full

    def read_breakdown(self) -> dict[str, int]:
        """Return the blocks read, split into audit steps, rows without evidence, and expired policy."""
        names = ("read", "full", "audit", "untrimmed", "expired")
        if self._counts is None:
            return dict.fromkeys(names, 0)
        return dict(zip(names, (int(x) for x in self._counts.tolist()), strict=True))

    def skip(self, seq_lens: torch.Tensor, query_lens: torch.Tensor) -> None:
        """A batch with a shared-prefix cascade reads the engine's own table and records no statistic.

        Its decode rows count as full reads, so the read fraction covers every decode row."""
        self.finish_audit()
        self.audit_step = False
        if self.audit is not None:
            self.audit.set(False)
        decoding = torch.from_numpy(query_lens.numpy() == 1).to(seq_lens.device)
        self._count(seq_lens[: len(query_lens)], seq_lens[: len(query_lens)], decoding)

    def may_audit(self) -> bool:
        return self.audit_gate is None or self.audit_gate()

    def start_audit(self) -> None:
        """Count an audit step that starts now. Its statistic overwrites the previous audit's, so a reader of
        that one must know: the step is published before the device runs it."""
        self.audits_started += 1
        if self.on_audit_start is not None:
            self.on_audit_start()

    def finish_audit(self) -> None:
        """Count the previous step as a completed audit once the device has finished writing it.

        The statistic is complete only when every layer of that step has run, so the controller is
        told about an audit only after a device synchronization, which audit steps alone pay for.
        The audited rows are those of the step's own batch, so this must run before ``reassign_rows``
        moves requests to other rows; the publisher calls it first.
        """
        if not self._audit_pending:
            return
        if self._device is not None and self._device.type == "cuda":
            torch.cuda.synchronize(self._device)
        self.last_audit_rows = self._pending_rows
        self.row_audited |= self._pending_rows
        self.audited_blocks[self._pending_rows] = self._pending_blocks[self._pending_rows]
        self.audits_completed += 1
        self._audit_pending = False

    def _keep_on(self, device: torch.device) -> torch.Tensor:
        if self._device_keep is None or self._device_version != self.version:
            self._device_keep = torch.from_numpy(self.keep).to(device, non_blocking=True)
            self._device_version = self.version
        return self._device_keep

    def _released_on(self, device: torch.device) -> torch.Tensor:
        if self._device_released is None or self._released_device_version != self.released_version:
            self._device_released = torch.from_numpy(self.released).to(device, non_blocking=True)
            self._released_device_version = self.released_version
        return self._device_released


_view: DecodeView | None = None
_plane: AuditPlane | None = None
_layers: tuple[str, ...] = ()  # the attention layers of the one cache group the view serves


def _enabled(name: str) -> bool:
    return os.environ.get(name, "") not in ("", "0", "false")


def _audit_plane(builder, cam) -> AuditPlane | None:  # a vLLM attention metadata builder and its batch
    """The plane that records the attention statistic: the compiler pass's, the audit kernel's, or none.

    ``EKV_ATTN_SHARE`` asks the compiler pass to record it inside the engine's Triton kernel.
    ``EKV_AUDIT_KERNEL`` records it with a kernel of its own after each attention layer, which
    works for any backend. They record the same statistic, so asking for both is an error.
    """
    if _enabled("EKV_ATTN_SHARE") and _enabled("EKV_AUDIT_KERNEL"):
        raise RuntimeError("EKV_ATTN_SHARE and EKV_AUDIT_KERNEL both record the statistic: set one")
    if _enabled("EKV_AUDIT_KERNEL"):
        from runtime.audit_kernel import RecordingPlane

        model, parallel = builder.vllm_config.model_config, builder.vllm_config.parallel_config
        return RecordingPlane(
            layers=model.get_num_layers_by_block_type(parallel, "attention"),
            rows=builder.vllm_config.scheduler_config.max_num_seqs,
            width=cam.block_table_tensor.shape[1],
            group_rows=model.get_num_attention_heads(parallel) // model.get_num_kv_heads(parallel),
            block_size=builder.kv_cache_spec.block_size,
        )
    return AuditPlane() if _enabled("EKV_ATTN_SHARE") else None


def require_supported_attention(builder) -> None:  # a vLLM attention metadata builder
    """Refuse, with the reason, a cache group whose attention the view cannot trim exactly.

    The view drops whole blocks from the table of a decode row. That is exact for full causal attention
    over one cache group: the kernel then reads the kept keys and values in their original order, with the
    positions they had. It is not exact, or not meaningful, for attention that depends on position beyond
    the order of the keys or that keeps a different set of keys per layer. So the view is installed only
    for a cache group that is plain full attention, and it refuses these cases instead of running on them:

    * a cache group of another kind (sliding window, chunked local, state-space, latent, encoder, cross),
    * a layer with a sliding window, a position bias (ALiBi), or attention that is not decoder self-attention,
    * a model that has more than one cache group (hybrid models).
    """
    from vllm.v1.attention.backend import AttentionType
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    spec = builder.kv_cache_spec
    if type(spec) is not FullAttentionSpec:
        raise RuntimeError(
            f"the view supports full causal attention only; this cache group is {type(spec).__name__}"
        )
    if spec.sliding_window is not None or spec.attention_chunk_size is not None:
        raise RuntimeError("the view does not support a sliding window or chunked attention in a cache group")
    layers = builder.vllm_config.compilation_config.static_forward_context
    for name in builder.layer_names:
        layer = layers[name]
        if layer.sliding_window is not None:
            raise RuntimeError(f"layer {name} has a sliding window: the view cannot trim its blocks exactly")
        if getattr(layer.impl, "alibi_slopes", None) is not None:
            raise RuntimeError(f"layer {name} has a position bias (ALiBi): the view cannot trim its blocks")
        if layer.attn_type != AttentionType.DECODER:
            raise RuntimeError(
                f"layer {name} is {layer.attn_type.value} attention: the view supports decoder attention"
            )


def _view_for(builder, cam) -> DecodeView:  # a vLLM attention metadata builder and its batch
    """The process's view, created by the first batch with the engine's row count, width, and block size."""
    global _view, _plane, _layers
    layers = tuple(builder.layer_names)
    if _view is None:
        require_supported_attention(builder)
        _plane = _audit_plane(builder, cam)
        prefill_audit = _enabled("EKV_PREFILL_AUDIT")
        if prefill_audit and not _enabled("EKV_ATTN_SHARE"):
            # the check that only a request's newest query tokens record exists for the pass's kernels
            raise RuntimeError("EKV_PREFILL_AUDIT needs the compiler pass's statistic (EKV_ATTN_SHARE)")
        _view = DecodeView(
            rows=builder.vllm_config.scheduler_config.max_num_seqs,
            width=cam.block_table_tensor.shape[1],
            block_size=builder.kv_cache_spec.block_size,
            audit=_plane,
            prefill_audit=prefill_audit,
        )
        if prefill_audit:
            assert _plane is not None  # EKV_ATTN_SHARE gives the pass's plane
            _view.on_prefill_audit = _plane.require_newest_rows
        _layers = layers
    if layers != _layers:
        raise RuntimeError("a block-table view supports one KV-cache group; this model has several")
    return _view


def _query_lens(cam) -> torch.Tensor:  # a vLLM CommonAttentionMetadata
    starts = cam.query_start_loc_cpu[: cam.num_reqs + 1]
    return starts[1:] - starts[:-1]


def install_decode_view() -> int:
    """Give every attention batch of the Triton and FlashInfer backends its view. Returns the builders patched.

    Install this before the engine is created. vLLM captures its CUDA graphs while it builds the
    first attention metadata, and a graph replays the buffers it captured. The first batch built
    creates the process's view, with the engine's own row count, block-table width, and block
    size, so the graphs capture the view's buffers and every later step is bound to them.

    The Triton builder takes the batch as it is and the view overwrites the table and lengths in its
    metadata. The FlashInfer builder plans its kernel launch on the host from the lengths, so it
    receives the view's table and lengths as its batch (the lengths come to the host once per step).
    """
    from vllm.v1.attention.backends.flashinfer import FlashInferMetadataBuilder
    from vllm.v1.attention.backends.triton_attn import TritonAttentionMetadataBuilder

    patched = 0
    triton_build = TritonAttentionMetadataBuilder.build
    if not getattr(triton_build, "_ekv_decode_view", False):

        def build_triton(self, common_prefix_len, common_attn_metadata, fast_build=False):
            metadata = triton_build(self, common_prefix_len, common_attn_metadata, fast_build)
            cam = common_attn_metadata
            view = _view_for(self, cam)
            if common_prefix_len > 0:
                view.skip(cam.seq_lens, _query_lens(cam))
                return metadata
            metadata.block_table, metadata.seq_lens = view.build(
                cam.block_table_tensor, cam.seq_lens, _query_lens(cam)
            )
            return metadata

        build_triton._ekv_decode_view = True  # ty: ignore[unresolved-attribute]  # marks the patched method
        TritonAttentionMetadataBuilder.build = build_triton
        patched += 1

    flashinfer_build = FlashInferMetadataBuilder.build
    if not getattr(flashinfer_build, "_ekv_decode_view", False):

        def build_flashinfer(self, common_prefix_len, common_attn_metadata, fast_build=False):
            cam = common_attn_metadata
            view = _view_for(self, cam)
            if common_prefix_len > 0:
                view.skip(cam.seq_lens, _query_lens(cam))
                return flashinfer_build(self, common_prefix_len, cam, fast_build)
            table, lens = view.build(cam.block_table_tensor, cam.seq_lens, _query_lens(cam))
            seen = cam.replace(
                block_table_tensor=table,
                seq_lens=lens,
                max_seq_len=int(lens.max()),
                _seq_lens_cpu=None,  # the host copy of the view's lengths, made on first use
            )
            return flashinfer_build(self, common_prefix_len, seen, fast_build)

        build_flashinfer._ekv_decode_view = True  # ty: ignore[unresolved-attribute]  # marks the patched method
        FlashInferMetadataBuilder.build = build_flashinfer
        patched += 1
    return patched


def install_audit_kernel() -> int:
    """After each attention layer of the Triton and FlashInfer backends, record the attention statistic.

    Install this before the engine is created, with ``EKV_AUDIT_KERNEL`` set. The launch is the
    same on every step, because the kernel reads the audit flag on the device, so a CUDA graph can
    capture it. Returns the attention implementations patched.
    """
    from vllm.v1.attention.backends.flashinfer import FlashInferImpl
    from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl

    from runtime.audit_kernel import RecordingPlane

    patched = 0
    for impl in (TritonAttentionImpl, FlashInferImpl):
        original = impl.forward
        if getattr(original, "_ekv_audit_kernel", False):
            continue

        def forward(
            self, layer, query, key, value, kv_cache, attn_metadata, *args, _original=original, **kwargs
        ):
            output = _original(self, layer, query, key, value, kv_cache, attn_metadata, *args, **kwargs)
            if attn_metadata is not None:  # a profiling run has no batch
                view = current_view()
                if not isinstance(view.audit, RecordingPlane):
                    raise RuntimeError("the audit kernel is installed but the view has no recording plane")
                table, lens = view.buffers
                view.audit.record(layer.layer_name, table, lens, query, kv_cache, self.scale)
            return output

        forward._ekv_audit_kernel = True  # ty: ignore[unresolved-attribute]  # marks the patched method
        impl.forward = forward
        patched += 1
    return patched


def current_view() -> DecodeView:
    """The process's view. It exists once the engine has built its first attention batch."""
    if _view is None:
        raise RuntimeError(
            "no attention batch has been built: install the decode view before the engine starts"
        )
    return _view


def attach_audit_plane(runner) -> DecodeView:  # a vLLM GPUModelRunner
    """Return the view, and give the audit plane the engine's KV-cache addresses.

    The plane exists when the statistic was asked for: by the compiler pass (``EKV_ATTN_SHARE``) or by
    the audit kernel (``EKV_AUDIT_KERNEL``). Its layer table is filled once the engine has allocated
    its KV caches, which is why this runs at the first scheduling step and not at install time. The
    pass finds a layer from the address of its cache, so the table holds the addresses in the order
    of ``runner.kv_caches``. The audit kernel numbers a layer by its name, so the table holds the
    address of each named layer's cache.
    """
    from runtime.audit_kernel import RecordingPlane

    view = current_view()
    if isinstance(_plane, RecordingPlane):
        layers = runner.compilation_config.static_forward_context
        _plane.attach([layers[name].kv_cache.data_ptr() for name in _plane.layer_names])
    elif _plane is not None:
        _plane.attach([cache.data_ptr() for cache in runner.kv_caches if cache is not None])
    return view
