"""The block-table view for SGLang's Triton attention backend.

SGLang's decode kernel reads a request's cache through a CSR list of token slots: ``kv_indptr`` marks where each
request's slots start in ``kv_indices``. Its page size is one token, so the engine has no block. The view defines
one, ``BLOCK_TOKENS`` consecutive tokens, and keeps or drops whole blocks of a request. That is the same read budget
as the vLLM view gives with its block table, and it is exact for the same reason: the kernel then reads the kept
keys and values in their original order, with the positions they had.

``runtime.block_view.DecodeView`` decides which blocks a batch keeps, from the policy that the controller committed
through the mailbox. This module gives it SGLang's batch. A safe point in the scheduler (``Scheduler.run_batch``)
publishes which request holds each row of the batch and applies a completed controller generation. After SGLang
builds the attention metadata of a decode batch, eagerly or for a CUDA graph replay, the view rewrites ``kv_indptr``
and ``kv_indices`` in place, in the buffers that the kernel and the captured graph read. The rewrite runs on the GPU
(``runtime.sglang_compact``) from the lengths the safe point read on the host, so the host never waits for the device
in a step without an audit.
"""

import os
from types import ModuleType
from typing import TYPE_CHECKING

import numpy as np
import torch

from observe.publication import ControlMailbox, ScheduledPublisher, mailbox_path
from runtime.audit import AuditPlane
from runtime.block_view import DecodeView
from runtime.sglang_compact import SlotCompactor
from runtime.sglang_hooks import when_imported

if TYPE_CHECKING:  # SGLang is imported by the engine; this module patches it after that import
    from sglang.srt.layers.attention.triton_backend import TritonAttnBackend
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.mem_cache.memory_pool import KVCache
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

BLOCK_TOKENS = 16


def kv_pointers(pool: "KVCache") -> list[int]:
    """The address of each layer's key cache in layer order, which is the order the pass looks a layer up in."""
    return [pool.get_key_buffer(layer).data_ptr() for layer in range(pool.layer_num)]


class SlotView(DecodeView):
    """The view of SGLang. It keeps the policy and the audits of ``DecodeView`` and compacts the key-value list of a
    decode batch on the GPU (``runtime.sglang_compact``). It reads fewer blocks and does not give their memory back to
    the engine."""

    def __init__(
        self, rows: int, width: int, block_size: int, audit: AuditPlane | None, device: torch.device
    ) -> None:
        super().__init__(rows, width, block_size, audit=audit)
        self.compactor = SlotCompactor(width, block_size, device)
        self._counts = (
            self.compactor.counts
        )  # reads() and read_breakdown() read the counters the kernel adds to

    def write_release(self, mask: np.ndarray) -> list[tuple[int, int]]:
        if np.asarray(mask).any():
            raise RuntimeError("the SGLang view does not release blocks: the controller asked for a release")
        return []

    def rewrite(
        self, indptr: torch.Tensor, slots: torch.Tensor, lengths: np.ndarray, n_rows: int, entries: int
    ) -> torch.Tensor:
        """Rewrite a decode batch's list in place to the view, and return the new length of each of its rows.

        ``lengths`` holds the tokens of each request, from the host, so an audit needs no device read. The batch has
        ``n_rows`` rows, more than the requests when a CUDA graph fills its batch, and ``entries`` list entries. The
        bookkeeping is that of ``DecodeView.build`` for a batch in which every row decodes."""
        self.finish_audit()
        requests = len(lengths)
        if requests > self.rows:
            raise ValueError(f"batch of {requests} requests exceeds the view's {self.rows} rows")
        self._device = indptr.device
        audited = np.zeros(self.rows, dtype=bool)
        audited[:requests] = self.audit_rows[:requests]
        if audited.any() and not self.may_audit():
            audited[:] = False  # the requests wait for a step after the controller has read the last audit
        self.audit_rows[audited] = False  # an audit request is consumed by the step that serves it
        self.audit_step = bool(audited.any())
        self._audit_pending = self.audit_step
        self._pending_rows = audited
        audit = None
        if self.audit_step:
            self.start_audit()
            self._pending_blocks[:requests] = -(-lengths // self.block_size)
            audit = torch.from_numpy(audited.astype(np.uint8)).to(indptr.device)
            if self.audit is not None:
                self.audit.select(audited[:requests], np.arange(requests))
        if self.audit is not None:
            self.audit.set(self.audit_step)
        keep = self._keep_on(indptr.device)
        return self.compactor.compact(indptr, slots, keep, audit, self.active(), requests, n_rows, entries)


class _Step:
    """What the safe point learned about the decode batch that the attention backend is about to see."""

    def __init__(self, lengths: np.ndarray) -> None:
        self.lengths = lengths  # tokens of each request, from the host
        self.total = int(lengths.sum())  # tokens in the batch before a CUDA graph fills it


class _State:
    view: SlotView
    publisher: ScheduledPublisher
    plane: AuditPlane | None
    pointers: list[int]  # the key-cache address of each layer, for the plane
    step: _Step | None = None


_state: _State | None = None


def require_supported(scheduler: "Scheduler") -> None:
    """Refuse, with the reason, an engine whose attention the view cannot trim exactly.

    The view drops whole blocks of a decode request from the key-value list of SGLang's Triton attention kernel. That
    is exact for full causal attention, in one cache, on one tensor-parallel rank, so the view refuses an engine
    that is anything else instead of running with reads it did not trim:

    * an attention backend other than Triton (a view that no backend uses would only look installed); a hybrid
      model with state-space layers has a backend of its own, so this case refuses it too,
    * a sliding window or multi-latent attention,
    * speculative decoding, whose batches carry their own index lists,
    * tensor, pipeline, and decode context parallelism, which split the cache or the layers over processes,
    * an encoder-decoder model,
    * more running requests than the view has rows (``EKV_MAX_ROWS``).
    """
    from sglang.srt.layers.attention.triton_backend import TritonAttnBackend

    runner = scheduler.tp_worker.model_runner
    backend = runner.attn_backend
    if not isinstance(backend, TritonAttnBackend):
        raise RuntimeError(  # noqa: TRY004  # an engine the view does not support, not a wrong argument type
            f"the view supports SGLang's Triton attention backend only; this engine uses {type(backend).__name__}"
        )
    if backend.sliding_window_size is not None and backend.sliding_window_size > 0:
        raise RuntimeError("the view does not support a sliding window: it cannot trim those layers exactly")
    if backend.use_mla:
        raise RuntimeError("the view does not support multi-latent attention")
    if not scheduler.spec_algorithm.is_none():
        raise RuntimeError("the view does not support speculative decoding")
    if backend.dcp_size > 1 or scheduler.server_args.tp_size > 1 or scheduler.server_args.pp_size > 1:
        raise RuntimeError("the view does not support tensor, pipeline, or context parallelism")
    if scheduler.model_config.is_encoder_decoder:
        raise RuntimeError("the view does not support an encoder-decoder model")
    rows = int(os.environ["EKV_MAX_ROWS"])
    if scheduler.max_running_requests > rows:
        raise RuntimeError(
            f"the engine runs up to {scheduler.max_running_requests} requests and the view has {rows} rows"
            " (EKV_MAX_ROWS): set max_running_requests to at most the rows"
        )


def _create(scheduler: "Scheduler") -> _State:
    """Create the view, its mailbox, and the audit plane, from the engine's own dimensions, once."""
    require_supported(scheduler)
    runner = scheduler.tp_worker.model_runner
    rows = int(os.environ["EKV_MAX_ROWS"])
    width = -(-scheduler.model_config.context_len // BLOCK_TOKENS)
    state = _State()
    state.plane = None
    if os.environ.get("EKV_ATTN_SHARE"):
        if int(os.environ["EKV_NCAP"]) < width:
            raise RuntimeError(
                f"EKV_NCAP={os.environ['EKV_NCAP']} is below the {width} blocks of the longest context"
            )
        state.plane = AuditPlane()
        state.pointers = kv_pointers(runner.token_to_kv_pool)
    state.view = SlotView(rows, width, BLOCK_TOKENS, state.plane, runner.device)
    # SGLang ends its scheduler with SIGKILL, so the process cannot remove its mailbox, and the launcher removes it
    # after the engine stops. A mailbox that already has this pid's name is from an earlier process with the same pid.
    path = mailbox_path(os.getpid())
    path.unlink(missing_ok=True)
    ControlMailbox.create(path, rows, width).close()
    state.publisher = ScheduledPublisher(path, state.view)
    return state


def safe_point(scheduler: "Scheduler", batch: "ScheduleBatch") -> None:
    """Run before a batch executes: publish its rows, apply a completed controller generation, and note the batch."""
    assert _state is not None
    if _state.plane is not None:
        _state.plane.attach(_state.pointers)  # fills the layer table once the pass has compiled the kernel
    if not batch.forward_mode.is_decode():
        _state.step = None
        return
    publisher = _state.publisher
    request_ids = [request.rid for request in batch.reqs]
    lengths = batch.seq_lens_cpu.numpy()
    publisher.reassign_changed_requests(request_ids)
    publisher.update_requests(request_ids, -(-lengths // BLOCK_TOKENS))
    publisher.drain()
    publisher.mark_audit()
    publisher.mark_reads()
    _state.step = _Step(lengths)


def apply_view(
    backend: "TritonAttnBackend",
    rows: int,
    indptr: torch.Tensor,
    slots: torch.Tensor,
    num_kv_splits: torch.Tensor,
) -> None:
    """Rewrite one decode batch's key-value list in place, and the number of KV splits that follows from it.

    ``rows`` is the batch size the kernel sees, which a CUDA graph pads above the number of requests. Each padded
    row holds the backend's fill length of tokens and keeps them."""
    assert _state is not None
    step, view = _state.step, _state.view
    if step is None:
        raise RuntimeError("a decode batch reached the attention backend without passing the safe point")
    _state.step = None
    padding = rows - len(step.lengths)
    entries = step.total + padding * backend.get_cuda_graph_seq_len_fill_value()
    lens = view.rewrite(indptr, slots, step.lengths, rows, entries)
    backend.get_num_kv_splits(num_kv_splits, lens)


def _patch_backend(module: ModuleType) -> None:
    backend_class = module.TritonAttnBackend
    init = backend_class.init_forward_metadata
    replay = backend_class.init_forward_metadata_out_graph

    def init_forward_metadata(self: "TritonAttnBackend", forward_batch: "ForwardBatch") -> None:
        init(self, forward_batch)
        if forward_batch.forward_mode.is_decode():
            metadata = self.forward_metadata
            apply_view(
                self,
                forward_batch.batch_size,
                metadata.kv_indptr,
                metadata.kv_indices,
                metadata.num_kv_splits,
            )

    def init_forward_metadata_out_graph(
        self: "TritonAttnBackend", forward_batch: "ForwardBatch", in_capture: bool = False
    ) -> None:
        replay(self, forward_batch, in_capture)
        if not in_capture and forward_batch.forward_mode.is_decode():
            rows = forward_batch.batch_size
            apply_view(
                self, rows, self.kv_indptr, self.cuda_graph_kv_indices, self.cuda_graph_num_kv_splits[:rows]
            )

    backend_class.init_forward_metadata = init_forward_metadata
    backend_class.init_forward_metadata_out_graph = init_forward_metadata_out_graph


def _patch_scheduler(module: ModuleType) -> None:
    scheduler_class = module.Scheduler
    init = scheduler_class.__init__
    run_batch = scheduler_class.run_batch

    def wrapped_init(self: "Scheduler", *args: object, **kwargs: object) -> None:
        global _state
        init(self, *args, **kwargs)
        _state = _create(self)  # refuses an engine the view cannot serve before the engine accepts a request

    def wrapped_run_batch(
        self: "Scheduler", batch: "ScheduleBatch", *args: object, **kwargs: object
    ) -> object:
        safe_point(self, batch)
        return run_batch(self, batch, *args, **kwargs)

    scheduler_class.__init__ = wrapped_init
    scheduler_class.run_batch = wrapped_run_batch


def install() -> None:
    """Install the view and its safe point in the SGLang modules, as soon as they are imported."""
    when_imported("sglang.srt.layers.attention.triton_backend", _patch_backend)
    when_imported("sglang.srt.managers.scheduler", _patch_scheduler)
