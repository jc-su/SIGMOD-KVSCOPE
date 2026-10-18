"""An oracle that watches SGLang's scheduler from inside, so a check can compare KVScope with the engine's own data.

``runtime/sglang_site`` imports this module in the scheduler when it is named in ``EKV_SITE_MODULES``. It does not
change what the engine computes, except that it switches the attention statistic on for one decode step. It writes
two files per scheduler process under the prefix in ``EKV_ORACLE_OUT``:

  <prefix>.<pid>.step.json   the last decode step: the key-value index list that the engine gave the attention kernel
                             (``kv_indptr`` and ``kv_indices``) and the step number. With the view installed this is
                             the list after the view, because the oracle wraps the view. The steps in
                             ``EKV_ORACLE_SAVE_STEPS`` are also kept in ``<prefix>.<pid>.step<N>.json``
  <prefix>.<pid>.share.json  for the decode step ``EKV_ORACLE_AUDIT_STEP`` only: the attention share of every block
                             of every row, recomputed in float64 from the queries and the cache, averaged over the
                             layers and over the query heads of the KV group that each layer records

The share needs the Python code of each layer to run, so a check that uses it must start the engine with CUDA graphs
off. A graph replays kernels and does not run the layer's Python code. The step file works with graphs on.
"""

import json
import os
from types import ModuleType

import torch

from runtime.audit import AuditPlane
from runtime.sglang_hooks import when_imported
from runtime.sglang_view import kv_pointers

OUT = os.environ["EKV_ORACLE_OUT"]
AUDIT_STEP = int(os.environ.get("EKV_ORACLE_AUDIT_STEP", "-1"))
SAVE_STEPS = {
    int(x) for x in os.environ.get("EKV_ORACLE_SAVE_STEPS", "").split(",") if x
}  # steps to keep apart
UNIT = int(os.environ.get("EKV_TOKENS_PER_BLOCK", "1"))  # the tokens in one recorded block


class _State:
    step = 0  # decode steps seen
    plane: AuditPlane | None = None
    acc: list[torch.Tensor] | None = None  # per row, the sum over layers of the share per block
    layers = 0  # layers added to ``acc``


_state = _State()


def _write(kind: str, payload: dict[str, object]) -> None:
    with open(f"{OUT}.{os.getpid()}.{kind}.json", "w") as f:
        json.dump(payload, f)


def _record_step(backend, forward_batch, indices_buffer) -> None:
    """Called after the engine built the attention metadata of a batch: remember a decode batch's index list.

    ``indices_buffer`` is the buffer that holds ``kv_indices``. In a graph replay the metadata object holds a view
    that is shorter than the buffer, and the kernel reads the buffer."""
    if not forward_batch.forward_mode.is_decode():
        return
    step = _state.step = _state.step + 1
    plane = _state.plane
    if plane is None and AUDIT_STEP >= 0:  # without an audit step, the view or nobody owns the statistic flag
        plane = _state.plane = AuditPlane()
        plane.attach(kv_pointers(backend.token_to_kv_pool))
    metadata = backend.forward_metadata
    rows = forward_batch.batch_size
    indptr = metadata.kv_indptr[: rows + 1].cpu().tolist()
    indices = indices_buffer[: indptr[-1]].cpu().tolist()
    _write("step", {"step": step, "indptr": indptr, "indices": indices})
    if step in SAVE_STEPS:
        _write(f"step{step}", {"step": step, "indptr": indptr, "indices": indices})
    if plane is not None:
        plane.set(step == AUDIT_STEP)  # the statistic is recorded in the step that follows this call
    _state.acc = None


def _reference(backend, q, layer, forward_batch) -> None:
    """Add this layer's true attention share per block to the running mean, and write it after the last layer."""
    metadata = backend.forward_metadata
    rows = forward_batch.batch_size
    indptr = metadata.kv_indptr[: rows + 1].cpu().tolist()
    keys = backend.token_to_kv_pool.get_key_buffer(layer.layer_id)
    group_size = layer.tp_q_head_num // layer.tp_k_head_num
    group = layer.layer_id % layer.tp_k_head_num  # the pass records KV group (layer mod groups)
    queries = q.reshape(-1, layer.tp_q_head_num, layer.qk_head_dim)
    per_row: list[torch.Tensor] = []
    for row in range(rows):
        slots = metadata.kv_indices[indptr[row] : indptr[row + 1]]
        k = keys[slots, group].double()
        score = (queries[row, group * group_size : (group + 1) * group_size].double() @ k.T) * layer.scaling
        weight = torch.softmax(score, dim=1)  # [heads of the group, positions]
        pad = -weight.shape[1] % UNIT
        blocks = torch.nn.functional.pad(weight, (0, pad)).reshape(group_size, -1, UNIT).sum(dim=2)
        per_row.append(blocks.mean(dim=0))
    acc = _state.acc
    _state.acc = per_row if acc is None else [a + b for a, b in zip(acc, per_row, strict=True)]
    _state.layers = 1 if acc is None else _state.layers + 1
    if _state.layers == backend.token_to_kv_pool.layer_num:
        lengths = [indptr[row + 1] - indptr[row] for row in range(rows)]
        shares = [(a / _state.layers).tolist() for a in _state.acc]
        _write("share", {"step": _state.step, "layers": _state.layers, "lengths": lengths, "share": shares})


def _patch(module: ModuleType) -> None:
    backend_class = module.TritonAttnBackend
    init = backend_class.init_forward_metadata
    replay = backend_class.init_forward_metadata_out_graph
    decode = backend_class.forward_decode

    def init_forward_metadata(self, forward_batch):
        init(self, forward_batch)
        _record_step(self, forward_batch, self.forward_metadata.kv_indices)

    def init_forward_metadata_out_graph(self, forward_batch, in_capture=False):
        replay(self, forward_batch, in_capture)
        if not in_capture:
            _record_step(self, forward_batch, self.cuda_graph_kv_indices)

    def forward_decode(self, q, k, v, layer, forward_batch, *args, **kwargs):
        output = decode(self, q, k, v, layer, forward_batch, *args, **kwargs)
        if _state.step == AUDIT_STEP:
            _reference(self, q, layer, forward_batch)
        return output

    backend_class.init_forward_metadata = init_forward_metadata
    backend_class.init_forward_metadata_out_graph = init_forward_metadata_out_graph
    backend_class.forward_decode = forward_decode


when_imported("sglang.srt.layers.attention.triton_backend", _patch)
