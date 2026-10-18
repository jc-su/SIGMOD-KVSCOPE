"""A probe that sets the view's policy from inside SGLang's scheduler, for ``_sglang_view_check.py``.

``runtime/sglang_site`` imports this module in the scheduler when it is named in ``EKV_SITE_MODULES``. At decode step
``EKV_PROBE_POLICY_STEP`` it commits a policy that keeps half of every request's blocks (pairs of blocks, two kept for
every two dropped), and at step ``EKV_PROBE_AUDIT_STEP`` it asks for an audit of rows 0 and 1, so that those rows read
in full and the others read the view. It writes the tokens each row held, and the tokens the view gave the kernel, at
the audit step to ``<EKV_PROBE_OUT>.<pid>.json``. The commits are the ones a controller makes through the mailbox, written to the view
directly, as ``tests/_view_kernel_check.py`` does for vLLM.
"""

import ctypes
import json
import os
import time

import numpy as np
import torch

from observe.tile_shares import share_descriptor
from runtime import sglang_view

POLICY_STEP = int(os.environ["EKV_PROBE_POLICY_STEP"])
AUDIT_STEP = int(os.environ["EKV_PROBE_AUDIT_STEP"])
AUDITED_ROWS = (0, 1)
OUT = os.environ["EKV_PROBE_OUT"]

_decode_steps = 0
# the full lengths of the rows of the audit step, until the view has built that step
_audit_step: dict[str, list[int]] = {}
safe_point = sglang_view.safe_point
apply_view = sglang_view.apply_view


def clear_statistic() -> None:
    """Set every cell of the statistic table to NaN, which is what the reader takes for a cell no launch wrote."""
    state = sglang_view._state
    assert state is not None and state.plane is not None
    entry = share_descriptor(os.getpid())
    size = (
        int(entry["tile_lse_layers"])
        * int(entry["max_rows"])
        * int(entry["ncap"])
        * int(entry["tile_lse_group_rows"])
        * 4
    )
    torch.cuda.synchronize()
    code = state.plane._cuda.cudaMemset(
        ctypes.c_void_p(int(entry["tile_lse_dev"])), 0xFF, ctypes.c_size_t(size)
    )
    if code != 0:
        raise RuntimeError(f"cudaMemset failed with code {code}")


def probing_safe_point(scheduler, batch) -> None:
    global _decode_steps
    safe_point(scheduler, batch)
    if not batch.forward_mode.is_decode():
        return
    _decode_steps += 1
    state = sglang_view._state
    assert state is not None
    view = state.view
    if _decode_steps == POLICY_STEP:
        pair = np.arange(view.width) // 2  # keep pair p of every row when p mod 4 < 2: half of the blocks
        view.write_row_controls(np.tile(((pair % 4) < 2).astype(np.uint8), (view.rows, 1)))
        view.write_lease_deadline(time.monotonic_ns() + 3_600 * 10**9)
    if _decode_steps == AUDIT_STEP:
        clear_statistic()  # a request's first decode step is an audit, so the table holds that step's longer records
        audit = np.zeros(view.rows, dtype=bool)
        audit[list(AUDITED_ROWS)] = True
        view.write_audit_rows(audit)
        _audit_step["full_lens"] = batch.seq_lens_cpu.tolist()


def probing_apply_view(backend, rows, indptr, slots, num_kv_splits) -> None:
    """After the view rewrote the batch of the audit step, write the tokens it gave each request."""
    apply_view(backend, rows, indptr, slots, num_kv_splits)
    full = _audit_step.pop("full_lens", None)
    if full is not None:
        given = torch.diff(indptr[: rows + 1])[: len(full)].cpu().tolist()
        with open(f"{OUT}.{os.getpid()}.json", "w") as f:
            json.dump({"step": _decode_steps, "full_lens": full, "given": given}, f)


sglang_view.safe_point = probing_safe_point  # ty: ignore[invalid-assignment]  # the probe wraps the function
sglang_view.apply_view = probing_apply_view  # ty: ignore[invalid-assignment]  # the probe wraps the function
