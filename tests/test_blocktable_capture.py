"""End-to-end GPU test: the eKV pass captures, GPU-side, the block_table row each
program reads — recovering request<->block where the map is *materialized* (so it
works even when the engine keeps that map device-resident, e.g. SGLang's
req_to_token, which pure host-side eBPF cannot invert).

This is the wire-up proof for the production path: the same capture the
tests/_gpu_blocktable_capture.py POC validated by hand, but injected into a real,
unmodified Triton kernel by the eKV MLIR pass plugin via runtime.ekv_plugin —
no kernel edit, no launch-site change, no NVBit.

Run under triton-src (which has the plugin ABI), with the built plugin:

  TS=/path/to/triton
  TRITON_EXT_ENABLED=1 \
  TRITON_PLUGIN_PATHS=<repo>/build/libeKVPlugin.so \
  PYTHONPATH=$TS/python:$(pwd) \
  LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
  EKV_LEGACY=1 \
  python tests/test_blocktable_capture.py

EKV_LEGACY=1 because this test drives the EXPLICIT-CONFIG oracle path (Python-stamped attrs
= ground truth); since auto became the pass's default, that path is opt-in. The structural
zero-touch path is proven separately by tests/_pass_autoconfig_check.py -- its POC kernel
here has no contraction, so the auto detector rightly refuses it.
"""
import os
import sys

import torch
import triton
import triton.language as tl

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from runtime.ekv_plugin import capturing, ControlBlock  # noqa: E402


@triton.jit
def gather_1d(bt_ptr, out_ptr, N_BLOCKS: tl.constexpr):
    pid = tl.program_id(0)                             # slot == a request's query block
    blk = tl.arange(0, N_BLOCKS)
    phys = tl.load(bt_ptr + pid * N_BLOCKS + blk)      # the kernel reads block_table[pid,:]
    tl.store(out_ptr + pid, tl.sum(phys))             # consume it so the load isn't DCE'd


@triton.jit
def gather_2d(bt_ptr, out_ptr, N_BLOCKS: tl.constexpr):
    pid_q = tl.program_id(0)                           # query block (slot)
    pid_h = tl.program_id(1)                           # attention head
    blk = tl.arange(0, N_BLOCKS)
    phys = tl.load(bt_ptr + pid_q * N_BLOCKS + blk)
    tl.store(out_ptr + pid_q * tl.num_programs(1) + pid_h, tl.sum(phys))


@triton.jit
def gather_pad(bt_ptr, out_ptr, N_BLOCKS: tl.constexpr, NROWS: tl.constexpr):
    # Mimics vLLM's unified kernel: the grid (program count) is an UPPER BOUND on the
    # real row count, so padding programs pid >= NROWS exist. The kernel clamps its own
    # read (as vLLM clamps seq_idx); the eKV capture must instead be masked off for
    # those rows (max_rows) so it neither reads past block_table nor writes past buf.
    pid = tl.program_id(0)
    row = tl.minimum(pid, NROWS - 1)
    blk = tl.arange(0, N_BLOCKS)
    phys = tl.load(bt_ptr + row * N_BLOCKS + blk)
    tl.store(out_ptr + pid, tl.sum(phys))


def main():
    assert torch.cuda.is_available(), "need a CUDA GPU"
    print("GPU:", torch.cuda.get_device_name(0))
    NQ, NH, N_BLOCKS, NPHYS = 64, 8, 32, 9000
    torch.manual_seed(0)
    bt = torch.randint(0, NPHYS, (NQ, N_BLOCKS), dtype=torch.int32, device="cuda")
    bt_host = bt.cpu()
    out = torch.empty(NQ * NH, dtype=torch.float32, device="cuda")

    # --- (1) 1D grid: one program per slot, no head axis -------------------------
    with capturing(max_slots=NQ, ncap=N_BLOCKS, bt_arg=0, row_stride=N_BLOCKS,
                   axis=0, allowlist=["gather_1d"]) as cap:
        gather_1d[(NQ,)](bt, out, N_BLOCKS=N_BLOCKS)   # compile + warm (instrumented)
        torch.cuda.synchronize()
        cap.reset()
        gather_1d[(NQ,)](bt, out, N_BLOCKS=N_BLOCKS)   # measured launch
        torch.cuda.synchronize()
        got1 = cap.read()
    assert (got1 != -1).any(), "nothing captured — the pass did not fire (check plugin/env)"
    ok1 = torch.equal(got1, bt_host)
    print(f"[1D]            captured == block_table : {ok1}   ({NQ}x{N_BLOCKS} rows)")
    assert ok1, "1D capture mismatch"

    # --- (2) 2D grid: head-deduped masked store (only pid_h==0 writes) -----------
    with capturing(max_slots=NQ, ncap=N_BLOCKS, bt_arg=0, row_stride=N_BLOCKS,
                   axis=0, head_axis=1, allowlist=["gather_2d"]) as cap:
        gather_2d[(NQ, NH)](bt, out, N_BLOCKS=N_BLOCKS)
        torch.cuda.synchronize()
        cap.reset()
        gather_2d[(NQ, NH)](bt, out, N_BLOCKS=N_BLOCKS)
        torch.cuda.synchronize()
        got2 = cap.read()
    ok2 = torch.equal(got2, bt_host)
    print(f"[2D head-dedup] captured == block_table : {ok2}   (grid {NQ}x{NH})")
    assert ok2, "2D head-deduped capture mismatch"

    # --- (3) combined: clock64 timer + capture in ONE install (both faces) -------
    ctrl = ControlBlock(max_slots=NQ, device="cuda")
    with capturing(max_slots=NQ, ncap=N_BLOCKS, bt_arg=0, row_stride=N_BLOCKS,
                   axis=0, allowlist=["gather_1d"], ctrl=ctrl) as cap:
        gather_1d[(NQ,)](bt, out, N_BLOCKS=N_BLOCKS)
        torch.cuda.synchronize()
        cap.reset(); ctrl.reset()
        gather_1d[(NQ,)](bt, out, N_BLOCKS=N_BLOCKS)
        torch.cuda.synchronize()
        got3 = cap.read(); cyc = ctrl.read()
    ok3 = torch.equal(got3, bt_host)
    timed = int((cyc > 0).sum())
    print(f"[timer+capture] capture ok: {ok3}, slots timed: {timed}/{NQ}")
    assert ok3, "combined capture mismatch"
    assert timed > 0, "combined: no cycles recorded (timer half did not fire)"

    # --- (4) max_rows mask: grid bigger than block_table rows (vLLM-unified-like) --
    NROWS, NPROG = 48, 64                              # 16 padding programs, pid >= NROWS
    btp = torch.randint(0, NPHYS, (NROWS, N_BLOCKS), dtype=torch.int32, device="cuda")
    outp = torch.empty(NPROG, dtype=torch.float32, device="cuda")
    with capturing(max_slots=NPROG, ncap=N_BLOCKS, bt_arg=0, row_stride=N_BLOCKS,
                   axis=0, max_rows=NROWS, allowlist=["gather_pad"]) as cap:
        gather_pad[(NPROG,)](btp, outp, N_BLOCKS=N_BLOCKS, NROWS=NROWS)
        torch.cuda.synchronize()
        cap.reset()
        gather_pad[(NPROG,)](btp, outp, N_BLOCKS=N_BLOCKS, NROWS=NROWS)
        torch.cuda.synchronize()
        got4 = cap.read()
    ok_rows = torch.equal(got4[:NROWS], btp.cpu())     # real rows captured
    ok_pad = bool((got4[NROWS:] == -1).all())          # padding rows left untouched (no OOB)
    print(f"[max_rows mask] real rows ok: {ok_rows}, padding untouched: {ok_pad} "
          f"(grid {NPROG} > rows {NROWS})")
    assert ok_rows and ok_pad, "max_rows mask failed (padding programs not masked)"

    print("PASS: GPU-side request<->block capture wired end-to-end through the eKV "
          "pass plugin — 1D, head-deduped 2D, combined with the clock64 timer, and the "
          "max_rows padding mask. The device-resident index_lookup is recovered with no "
          "kernel/launch edit.")


if __name__ == "__main__":
    main()
