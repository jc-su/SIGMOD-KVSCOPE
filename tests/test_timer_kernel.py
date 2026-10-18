"""End-to-end GPU test: the eKVTimer pass instruments a real Triton kernel on
this box and records per-slot clock64 cycles that grow with per-slot work.

Run under triton-src (which has the plugin ABI), with the built plugin:

  TS=/path/to/triton
  TRITON_EXT_ENABLED=1 \
  TRITON_PLUGIN_PATHS=<repo>/build/libeKVPlugin.so \
  PYTHONPATH=$TS/python:$(pwd) \
  EKV_LEGACY=1 python tests/test_timer_kernel.py   # oracle path (auto is the pass default)
"""
import os
import sys

import torch
import triton
import triton.language as tl

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from runtime.ekv_plugin import timing  # noqa: E402


@triton.jit
def busy_kernel(work_ptr, out_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    w = tl.load(work_ptr + pid)                 # per-slot work (runtime scalar)
    x = tl.full((BLOCK,), 1.0, tl.float32)
    i = 0
    while i < w:                                 # variable trip count -> variable time
        x = x * 1.0000001 + 0.5
        i += 1
    tl.store(out_ptr + pid, tl.sum(x))          # keep the loop live (no DCE)


def main():
    assert torch.cuda.is_available(), "need a CUDA GPU"
    print("GPU:", torch.cuda.get_device_name(0))
    N = 16
    work = (torch.arange(1, N + 1, device="cuda", dtype=torch.int32)) * 3000
    out = torch.empty(N, device="cuda", dtype=torch.float32)

    with timing(max_slots=N, axis=0, allowlist=["busy_kernel"]) as ctrl:
        busy_kernel[(N,)](work, out, BLOCK=256)   # compile + warm (instrumented)
        torch.cuda.synchronize()
        ctrl.reset()
        busy_kernel[(N,)](work, out, BLOCK=256)   # measured launch
        torch.cuda.synchronize()
        cyc = ctrl.read().tolist()

    print("work  :", work.cpu().tolist())
    print("cycles:", cyc)
    assert max(cyc) > 0, "no cycles recorded — the pass did not fire (check plugin/env)"
    assert cyc[-1] > cyc[0], f"cycles did not grow with work: {cyc[0]} -> {cyc[-1]}"
    mono = sum(1 for i in range(N - 1) if cyc[i] <= cyc[i + 1] * 1.20)
    print(f"monotone-ish pairs: {mono}/{N - 1}")
    print("PASS: per-slot clock64 timing recorded on the live kernel and scales with work")


if __name__ == "__main__":
    main()
