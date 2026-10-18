"""GPU validation of the REGION-AWARE reachability generalization (the generality lesson): the eKV index_lookup
walk now crosses loop boundaries (mlir::LoopLikeOpInterface), so an index map whose load BASE is a
loop-carried iter-arg is detected + captured -- a case the old narrow (stop-at-block-arg) walk skipped.

The kernel `attn_loopcarried` advances the block_table row pointer each loop iteration (`bt = bt + BLK`,
a loop-carried base) and loads the index through it. Without loop-crossing the index load's base is a
block arg with no defining op -> derivesDirectly returns false -> idxArg = -1 -> the pass skips. With it,
derivesDirectly / addptrOffsetOnBase cross the iter-arg to its init (block_table + pid*NB) and recover
the index_lookup. PASS = the pass captured block_table[pid] exactly via the loop-carried base.

  TRITON_EXT_ENABLED=1 TRITON_PLUGIN_PATHS=.../libeKVPlugin.so EKV_PASS_AUTO=1 EKV_ENABLE=1 \
  PYTHONPATH=triton-src LD_PRELOAD=libstdc++ CUDA_VISIBLE_DEVICES=1 python this.py
"""
import ctypes
import json
import os

import numpy as np
import torch
import triton
import triton.language as tl



from ekv_vllm_plugin import install_pass  # noqa: E402

install_pass()


@triton.jit
def attn_loopcarried(block_table, kv_cache, q_ptr, out_ptr,
                     NB: tl.constexpr, BLK: tl.constexpr, D: tl.constexpr):
    pid = tl.program_id(0)
    bt = block_table + pid * NB             # row base -- advanced in the loop (loop-carried iter-arg)
    blk = tl.arange(0, BLK)
    d = tl.arange(0, D)
    acc = tl.zeros([D], tl.float32)
    for _ in range(0, NB, BLK):
        phys = tl.load(bt + blk)            # VECTOR index load whose BASE is the loop iter-arg
        k = tl.load(kv_cache + phys[:, None] * D + d[None, :])   # KV gather through phys
        q = tl.load(q_ptr + pid * D + d)
        acc += tl.sum(q[None, :] * k, 0)    # <q,k> contraction
        bt = bt + BLK                        # advance the base (loop-carried)
    tl.store(out_ptr + pid * D + d, acc)


def main():
    assert torch.cuda.is_available()
    NQ, NB, BLK, NPHYS, D, MAXROWS = 16, 32, 16, 4000, 32, 64
    os.environ.setdefault("EKV_NCAP", str(NB))
    os.environ.setdefault("EKV_MAX_ROWS", str(MAXROWS))
    os.environ.setdefault("EKV_HEAD_AXIS", "-1")
    os.environ.setdefault("EKV_SLOT_AXIS", "0")
    torch.manual_seed(0)
    bt = torch.randint(0, NPHYS, (NQ, NB), dtype=torch.int32, device="cuda")
    kv = torch.randn(NPHYS * D, dtype=torch.float16, device="cuda")
    q = torch.randn(NQ * D, dtype=torch.float16, device="cuda")
    out = torch.zeros(NQ * D, dtype=torch.float32, device="cuda")
    attn_loopcarried[(NQ,)](bt, kv, q, out, NB=NB, BLK=BLK, D=D)
    torch.cuda.synchronize()

    path = f"/tmp/ekv_pass_{os.getpid()}.json"
    if not os.path.exists(path):
        print("FAIL: pass did not fire on the loop-carried-index kernel "
              "(loop-crossing missing -> the index base, a loop iter-arg, was not reached)")
        return
    meta = json.load(open(path))
    arr = np.ctypeslib.as_array(
        (ctypes.c_int32 * (int(meta["bytes"]) // 4)).from_address(int(meta["host_va"]))
    ).reshape(MAXROWS, NB)
    exact = bool((arr[:NQ] == bt.cpu().numpy()).all())
    pad_ok = bool((arr[NQ:] == -1).all())
    print(f"capture == block_table (rows 0..{NQ}, via the loop-carried base): {exact}")
    print(f"padding rows untouched (-1): {pad_ok}")
    assert exact and pad_ok, "loop-carried capture mismatch"
    print("PASS: region-aware (loop-crossing) index_lookup walk detected + captured an index map whose load "
          "base is a loop-carried iter-arg -- a case the narrow walk skipped. One K(addr) principle, "
          "now region-aware.")


if __name__ == "__main__":
    main()
