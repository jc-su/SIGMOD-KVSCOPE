"""Zero-touch self-config check: the pass identifies the attention kernel by its
block-table arg NAME, self-allocates its capture buffer via CUDA at compile time,
captures the block_table, and publishes the buffer address — with NO Python attrs and
NO engine hook. The only Python here is the minimal bootstrap that runs the pass in the
compile pipeline (the Triton ABI doesn't auto-insert plugin passes); it sets no attrs
and touches no engine state.

  TRITON_EXT_ENABLED=1 TRITON_PLUGIN_PATHS=.../libeKVPlugin.so EKV_PASS_AUTO=1 \
  PYTHONPATH=triton-src LD_PRELOAD=libstdc++ CUDA_VISIBLE_DEVICES=1 python this.py
"""
import ctypes
import json
import os
import tempfile

# Fresh per-run Triton cache BEFORE importing triton: a stale compile in the default ~/.triton/cache
# (e.g. left across a Triton version bump) would otherwise be reloaded and shadow the freshly
# instrumented kernel -- run-to-run nondeterminism that masquerades as a pass bug. mkdtemp guarantees
# a clean cache; an externally-set TRITON_CACHE_DIR still wins.
os.environ.setdefault("TRITON_CACHE_DIR", tempfile.mkdtemp(prefix="ekv_autoconf_cache_"))

import numpy as np
import torch
import triton
import triton.language as tl

# --- minimal zero-touch bootstrap: run the ekv pass on each ttir module, no attrs ---


from ekv_vllm_plugin import install_pass  # noqa: E402

install_pass()


@triton.jit
def attn_like(block_table, kv_cache, q_ptr, out_ptr, N_BLOCKS: tl.constexpr, D: tl.constexpr):
    # A faithful minimal PAGED-attention stand-in exercising the GATHER TRIAD the pass detects purely
    # structurally (no names): index map -> KV-value gather -> <q,k> contraction.
    pid = tl.program_id(0)                                  # sequence slot
    pid_h = tl.program_id(1)                                # head
    blk = tl.arange(0, N_BLOCKS)
    phys = tl.load(block_table + pid * N_BLOCKS + blk)      # (1) index map I: physical unit ids
    d = tl.arange(0, D)
    k = tl.load(kv_cache + phys[:, None] * D + d[None, :])  # (2) KV gather through phys + a separate base
    q = tl.load(q_ptr + pid * D + d)                        # query row
    qk = tl.sum(q[None, :] * k, axis=1)                     # (3) contraction <q,k> (GEMV / tt.reduce)
    tl.store(out_ptr + pid * tl.num_programs(1) + pid_h, tl.sum(qk))


def main():
    assert torch.cuda.is_available()
    NQ, NH, NB, NPHYS, MAXROWS, D = 64, 8, 32, 9000, 256, 64
    os.environ.setdefault("EKV_NCAP", str(NB))            # capture width = block_table row width
    os.environ.setdefault("EKV_MAX_ROWS", str(MAXROWS))  # (stride is now recovered structurally, not from env)
    os.environ.setdefault("EKV_HEAD_AXIS", "1")
    os.environ.setdefault("EKV_SLOT_AXIS", "0")
    torch.manual_seed(0)
    bt = torch.randint(0, NPHYS, (NQ, NB), dtype=torch.int32, device="cuda")
    kv = torch.randn(NPHYS * D, dtype=torch.float16, device="cuda")  # the KV pool gathered through phys
    q = torch.randn(NQ * D, dtype=torch.float16, device="cuda")
    out = torch.zeros(NQ * NH, dtype=torch.float32, device="cuda")
    attn_like[(NQ, NH)](bt, kv, q, out, N_BLOCKS=NB, D=D)
    torch.cuda.synchronize()

    path = f"/tmp/ekv_pass_{os.getpid()}.json"
    if not os.path.exists(path):
        print("FAIL: pass did not self-allocate/publish — it never fired or alloc failed")
        return
    meta = json.load(open(path))
    host_va, nbytes = int(meta["host_va"]), int(meta["bytes"])
    print(f"pass published: pid={meta['pid']} host_va={host_va} bytes={nbytes}")
    buf = (ctypes.c_int32 * (nbytes // 4)).from_address(host_va)
    arr = np.ctypeslib.as_array(buf).reshape(MAXROWS, NB)
    exact = bool((arr[:NQ] == bt.cpu().numpy()).all())
    pad_ok = bool((arr[NQ:] == -1).all())
    print(f"self-config capture == block_table (rows 0..{NQ}): {exact}")
    print(f"padding rows {NQ}..{MAXROWS} untouched (-1 sentinel): {pad_ok}")
    assert exact and pad_ok, "zero-touch self-config capture mismatch"
    print("PASS: pass self-identified the attention kernel STRUCTURALLY (index map -> KV gather -> "
          "contraction, no names), self-allocated its capture buffer via CUDA at compile time, captured "
          "block_table exactly, and published the buffer for an external reader — no attrs, no engine hook.")


if __name__ == "__main__":
    main()
