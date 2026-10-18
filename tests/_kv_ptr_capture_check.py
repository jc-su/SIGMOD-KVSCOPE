"""Zero-touch KV-pool identification (option A): the pass captures the KV-cache base
POINTER the attention kernel is handed — by arg name, the same 'input the kernel must be
handed' as the block table — and publishes it. An external reader feeds it to
identify_kv_pool to pin the KV pool EXACTLY among captured allocations, with no size
heuristic and no offload traffic needed. Here we assert the captured pointer ==
key_cache.data_ptr(). No engine hook; the only Python is the minimal pass-bootstrap.

  TRITON_EXT_ENABLED=1 TRITON_PLUGIN_PATHS=.../libeKVPlugin.so EKV_PASS_AUTO=1 \
  PYTHONPATH=triton-src LD_PRELOAD=libstdc++ CUDA_VISIBLE_DEVICES=1 python this.py
"""
import ctypes
import json
import os

import torch
import triton
import triton.language as tl



from ekv_vllm_plugin import install_pass  # noqa: E402

install_pass()


@triton.jit
def attn_kv(block_table, key_cache, out_ptr, N_BLOCKS: tl.constexpr):
    pid = tl.program_id(0)
    pid_h = tl.program_id(1)
    blk = tl.arange(0, N_BLOCKS)
    phys = tl.load(block_table + pid * N_BLOCKS + blk)
    kv = tl.load(key_cache + pid)                 # touch key_cache so it stays a live arg
    tl.store(out_ptr + pid * tl.num_programs(1) + pid_h, tl.sum(phys).to(tl.float32) + kv)


def main():
    assert torch.cuda.is_available()
    NQ, NH, NB, NPHYS, MAXROWS = 64, 8, 32, 9000, 256
    os.environ.setdefault("EKV_NCAP", str(NB))
    os.environ.setdefault("EKV_MAX_ROWS", str(MAXROWS))
    os.environ.setdefault("EKV_HEAD_AXIS", "1")
    os.environ.setdefault("EKV_SLOT_AXIS", "0")
    torch.manual_seed(0)
    bt = torch.randint(0, NPHYS, (NQ, NB), dtype=torch.int32, device="cuda")
    key_cache = torch.zeros(NQ, dtype=torch.float32, device="cuda")   # the "KV cache" pool
    out = torch.zeros(NQ * NH, dtype=torch.float32, device="cuda")
    attn_kv[(NQ, NH)](bt, key_cache, out, N_BLOCKS=NB)
    torch.cuda.synchronize()

    path = f"/tmp/ekv_pass_{os.getpid()}.json"
    assert os.path.exists(path), "FAIL: pass did not publish — it never fired or alloc failed"
    meta = json.load(open(path))
    kv_va, kv_bytes = int(meta.get("kv_ptr_host_va", 0)), int(meta.get("kv_ptr_bytes", 0))
    print(f"pass published kv_ptr_host_va={kv_va} kv_ptr_bytes={kv_bytes}")
    assert kv_va and kv_bytes >= 8, "pass did not publish a KV pointer (kv arg not found?)"
    captured = int((ctypes.c_uint64 * 1).from_address(kv_va)[0])
    expected = key_cache.data_ptr()
    print(f"captured KV-cache ptr = {captured:#x}   key_cache.data_ptr() = {expected:#x}")
    assert captured == expected, "KV-pointer mismatch"

    # the captured pointer identifies the pool exactly, even against larger allocations
    from observe import identify_kv_pool
    allocs = [dict(addr=0x10_000000000, size=8 << 30),                # a bigger 'weights' alloc
              dict(addr=captured - 4096, size=1 << 20)]               # the 'KV pool' (smaller)
    pool = identify_kv_pool(allocs, kv_ptr=captured)
    assert pool == (captured - 4096, 1 << 20), pool
    print("identify_kv_pool(kv_ptr=captured) picks the KV pool, not the larger alloc  OK")
    print("PASS: pass captured the kernel's KV-cache base pointer (== key_cache.data_ptr()); "
          "exact pool identification, no size heuristic, no offload needed.")


if __name__ == "__main__":
    main()
