"""Validate the arithmetic-index path for a non-paged, contiguous attention kernel: the kernel
attention kernel addresses its KV by PURE ARITHMETIC -- kv_ptr + pid_b*stride_b + ... -- with NO index
array (no block_table / kv_indices). The index path (findIndexLookup) finds no loaded index and the pass
uses findArithmeticIndex to recover the closed-form request-to-region map's two scalars: the KV
base and the per-request stride. PASS = the pass publishes row_form=arith with stride==stride_b and
kv_base==kv.data_ptr() (no GPU capture array -- the map is the formula).

  TRITON_EXT_ENABLED=1 TRITON_PLUGIN_PATHS=.../libeKVPlugin.so EKV_PASS_AUTO=1 \
  PYTHONPATH=triton-src/python LD_PRELOAD=libstdc++ CUDA_VISIBLE_DEVICES=1 python this.py
"""
import ctypes
import json
import os
import tempfile

os.environ.setdefault("TRITON_CACHE_DIR", tempfile.mkdtemp(prefix="ekv_arith_cache_"))
os.environ.setdefault("EKV_ARITH_ATTENTION", "1")  # explicit backend selection is required

import torch
import triton
import triton.language as tl


from ekv_vllm_plugin import install_pass  # noqa: E402

install_pass()


@triton.jit
def attn_contig(q_ptr, kv_ptr, out_ptr, stride_b, N: tl.constexpr, D: tl.constexpr):
    # PURE-ARITHMETIC paged-free attention: request pid_b's KV is kv_ptr + pid_b*stride_b + ...
    # No index map -- the request->region map is the closed form, not a table.
    pid_b = tl.program_id(0)                                   # the request (slot axis 0)
    offs_n = tl.arange(0, N)
    offs_d = tl.arange(0, D)
    k = tl.load(kv_ptr + pid_b * stride_b + offs_n[:, None] * D + offs_d[None, :])  # (2) KV gather, arithmetic addr
    q = tl.load(q_ptr + pid_b * D + offs_d)
    qk = tl.sum(q[None, :].to(tl.float32) * k.to(tl.float32), axis=1)               # (3) contraction <q,k>
    tl.store(out_ptr + pid_b, tl.sum(qk))


def main():
    assert torch.cuda.is_available()
    NB, N, D = 16, 32, 64
    STRIDE_B = N * D                                    # per-request stride (contiguous KV): 2048 elements
    os.environ.setdefault("EKV_MAX_ROWS", "64")
    torch.manual_seed(0)
    kv = torch.randn(NB * STRIDE_B, dtype=torch.float16, device="cuda")
    q = torch.randn(NB * D, dtype=torch.float16, device="cuda")
    out = torch.zeros(NB, dtype=torch.float32, device="cuda")
    attn_contig[(NB,)](q, kv, out, STRIDE_B, N=N, D=D)
    torch.cuda.synchronize()

    path = f"/tmp/ekv_pass_{os.getpid()}.json"
    if not os.path.exists(path):
        print("FAIL: pass did not publish — arithmetic index_lookup not detected or alloc failed")
        return
    meta = json.load(open(path))
    print(f"published: row_form={meta.get('row_form')} kv_ptr_host_va={meta.get('kv_ptr_host_va')} "
          f"arith_stride_host_va={meta.get('arith_stride_host_va')}")
    if meta.get("row_form") != "arith":
        print(f"FAIL: row_form={meta.get('row_form')} (expected arith)")
        return
    pub_stride = ctypes.c_int64.from_address(int(meta["arith_stride_host_va"])).value
    pub_kv = ctypes.c_uint64.from_address(int(meta["kv_ptr_host_va"])).value
    print(f"published stride={pub_stride} (expected {STRIDE_B})")
    print(f"published kv_base=0x{pub_kv:x} (kv.data_ptr=0x{kv.data_ptr():x})")

    # the production zero-touch reader (pure-stdlib /proc readout) consumes the arith map
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    from observe import zero_touch
    region = zero_touch.read_pass_region(os.getpid())
    kvb = zero_touch.read_pass_kv_ptr(os.getpid())
    print(f"reader region[1]={region.get(1)} region[2]={region.get(2)} (expect {STRIDE_B}, {2 * STRIDE_B}); "
          f"reader kv_base=0x{(kvb or 0):x}")
    ok = (pub_stride == STRIDE_B and pub_kv == kv.data_ptr()
          and region.get(1) == STRIDE_B and region.get(2) == 2 * STRIDE_B and kvb == kv.data_ptr())
    print("PASS: arithmetic-mu index_lookup — pass recovered + published the closed-form map "
          "region(seq)=kv_base+seq*stride for a NON-PAGED kernel (no index array); zero_touch."
          "read_pass_region reconstructs each request's KV region. stride + base match end-to-end."
          if ok else "FAIL: published/reader stride/base mismatch")


if __name__ == "__main__":
    main()
