"""Instrumentation anatomy: what each emission costs in REGISTERS (the Neutrino lesson -- probe registers
tax occupancy, and our post-gating statistic residual was exactly instructions+registers). Compiles the
mask-harness softmax kernel under the env the runner sets and prints n_regs/n_spills; the runner
diffs across capability combinations."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
import triton
import triton.language as tl


if os.environ.get("EKV_ENABLE"):
    from ekv_vllm_plugin import install_pass

    install_pass()


@triton.jit
def attn_softmax(block_table, kv_cache, v_cache, q_ptr, out_ptr,
                 W: tl.constexpr, D: tl.constexpr):
    pid = tl.program_id(0)
    pid_h = tl.program_id(1)
    blk = tl.arange(0, W)
    phys = tl.load(block_table + pid * W + blk)
    d = tl.arange(0, D)
    q = tl.load(q_ptr + pid * D + d).to(tl.float32)
    k = tl.load(kv_cache + phys[:, None] * D + d[None, :]).to(tl.float32)
    qk = tl.sum(q[None, :] * k, axis=1) / 8.0
    m = tl.max(qk, axis=0)
    p = tl.exp(qk - m)
    v = tl.load(v_cache + phys[:, None] * D + d[None, :]).to(tl.float32)
    num = tl.sum(p[:, None] * v, axis=0)
    out = num / tl.sum(p, axis=0)
    tl.store(out_ptr + (pid * tl.num_programs(1) + pid_h) * D + d, out)


def main():
    NQ, W, D, NPHYS = 8, 64, 512, 9000
    torch.manual_seed(0)
    bt = torch.randint(1, NPHYS, (NQ, W), dtype=torch.int32, device="cuda")
    kv = torch.randn(NPHYS * D, dtype=torch.float16, device="cuda")
    vv = torch.randn(NPHYS * D, dtype=torch.float16, device="cuda")
    q = torch.randn(NQ * D, dtype=torch.float16, device="cuda")
    out = torch.zeros(NQ * 2 * D, dtype=torch.float32, device="cuda")
    attn_softmax[(NQ, 2)](bt, kv, vv, q, out, W=W, D=D)
    torch.cuda.synchronize()

    found = []

    def find(obj, depth=0):
        if depth > 4 or id(obj) in seen:
            return
        seen.add(id(obj))
        if hasattr(obj, "n_regs"):
            meta = getattr(obj, "metadata", None)
            shared = int(getattr(meta, "shared", getattr(obj, "shared", 0)) or 0)
            found.append((int(obj.n_regs), int(getattr(obj, "n_spills", -1)), shared))
            return
        if isinstance(obj, dict):
            for v in obj.values():
                find(v, depth + 1)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                find(v, depth + 1)

    seen = set()
    for attr in ("cache", "device_caches"):
        find(getattr(attn_softmax, attr, {}), 0)
    if not found:
        print("ANATOMY error=no_compiled_kernel_found")
        sys.exit(1)
    regs, spills, shared = found[0]
    print(f"ANATOMY label={os.environ.get('EKV_ANATOMY_LABEL', '?')} "
          f"n_regs={regs} n_spills={spills} shared_bytes={shared}")


if __name__ == "__main__":
    main()
