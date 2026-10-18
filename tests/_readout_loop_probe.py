"""Workload for the eBPF SYNC-ANCHORED windowing loop (gap 1), on a SINGLE-variant kernel so the
eBPF single-buffer reader (ekv_readout) is unambiguous -- real vLLM recompiles attention per shape
(many variants) whose reader-side merge is the Python /proc path; here one paged-attention kernel
compiles once, publishes one buffer, and is launched in a loop with a sync after each launch. So
ekv_readout's per-launch timer read is sync-anchored (the prior launch completed), the block table
is constant (clean windows), and the cumulative cycles grow per launch -- exactly the stream
window_series expects. Planted per-request context makes the cycles track work. GPU 1 via the runner.
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import torch
import triton
import triton.language as tl


from ekv_vllm_plugin import install_pass  # noqa: E402

install_pass()


@triton.jit
def attn_varwork(block_table, seq_nblks, kv_cache, q_ptr, out_ptr,
                 QB: tl.constexpr, W: tl.constexpr, D: tl.constexpr, REPS: tl.constexpr):
    pid = tl.program_id(0)
    pid_h = tl.program_id(1)
    seq = pid // QB                                        # kernel's own request index
    blk = tl.arange(0, W)
    phys = tl.load(block_table + seq * W + blk)
    nb = tl.load(seq_nblks + seq)
    d = tl.arange(0, D)
    q = tl.load(q_ptr + seq * D + d)
    acc = 0.0
    for rep in range(0, REPS):
        for i in range(0, nb):
            pb = tl.sum(tl.where(blk == i, phys, 0)) + rep * 0
            k = tl.load(kv_cache + pb * D + d)
            acc += tl.sum(q * k)
    tl.store(out_ptr + pid * tl.num_programs(1) + pid_h, acc)


NQ, QB, NH, W, D, NPHYS, REPS = 8, 2, 2, 64, 64, 9000, 8
os.environ.setdefault("EKV_NCAP", "64")
os.environ.setdefault("EKV_MAX_ROWS", "32")
torch.manual_seed(0)
bt = torch.randint(1, NPHYS, (NQ, W), dtype=torch.int32, device="cuda")
bt[:, 0] = torch.arange(1, NQ + 1, dtype=torch.int32)     # distinct first-block anchors
nblks = torch.tensor([2 * (r + 1) for r in range(NQ)], dtype=torch.int32, device="cuda")
kv = torch.randn(NPHYS * D, dtype=torch.float16, device="cuda")
q = torch.randn(NQ * D, dtype=torch.float16, device="cuda")
out = torch.zeros(NQ * QB * NH, dtype=torch.float32, device="cuda")
attn_varwork[(NQ * QB, NH)](bt, nblks, kv, q, out, QB=QB, W=W, D=D, REPS=REPS)   # compile+publish
torch.cuda.synchronize()

print(f"EKV_PID {os.getpid()}", flush=True)
print("EKV_NBLKS " + ",".join(str(int(x)) for x in nblks.cpu().tolist()), flush=True)
print("EKV_READY", flush=True)
time.sleep(8)                                             # eBPF attach window

print("EKV_GEN_START", flush=True)
for _ in range(500):                                     # many synced launches -> many snapshots
    attn_varwork[(NQ * QB, NH)](bt, nblks, kv, q, out, QB=QB, W=W, D=D, REPS=REPS)
    torch.cuda.synchronize()                             # sync-anchor each launch
    time.sleep(0.006)
print("EKV_GEN_DONE", flush=True)
time.sleep(1)
