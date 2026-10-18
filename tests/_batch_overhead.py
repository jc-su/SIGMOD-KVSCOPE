"""Controlled microbench for the cuMemcpyBatchAsync uprobe cost — the offload path
eKV's kv_copy hooks. Times ONLY the enqueue call (where the uprobe fires), not the
GPU work: a tight loop of N batches of B blocks, syncing every SYNC_EVERY calls to
drain the queue (sync time EXCLUDED). The enqueue is CPU-bound, so the per-call time
is insensitive to GPU contention / co-tenants — the confound that makes the serving
tok/s number unusable on this shared box. Run it twice (collector off, then on) and
diff the median ns/call: that delta IS the uprobe's per-batch overhead, un-confounded.

  EKV_B=512 EKV_N=4000 python tests/_batch_overhead.py
"""
import ctypes
import os
import statistics
import time

import torch


class _Loc(ctypes.Structure):
    _fields_ = [("type", ctypes.c_uint), ("id", ctypes.c_int)]


class _Attr(ctypes.Structure):
    _fields_ = [("srcAccessOrder", ctypes.c_uint), ("srcLocHint", _Loc),
                ("dstLocHint", _Loc), ("flags", ctypes.c_uint)]


def main():
    B = int(os.environ.get("EKV_B", "512"))        # blocks per batch (real offload: ~hundreds)
    N = int(os.environ.get("EKV_N", "4000"))       # timed batches
    SYNC_EVERY = int(os.environ.get("EKV_SYNC_EVERY", "64"))
    ELEMS = 256                                    # 1KB/block (f32) — copy is real but small

    torch.cuda.init()
    cu = ctypes.CDLL("libcuda.so.1")

    kv = torch.zeros(B, ELEMS, dtype=torch.float32, device="cuda")
    base, stride = kv.data_ptr(), kv.stride(0) * kv.element_size()
    host = torch.zeros(B * ELEMS, dtype=torch.float32, pin_memory=True)
    hbase = host.data_ptr()

    # D2H (offload store): dst=host block i, src=device block i, B copies/batch.
    dsts = (ctypes.c_uint64 * B)(*[hbase + i * stride for i in range(B)])
    srcs = (ctypes.c_uint64 * B)(*[base + i * stride for i in range(B)])
    sizes = (ctypes.c_uint64 * B)(*([stride] * B))
    attrs = (_Attr * 1)(_Attr(srcAccessOrder=3))
    attrs_idx = (ctypes.c_size_t * B)(*([0] * B))
    fail = ctypes.c_size_t(0)
    s = torch.cuda.Stream()
    stream = ctypes.c_void_p(s.cuda_stream)

    fn = cu.cuMemcpyBatchAsync
    fn.restype = ctypes.c_int
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                   ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                   ctypes.c_void_p, ctypes.c_void_p]
    pd = ctypes.cast(dsts, ctypes.c_void_p)
    ps = ctypes.cast(srcs, ctypes.c_void_p)
    pz = ctypes.cast(sizes, ctypes.c_void_p)
    pa = ctypes.cast(attrs, ctypes.c_void_p)
    pai = ctypes.cast(attrs_idx, ctypes.c_void_p)

    def call():
        return fn(pd, ps, pz, B, pa, pai, 1, ctypes.byref(fail), stream)

    for _ in range(256):                            # warmup (prime driver + page-in)
        call()
    s.synchronize()

    samples = []
    for i in range(N):
        t0 = time.perf_counter_ns()
        err = call()
        t1 = time.perf_counter_ns()
        samples.append(t1 - t0)
        if err != 0:
            raise RuntimeError(f"cuMemcpyBatchAsync err={err} at i={i}")
        if (i + 1) % SYNC_EVERY == 0:
            s.synchronize()                         # drain queue; NOT timed
    s.synchronize()

    samples.sort()
    med = statistics.median(samples)
    p10 = samples[len(samples) // 10]
    p90 = samples[len(samples) * 9 // 10]
    mean = statistics.mean(samples)
    tag = os.environ.get("EKV_TAG", "?")
    print(f"BATCH_BENCH[{tag}] B={B} N={N}  median={med}ns  p10={p10} p90={p90}  "
          f"mean={mean:.0f}ns")


if __name__ == "__main__":
    main()
