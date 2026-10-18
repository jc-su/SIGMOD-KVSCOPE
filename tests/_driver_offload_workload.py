"""Trigger a real, successful driver-API cuMemcpyBatchAsync (vLLM's KV-offload
call) on an explicit stream, then sync it — so the collector captures both the
batch copies AND, via the stream sync, their completion latency."""
import ctypes
import json
import time

import torch


class _Loc(ctypes.Structure):
    _fields_ = [("type", ctypes.c_uint), ("id", ctypes.c_int)]


class _Attr(ctypes.Structure):
    _fields_ = [("srcAccessOrder", ctypes.c_uint), ("srcLocHint", _Loc),
                ("dstLocHint", _Loc), ("flags", ctypes.c_uint)]


torch.cuda.init()
cu = ctypes.CDLL("libcuda.so.1")

N, ELEMS = 64, 4096
COPIED = [2, 5, 7, 13, 31]
kv = torch.zeros(N, ELEMS, dtype=torch.float32, device="cuda")   # GPU KV cache
base, stride = kv.data_ptr(), kv.stride(0) * kv.element_size()
host = torch.zeros(len(COPIED) * ELEMS, dtype=torch.float32, pin_memory=True)
hbase = host.data_ptr()

k = len(COPIED)
dsts = (ctypes.c_uint64 * k)(*[hbase + i * stride for i in range(k)])
srcs = (ctypes.c_uint64 * k)(*[base + b * stride for b in COPIED])
sizes = (ctypes.c_uint64 * k)(*([stride] * k))
attrs = (_Attr * 1)(_Attr(srcAccessOrder=3))     # CU_MEMCPY_SRC_ACCESS_ORDER_ANY
attrs_idx = (ctypes.c_size_t * k)(*([0] * k))
fail = ctypes.c_size_t(0)
s = torch.cuda.Stream()

fn = cu.cuMemcpyBatchAsync
fn.restype = ctypes.c_int
fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
               ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
               ctypes.c_void_p, ctypes.c_void_p]
err = fn(ctypes.cast(dsts, ctypes.c_void_p), ctypes.cast(srcs, ctypes.c_void_p),
         ctypes.cast(sizes, ctypes.c_void_p), k,
         ctypes.cast(attrs, ctypes.c_void_p), ctypes.cast(attrs_idx, ctypes.c_void_p),
         1, ctypes.byref(fail), ctypes.c_void_p(s.cuda_stream))
s.synchronize()  # cudaStreamSynchronize(s) -> completion for the batch

json.dump({"base": base, "stride": stride, "n": N, "copied": COPIED},
          open("/tmp/ekv_drvmap.json", "w"))
time.sleep(1.0)
print(f"cuMemcpyBatchAsync err={err} (0=SUCCESS) offloaded {COPIED} on stream "
      f"{s.cuda_stream:#x}; synced")
