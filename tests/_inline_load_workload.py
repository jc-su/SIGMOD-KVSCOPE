"""Trigger a real synchronous H2D reload (the inline-load path a non-async connector
takes: CPU->GPU back into KV blocks) via cuMemcpyBatchAsync on an explicit stream, then
sync — so the collector captures the batch AND its completion latency (the host-paid
load wait that is the inline-load latency). Mirror of _driver_offload_workload but in the
LOAD direction (dst = GPU KV blocks, src = host)."""
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
LOADED = [3, 6, 9, 17, 23]                       # GPU blocks being filled from host
kv = torch.zeros(N, ELEMS, dtype=torch.float32, device="cuda")    # GPU KV cache
base, stride = kv.data_ptr(), kv.stride(0) * kv.element_size()
host = torch.zeros(len(LOADED) * ELEMS, dtype=torch.float32, pin_memory=True)
hbase = host.data_ptr()

k = len(LOADED)
dsts = (ctypes.c_uint64 * k)(*[base + b * stride for b in LOADED])   # dst = GPU block
srcs = (ctypes.c_uint64 * k)(*[hbase + i * stride for i in range(k)])  # src = host
sizes = (ctypes.c_uint64 * k)(*([stride] * k))
attrs = (_Attr * 1)(_Attr(srcAccessOrder=3))
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
s.synchronize()

json.dump({"base": base, "stride": stride, "n": N, "loaded": LOADED},
          open("/tmp/ekv_inmap.json", "w"))
time.sleep(1.0)
print(f"cuMemcpyBatchAsync H2D err={err} (0=SUCCESS) reloaded {LOADED} on stream "
      f"{s.cuda_stream:#x}; synced")
