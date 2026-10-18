"""Workload for the eBPF test: 5 device->host cudaMemcpyAsync copies of 4 MB
each, each on an explicit stream followed by stream.synchronize() — the KV
offload/swap pattern (copy then wait), so the collector sees one enqueue and one
completion per copy."""
import time

import torch

N = 1 << 20  # 4 MB of float32
s = torch.cuda.Stream()
g = torch.randn(N, device="cuda")
h = torch.empty(N, dtype=torch.float32, pin_memory=True)
for _ in range(5):
    with torch.cuda.stream(s):
        h.copy_(g, non_blocking=True)  # D2H cudaMemcpyAsync on stream s
    s.synchronize()                    # cudaStreamSynchronize(s) -> completion
torch.cuda.synchronize()
time.sleep(1.0)  # let the collector's ring-buffer poll flush
print(f"workload done: 5 D2H copies of {N * 4} bytes each (copy+sync)")
