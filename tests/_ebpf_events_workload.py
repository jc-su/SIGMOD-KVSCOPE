"""Event-based offload workload — the pattern real vLLM/LMCache use (record a
CUDA event after the copy, then synchronize on the EVENT, never on the stream).
Exercises the event-completion path (cudaEventRecord + cudaEventSynchronize)."""
import time

import torch

N = 1 << 20  # 4 MiB float32
s = torch.cuda.Stream()
g = torch.randn(N, device="cuda")
h = torch.empty(N, dtype=torch.float32, pin_memory=True)
for _ in range(5):
    ev = torch.cuda.Event()
    with torch.cuda.stream(s):
        h.copy_(g, non_blocking=True)  # cudaMemcpyAsync D2H on stream s
        ev.record()                    # cudaEventRecord(ev, s)
    ev.synchronize()                   # cudaEventSynchronize(ev) -> completion
torch.cuda.synchronize()
time.sleep(1.0)
print(f"event workload: 5 D2H copies of {N * 4} bytes, completed via cudaEvent")
