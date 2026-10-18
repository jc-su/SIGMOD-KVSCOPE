"""Join test workload: allocate a KV-cache-like tensor (one block per row),
export the block->address map (what the control plane would provide), then D2H
cudaMemcpyAsync specific blocks (the offload). The collector captures the copies
by address; the checker resolves them back to block ids."""
import json
import time

import torch

N_BLOCKS = 64
BLOCK_ELEMS = 4096  # 16 KiB/block (float32) — KV-block-ish
kv = torch.zeros(N_BLOCKS, BLOCK_ELEMS, dtype=torch.float32, device="cuda")
base = kv.data_ptr()
stride = kv.stride(0) * kv.element_size()  # bytes per block

copied = [2, 5, 7, 13, 31, 63]
host = torch.empty(BLOCK_ELEMS, dtype=torch.float32, pin_memory=True)
torch.cuda.synchronize()
for bid in copied:
    host.copy_(kv[bid], non_blocking=True)  # D2H cudaMemcpyAsync, src = base+bid*stride
torch.cuda.synchronize()

json.dump({"base": base, "stride": stride, "n_blocks": N_BLOCKS, "copied": copied},
          open("/tmp/ekv_kvmap.json", "w"))
time.sleep(1.0)  # let the collector flush
print(f"workload: kv base={base:#x} stride={stride} copied_blocks={copied}")
