"""Fixed, deterministic vLLM serving workload for the end-to-end overhead A/B. Prints the
generation-phase throughput (tokens/s) only -- model load and offload-priming are excluded
from the timer, so the paired off-vs-on difference reflects the collector's drag on serving,
not init noise. The OffloadingConnector + a small GPU pool exercise the data plane (offload
copies) and the metadata DMA the collector captures."""
import os
import time

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")


def main():
    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig
    kv = KVTransferConfig(kv_connector="OffloadingConnector", kv_role="kv_both",
                          kv_connector_extra_config={"cpu_bytes_to_use": 4 * 1024**3})
    llm = LLM(model="facebook/opt-125m", enforce_eager=True, gpu_memory_utilization=0.30,
              max_model_len=2048, enable_prefix_caching=True, kv_transfer_config=kv,
              num_gpu_blocks_override=int(os.environ.get("EKV_GPU_BLOCKS", "256")))
    nP = 32
    prefixes = [f"P {p}: " + ("tok " * 300) for p in range(nP)]
    sp = SamplingParams(max_tokens=int(os.environ.get("EKV_MAXTOK", "24")), temperature=0.0)
    batch = int(os.environ.get("EKV_BATCH", "64"))
    rounds = int(os.environ.get("EKV_ROUNDS", "5"))

    llm.generate([prefixes[i % nP] + f" w{i}" for i in range(batch)], sp)   # warmup: prime offload
    import torch
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    ntok = 0
    for r in range(rounds):
        outs = llm.generate([prefixes[i % nP] + f" r{r} q{i}" for i in range(batch)], sp)
        ntok += sum(len(o.outputs[0].token_ids) for o in outs)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    print(f"TPS={ntok / dt:.1f} TOKENS={ntok} SEC={dt:.2f}")


if __name__ == "__main__":
    main()
