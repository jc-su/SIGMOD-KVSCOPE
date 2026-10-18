"""KV-offload workload: vLLM + SimpleCPUOffloadConnector (DmaCopyBackend =
cuMemcpyBatchAsync) under GPU-KV pressure with recurring prefixes, so blocks are
stored→evicted→offloaded to CPU (D2H) and reloaded on a later prefix hit (H2D) — the
real offload copies the eKV kv_copy collector captures. Prints tokens/s for the
overhead with/without comparison."""
import os
import sys
import time

os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vllm import LLM, SamplingParams  # noqa: E402
from vllm.config import KVTransferConfig  # noqa: E402


def main():
    kv = KVTransferConfig(
        kv_connector="SimpleCPUOffloadConnector", kv_role="kv_both",
        kv_connector_extra_config={"cpu_bytes_to_use": 4 * 1024**3})
    llm = LLM(model="facebook/opt-125m", enforce_eager=True, gpu_memory_utilization=0.20,
              max_model_len=2048, enable_prefix_caching=True, kv_transfer_config=kv)
    # 8 long shared prefixes; 96 requests cycling them under tight GPU KV -> evict to
    # CPU then reload on recurrence (store D2H + load H2D via cuMemcpyBatchAsync).
    import statistics
    prefixes = [f"Prefix {chr(65 + p)}: " + ("token " * 280) for p in range(8)]
    sp = SamplingParams(max_tokens=24, temperature=0.0)

    def batch(run):  # shared prefixes (offload/reload) + fresh tails (prefill) each run
        return [prefixes[i % 8] + f" run {run} unique tail {i}." for i in range(96)]

    iters = int(os.environ.get("EKV_ITERS", "4"))
    llm.generate(batch(-1), sp)                          # warmup (prime CUDA + caches)
    llm.generate(batch(-2), sp)                          # second warmup (steady state)
    rates = []
    for r in range(iters):                               # measured, warm
        t0 = time.monotonic()
        out = llm.generate(batch(r), sp)
        dt = time.monotonic() - t0
        rates.append(sum(len(o.outputs[0].token_ids) for o in out) / dt)
    mean = statistics.mean(rates)
    std = statistics.pstdev(rates)
    print(f"OFFLOAD_RUN tok/s mean={mean:.1f} std={std:.1f} ({100 * std / mean:.1f}%) "
          f"median={statistics.median(rates):.1f} n={iters}")


if __name__ == "__main__":
    main()
