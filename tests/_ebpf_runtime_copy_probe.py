"""A real vLLM engine, in process, that makes CUDA runtime copies and KV offload copies.

The engine builds first, so the process has mapped its CUDA runtimes (torch 2.11+cu130 calls
libcudart.so.13; a dependency maps libcudart.so.12) before ``tests/_ebpf_runtime_copy_check.py``
attaches the kv_copy collector. Then it waits for the start file and serves two phases:

1. ``EKV_PROMPTS`` distinct prompts of about ``EKV_CTX`` tokens. They need more blocks than the GPU
   cache has (``EKV_GPU_BLOCKS``), so vLLM's native CPU offload stores evicted blocks in host memory.
2. The first ``EKV_REPEAT`` prompts again. Their prefixes are no longer on the GPU, so the offload
   connector loads them back from host memory.

Every engine step copies its inputs to the GPU with torch, which calls cudaMemcpyAsync in the CUDA
runtime. The offload copies use the driver's cuMemcpyBatchAsync.

Env: EKV_START (the start file), EKV_MODEL, EKV_CTX, EKV_PROMPTS, EKV_REPEAT, EKV_GPU_BLOCKS.
"""

import os
import time
from pathlib import Path

os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"  # in-process engine: the collector attaches to this pid

MODEL = os.environ.get("EKV_MODEL", "meta-llama/Llama-3.2-1B-Instruct")
CTX = int(os.environ.get("EKV_CTX", "1024"))
PROMPTS = int(os.environ.get("EKV_PROMPTS", "12"))
REPEAT = int(os.environ.get("EKV_REPEAT", "4"))
GPU_BLOCKS = int(os.environ.get("EKV_GPU_BLOCKS", "256"))


def main() -> None:
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    llm = LLM(
        model=MODEL,
        gpu_memory_utilization=0.25,
        max_model_len=CTX + 64,
        num_gpu_blocks_override=GPU_BLOCKS,
        enable_prefix_caching=True,
        kv_offloading_size=1.0,
        max_num_seqs=4,
    )
    params = SamplingParams(max_tokens=1, temperature=0.0)
    llm.generate(
        [TokensPrompt(prompt_token_ids=[1] * 64)], params
    )  # compile and warm up before the collector
    prompts = [
        TokensPrompt(prompt_token_ids=[1000 + p] + [(p * 7919 + i) % 30000 + 100 for i in range(CTX - 1)])
        for p in range(PROMPTS)
    ]

    print(f"EKV_PID {os.getpid()}", flush=True)
    start = Path(os.environ["EKV_START"])
    while not start.exists():
        time.sleep(0.05)
    for prompt in prompts:  # one at a time: each later prompt evicts earlier blocks to host memory
        llm.generate([prompt], params, use_tqdm=False)
    llm.generate(prompts[:REPEAT], params, use_tqdm=False)
    print("EKV_DONE", flush=True)
    time.sleep(1.0)


if __name__ == "__main__":
    main()
