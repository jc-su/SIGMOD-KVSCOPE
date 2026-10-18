"""Timed generation for the pass+timer overhead A/B (tests/_cycles_overhead_run.sh sets the env:
pass OFF = stock Triton, pass ON = plugin + EKV_TIMER + capture). Prints one EKV_RESULT line.
The workload is decode-heavy (the production regime the timer targets); warmup covers compile,
CUDA-graph capture, and the profile run so the timed section is steady-state serving only."""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

from vllm import LLM, SamplingParams

LABEL = os.environ.get("EKV_LABEL", "?")
MODEL = os.environ.get("EKV_MODEL", "facebook/opt-125m")
llm = LLM(model=MODEL, gpu_memory_utilization=float(os.environ.get("EKV_GPU_UTIL", "0.35")),
          max_model_len=2048, enable_prefix_caching=False, attention_backend="TRITON_ATTN")

prompts = [("word " * (60 * (r % 8 + 1))).strip() for r in range(32)]
llm.generate(prompts[:8], SamplingParams(max_tokens=4, temperature=0.0))     # warmup

t0 = time.perf_counter()
out = llm.generate(prompts, SamplingParams(max_tokens=128, temperature=0.0))
dt = time.perf_counter() - t0
toks = sum(len(o.outputs[0].token_ids) for o in out)
print(f"EKV_RESULT label={LABEL} wall_s={dt:.3f} tokens={toks} toks_per_s={toks / dt:.1f}",
      flush=True)
