"""Minimal real-vLLM run (offline, cached model) so the eBPF collector can be
attached to a real serving process. eager mode so copies aren't hidden inside
CUDA graphs (maximal visibility for the go/no-go)."""
import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

from vllm import LLM, SamplingParams  # noqa: E402

MODEL = os.environ.get("EKV_MODEL", "meta-llama/Llama-3.2-1B")
llm = LLM(model=MODEL, gpu_memory_utilization=0.3, max_model_len=2048,
          enforce_eager=True)
outs = llm.generate(["The capital of France is"] * 8,
                    SamplingParams(max_tokens=64, temperature=0.0))
print(f"vLLM generated {len(outs)} outputs on {MODEL}")
