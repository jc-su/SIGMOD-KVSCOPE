"""SGLang SERVING e2e (gap 2): a real sglang.Engine (scheduler + radix allocator + continuous
batching) serving a BATCH of varied-length requests. The sitecustomize (set by the runner, running
in the spawned scheduler) accumulates, per request, eKV's pass-captured KV slots AND SGLang's own
kv_indices ground truth over the whole run, so the check can validate the per-request reconstruction
and compute per-request KV accounting. Distinct prefixes -> each request's first slot is a unique
content-addressed anchor. MASK REQUIRED: SGLang spawns its scheduler; a without mask enforcement main re-spawns."""
import os
import sys

import sglang


def main():
    model = os.environ.get("EKV_SGLANG_MODEL", "facebook/opt-125m")
    try:
        llm = sglang.Engine(model_path=model, attention_backend="triton",
                            mem_fraction_static=0.45, disable_cuda_graph=True, log_level="warning")
    except Exception as e:  # noqa: BLE001
        print("ENGINE_INIT_FAIL:", repr(e)[:700])
        sys.exit(2)
    try:
        # varied-length, distinct-prefix requests -> continuous batching, unique per-request anchors
        prompts = [f"Request {i}: describe " + "several distinct topics and ideas " * (4 * (i + 1))
                   for i in range(6)]
        out = llm.generate(prompts, {"max_new_tokens": 48, "temperature": 0.0})
        print("GEN_OK", len(out) if isinstance(out, list) else 1, flush=True)
    finally:
        llm.shutdown()


if __name__ == "__main__":
    main()
