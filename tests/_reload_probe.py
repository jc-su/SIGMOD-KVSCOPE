"""vLLM offline with CPU KV offload (SimpleCPUOffloadConnector) and a small GPU pool, so blocks evict
to host memory and reload on re-access. Cycles long shared prefixes that overflow the pool to force
eviction then reload. For the reload-aware use case: eKV attributes the per-request reload bytes from
the captured host->device copies. Announces PID and pauses for the eBPF attach."""
import os
import time

os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"   # required by the offload connector

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

# Ground-truth hook: record per-step {req_id -> (prompt tokens, physical block ids)} for validation.
import json as _json
from vllm.v1.core.sched.scheduler import Scheduler
_orc = []
_orig_sched = Scheduler.schedule


def _patched_sched(self):
    out = _orig_sched(self)
    snap = {}
    for req in list(self.running):
        try:
            b = self.kv_cache_manager.get_block_ids(req.request_id)
            snap[req.request_id] = {"p": list(req.prompt_token_ids or []), "b": list(b[0]) if b else []}
        except Exception:
            pass
    _orc.append(snap)
    return out


Scheduler.schedule = _patched_sched

MODEL = os.environ.get("EKV_MODEL", "facebook/opt-125m")
BLOCKS = int(os.environ.get("EKV_GPU_BLOCKS", "160"))
print(f"EKV_PID {os.getpid()}", flush=True)
time.sleep(float(os.environ.get("EKV_ALLOC_WAIT", "6")))   # window to attach kv_alloc BEFORE the KV-pool cudaMalloc
llm = LLM(model=MODEL, gpu_memory_utilization=float(os.environ.get("EKV_GPU_UTIL", "0.30")),
          max_model_len=1024, num_gpu_blocks_override=BLOCKS,
          kv_transfer_config=KVTransferConfig(
              kv_connector="SimpleCPUOffloadConnector", kv_role="kv_both",
              kv_connector_extra_config={"cpu_bytes_to_use": 2 * 1024 ** 3}))
print("EKV_READY", flush=True)
time.sleep(7)

chunk = "the key value cache stores attention state for every token so decoding need not recompute "
prefixes = [f"document {i}: " + chunk * 45 for i in range(8)]   # ~600-token prefixes, overflow the pool
ORDER = os.environ.get("EKV_ORDER", "interleaved")             # interleaved (baseline) | grouped (eKV-guided)
if ORDER == "grouped":                                         # group each prefix's requests together
    items = [(i, r) for i in range(8) for r in range(4)]
else:                                                          # round-major: re-access all prefixes each round
    items = [(i, r) for r in range(4) for i in range(8)]
prompts = [prefixes[i] + f" question {r} {i}" for (i, r) in items]
print("EKV_GEN_START", flush=True)
t0 = time.monotonic()
out = llm.generate(prompts, SamplingParams(max_tokens=8, temperature=0.0))
wall = time.monotonic() - t0
print(f"EKV_GEN_DONE n={len(out)} wall={wall:.2f}", flush=True)
tok = llm.get_tokenizer()
_json.dump({"oracle": _orc, "order": ORDER, "wall": wall,
            "prompts": [{"idx": j, "prefix": items[j][0], "round": items[j][1],
                         "tok": list(tok(p, add_special_tokens=True)["input_ids"])}
                        for j, p in enumerate(prompts)]},
           open(os.environ.get("EKV_RELOAD_ORACLE", "/tmp/reload_oracle.json"), "w"))
print("EKV_ORACLE_DONE", flush=True)
time.sleep(1)
