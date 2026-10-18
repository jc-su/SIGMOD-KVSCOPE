"""vLLM offline workload for the eBPF-only compute-model fit. Identifiability requires Q=Σqlen and
W=Σ(attention area) to DECORRELATE; a single decode stream keeps them collinear because both scale
with the batch size N (corr(Q,W)->1, a<0, the fit degenerates). So we issue several phases with
different (batch, context) profiles -- the SAME batch at very different context, and high-batch/short
vs low-batch/long -- which spans the (Q,W) plane the way real mixed traffic does. Within each phase
varied decode budgets stagger the finish so N also sweeps to 1 (model-free anchors). Announces its PID
and pauses for the eBPF attach. GPU 1 via the runner."""
import os
import time

os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
from vllm import LLM, SamplingParams

MODEL = os.environ.get("EKV_MODEL", "facebook/opt-125m")   # override for a compute-dominated model
MAXLEN = int(os.environ.get("EKV_MAXLEN", "2048"))
UTIL = float(os.environ.get("EKV_GPU_UTIL", "0.35"))
print(f"EKV_PID {os.getpid()}", flush=True)
print(f"EKV_MODEL {MODEL}", flush=True)
llm = LLM(model=MODEL, gpu_memory_utilization=UTIL, max_model_len=MAXLEN,
          enable_prefix_caching=False)          # independent blocks -> clean per-request ctxlen
print("EKV_READY", flush=True)
time.sleep(7)

chunk = ("the serving system handles requests of very different lengths and decode budgets and we "
         "want to attribute per request attention cost from the boundary metadata alone ")
CW = len(chunk.split())
def make(n, words):                                   # n independent prompts of ~words context
    reps = max(1, words // CW)
    return [chunk * reps + f" tag {p} {words}" for p in range(n)]
# (batch, context-words) phases chosen so Q and W decorrelate: the same batch (8, 9) appears at very
# different context, and high-batch/short (10,90) contrasts with low-batch/long (2,1150), spanning
# the (Q,W) plane instead of riding one line through the origin.
phases = [(10, 90), (8, 1150), (2, 1150), (6, 350), (4, 800), (3, 1000), (9, 150)]
print("EKV_GEN_START", flush=True)
tot = 0
for (n, words) in phases:
    ps = make(n, words)
    sps = [SamplingParams(max_tokens=8 + 16 * j, temperature=0.0) for j in range(n)]  # staggered finish
    o = llm.generate(ps, sps)
    tot += sum(len(x.outputs[0].token_ids) for x in o)
print(f"EKV_GEN_DONE tokens={tot}", flush=True)
time.sleep(1)
