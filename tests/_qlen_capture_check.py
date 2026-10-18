"""Validate the PASS-side capture of per-seq query length (cu_seqlens_q) on a live engine.

The eKV pass (EKV_PASS_AUTO + EKV_QLEN) snapshots the attention kernel's cu_seqlens_q array
into a published host buffer each invocation; zero_touch.read_pass_qlen computes per-seq
query_len = qsl[s+1]-qsl[s] from it (externally, via /proc). Here we read that buffer right
after each step and compare to the engine's OWN query_start_loc (ground truth) — so we confirm
the pass captures the exact compute signal (large at prefill = recompute, ~1 at decode =
cache-hit/read). Needs the triton-src pass env + GPU; see _qlen_capture_run.sh.
"""
import os
import sys

os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"
os.environ["EKV_PASS_AUTO"] = "1"
os.environ["EKV_QLEN"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from observe import zero_touch                       # noqa: E402

_cmp = {"prefill_ok": 0, "prefill_tot": 0, "decode_ok": 0, "decode_tot": 0, "ex": None}


from ekv_vllm_plugin import install_pass  # noqa: E402

install_pass()


def _install_oracle():
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner as GR
    orig = GR.execute_model

    def execute_model(runner, *a, **k):
        out = orig(runner, *a, **k)
        try:
            ib = runner.input_batch
            n = int(ib.num_reqs)
            cu = getattr(runner, "query_start_loc", None)
            arr = None
            for at in ("np", "cpu"):
                t = getattr(cu, at, None)
                if t is not None:
                    arr = t.tolist() if hasattr(t, "tolist") else list(t)
                    break
            if n <= 0 or arr is None or len(arr) < n + 1:
                return out
            truth = {s: int(arr[s + 1]) - int(arr[s]) for s in range(n)}      # engine ground truth
            passq = zero_touch.read_pass_qlen(os.getpid())                    # the pass capture (external read)
            for s in range(n):
                is_prefill = truth[s] > 1
                key = "prefill" if is_prefill else "decode"
                _cmp[key + "_tot"] += 1
                if passq.get(s) == truth[s]:
                    _cmp[key + "_ok"] += 1
                elif _cmp["ex"] is None and is_prefill:
                    _cmp["ex"] = (s, truth[s], passq.get(s))
        except Exception:  # noqa: BLE001
            pass
        return out
    GR.execute_model = execute_model


def main():
    _install_oracle()
    from vllm import LLM, SamplingParams
    prompts = [("tok " * (12 + i * 17)).strip() for i in range(6)]   # DISTINCT prefill lengths
    llm = LLM(model="facebook/opt-125m", enforce_eager=True, attention_backend="TRITON_ATTN",
              gpu_memory_utilization=0.30, max_model_len=2048, enable_prefix_caching=False,
              enable_chunked_prefill=False, max_num_seqs=8)
    llm.generate(prompts, SamplingParams(max_tokens=8, temperature=0.0, ignore_eos=True))

    pf, pt = _cmp["prefill_ok"], _cmp["prefill_tot"]
    df, dt = _cmp["decode_ok"], _cmp["decode_tot"]
    print(f"pass cu_seqlens_q capture vs engine query_start_loc:")
    print(f"  prefill seqs (query_len>1): {pf}/{pt} match   decode seqs (query_len==1): {df}/{dt} match")
    if _cmp["ex"]:
        print(f"  first prefill mismatch: slot={_cmp['ex'][0]} engine={_cmp['ex'][1]} pass={_cmp['ex'][2]}")
    if pt > 0 and pf == pt and dt > 0 and df == dt:
        print("PASS: the pass captures the kernel's cu_seqlens_q exactly — per-seq query length "
              "(the recompute-vs-cache-hit signal) recovered zero-touch via /proc, matching the "
              "engine's own query_start_loc on every prefill and decode seq.")
    else:
        print(f"CHECK: prefill {pf}/{pt}, decode {df}/{dt} — inspect (timing/overwrite or arg match).")


if __name__ == "__main__":
    main()
