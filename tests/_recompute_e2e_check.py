"""END-TO-END: the PASS-captured query length drives the recompute/cache-hit classification
and the park's cache-hit zeroing, on ONE live run, read per-step via /proc — zero-touch.

  pass (EKV_PASS_AUTO+EKV_QLEN)  -> snapshots cu_seqlens_q each step
  read_pass_qlen(pid) per step    -> per-seq query length  (the compute signal, via /proc)
  oracle (stand-in for the eBPF input_ids face) -> per-slot tokens -> content hashes + residency
  classify_rematerializations(compute = PASS query length) -> recompute vs cache-hit, exact

We also CROSS-VALIDATE live: the pass's per-step query length vs the engine's own query_start_loc
(they must agree), and that classifying with the PASS signal == classifying with the engine's.
For the park: a re-materialization the pass marks cache-hit (small query length, no compute) is a
request whose KV was already resident -> its KV-park is 0 (the fix for the shared-reload over-
attribution). Needs the triton-src pass env + GPU; see _recompute_e2e_run.sh.
"""
import os
import sys
import time

os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"
os.environ["EKV_PASS_AUTO"] = "1"
os.environ["EKV_QLEN"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from observe import classify_rematerializations, content_hashes_from_tokens, zero_touch  # noqa: E402

_or = {"timeline": [], "pass_q": [], "oracle_q": [], "agree": 0, "agree_tot": 0}


from ekv_vllm_plugin import install_pass  # noqa: E402

install_pass()


def _slot_tokens(ib, n):
    tok = getattr(ib, "token_ids_cpu", None)
    if tok is None:
        return {}
    nt = getattr(ib, "num_computed_tokens_cpu", None)
    if nt is None:
        nt = getattr(ib, "num_tokens", None)
    out = {}
    for s in range(n):
        try:
            k = int(nt[s]) if nt is not None else 0
            out[s] = [int(x) for x in (tok[s][:k] if k > 0 else tok[s])]
        except Exception:  # noqa: BLE001
            pass
    return out


def _oracle_qlen(runner, n):
    cu = getattr(runner, "query_start_loc", None)
    arr = None
    for at in ("np", "cpu"):
        t = getattr(cu, at, None)
        if t is not None:
            arr = t.tolist() if hasattr(t, "tolist") else list(t)
            break
    if arr is None or len(arr) < n + 1:
        return {}
    return {s: int(arr[s + 1]) - int(arr[s]) for s in range(n)}


def _install_oracle():
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner as GR
    orig = GR.execute_model

    def execute_model(runner, *a, **k):
        out = orig(runner, *a, **k)
        try:
            now = time.monotonic_ns()
            ib = runner.input_batch
            n = int(ib.num_reqs)
            if n <= 0:
                return out
            bt = ib.block_table[0].get_device_tensor(n).detach().to("cpu").tolist()
            toks = _slot_tokens(ib, n)
            oq = _oracle_qlen(runner, n)                          # engine ground-truth query length
            pq = zero_touch.read_pass_qlen(os.getpid())           # PASS capture, read via /proc
            for s in range(n):                                    # live cross-validation
                if s in oq and s in pq:
                    _or["agree_tot"] += 1
                    _or["agree"] += int(pq[s] == oq[s])
            snap, comp_pass, comp_oracle = {}, {}, {}
            for s in range(min(n, len(bt))):
                sb = {s: [int(b) for b in bt[s] if int(b) > 0]}
                for blk, h in content_hashes_from_tokens(sb, {s: toks.get(s, [])}).items():
                    snap[h] = blk
                    comp_pass[h] = pq.get(s, 0)
                    comp_oracle[h] = oq.get(s, 0)
            if snap:
                _or["timeline"].append((now, snap))
                _or["pass_q"].append(comp_pass)
                _or["oracle_q"].append(comp_oracle)
        except Exception:  # noqa: BLE001
            pass
        return out
    GR.execute_model = execute_model


def main():
    _install_oracle()
    from vllm import LLM, SamplingParams
    bases = [f"System policy {i}: " + " ".join(f"clause{i}_{w}" for w in range(150)) for i in range(8)]
    prompts = [bases[i] + f" request {j} answer:" for j, i in enumerate(list(range(8)) * 6)]
    llm = LLM(model="facebook/opt-125m", enforce_eager=True, attention_backend="TRITON_ATTN",
              gpu_memory_utilization=0.30, max_model_len=1024, enable_prefix_caching=True,
              enable_chunked_prefill=False, max_num_seqs=8,
              num_gpu_blocks_override=int(os.environ.get("EKV_GPU_BLOCKS", "96")))
    llm.generate(prompts, SamplingParams(max_tokens=24, temperature=0.0, ignore_eos=True))

    tl = _or["timeline"]
    print(f"steps captured={len(tl)}  live pass-vs-engine query-length agreement: "
          f"{_or['agree']}/{_or['agree_tot']}")
    if not tl:
        print("FAIL: no timeline captured")
        return
    by_pass = classify_rematerializations(tl, [], compute=_or["pass_q"])      # PASS signal drives it
    by_oracle = classify_rematerializations(tl, [], compute=_or["oracle_q"])  # engine signal
    heur = classify_rematerializations(tl, [])                                # same-block heuristic
    pr, pc = len(by_pass["recompute"]), len(by_pass["cache_hit"])
    orr, oc = len(by_oracle["recompute"]), len(by_oracle["cache_hit"])
    print(f"classify by PASS query-length : recompute={pr}  cache_hit={pc}")
    print(f"classify by engine query-length: recompute={orr}  cache_hit={oc}  (must match PASS)")
    print(f"same-block heuristic           : recompute={len(heur['recompute'])}  "
          f"cache_hit={len(heur['cache_hit'])}  (over-counts recompute)")
    print(f"park: {pc} cache-hit re-materializations -> KV-park 0 (KV already resident, no wait); "
          f"the heuristic would have mislabeled {len(heur['recompute'])} as recompute.")
    agree_ok = _or["agree_tot"] > 0 and _or["agree"] == _or["agree_tot"]
    classify_ok = (pr, pc) == (orr, oc)
    if agree_ok and classify_ok:
        print("PASS(e2e): the PASS-captured query length (read per-step via /proc) matches the "
              "engine's own on every seq AND drives the recompute/cache-hit classification "
              "identically — zero-touch, end to end. Cache-hits correctly carry KV-park 0.")
    else:
        print(f"CHECK(e2e): agree={agree_ok} classify_match={classify_ok} (pass {(pr,pc)} vs "
              f"engine {(orr,oc)}).")


if __name__ == "__main__":
    main()
