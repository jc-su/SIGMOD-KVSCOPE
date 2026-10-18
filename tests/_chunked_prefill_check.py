"""Validate the seq_idx-indexed index_lookup capture under CHUNKED PREFILL (vLLM's production default) --
the exact regime the index_lookup fix targets. Long prompts are split into chunks across steps and batched
WITH decodes, so a sequence spans multiple query blocks and program_id (query-block index) != seq_idx.
At every such multi-query-block step we read the pass's slot->blocks EXTERNALLY (/proc) and compare it
to the engine's OWN block_table per sequence -- they must match. enable_chunked_prefill=True with a
small max_num_batched_tokens so prompts actually chunk. Zero-touch (the pass + the /proc read; the
oracle is validation-only). See _chunked_prefill_run.sh.
"""
import os
import sys

os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"
os.environ["EKV_PASS_AUTO"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from observe import zero_touch                                    # noqa: E402


from ekv_vllm_plugin import install_pass  # noqa: E402

install_pass()
_or = {"match": 0, "total": 0, "chunked_steps": 0, "max_qlen": 0, "mism": []}


def _install_oracle():
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner as GR
    orig = GR.execute_model

    def execute_model(runner, *a, **k):
        out = orig(runner, *a, **k)
        try:
            ib = runner.input_batch
            n = int(ib.num_reqs)
            cu = getattr(runner, "query_start_loc", None)
            lens = None
            for at in ("np", "cpu"):
                t = getattr(cu, at, None)
                if t is not None:
                    lens = [int(t[i + 1]) - int(t[i]) for i in range(n)]
                    break
            if not lens or n < 1:
                return out
            _or["max_qlen"] = max(_or["max_qlen"], max(lens))
            # a step that EXERCISES the fix: a sequence spanning multiple query blocks (qlen > the
            # query tile) -> program_id != seq_idx, so a program_id-indexed dump would be wrong here.
            if max(lens) >= 64:
                bt = ib.block_table[0].get_device_tensor(n).detach().to("cpu").tolist()
                passed = zero_touch.read_pass_slots(os.getpid())   # external /proc read of the pass buffer
                _or["chunked_steps"] += 1
                for s in range(min(n, len(bt))):
                    truth = sorted(int(b) for b in bt[s] if int(b) > 0)
                    if not truth:
                        continue                                   # no blocks allocated yet for this slot
                    got = sorted(int(b) for b in passed.get(s, []) if int(b) > 0)
                    _or["total"] += 1
                    if got == truth:
                        _or["match"] += 1
                    elif len(_or["mism"]) < 6:
                        _or["mism"].append((s, max(lens), truth, got))
        except Exception:  # noqa: BLE001
            pass
        return out
    GR.execute_model = execute_model


def main():
    _install_oracle()
    from vllm import LLM, SamplingParams
    # Long prompts chunk across steps at a 256-token budget; short ones decode alongside -> MIXED
    # chunked batches where program_id != seq_idx for the prefilling sequences.
    word_counts = [400, 120, 550, 80, 300, 60, 480, 100]
    prompts = [f"doc{i} " + ("token " * w).strip() for i, w in enumerate(word_counts)]
    sp = SamplingParams(max_tokens=16, temperature=0.0, ignore_eos=True)
    eager = os.environ.get("EKV_EAGER", "1") == "1"   # EKV_EAGER=0 -> CUDA graphs ON (prod default)
    model = os.environ.get("EKV_MODEL", "facebook/opt-125m")     # EKV_MODEL=... for a 2nd/larger model
    gpu_mem = float(os.environ.get("EKV_GPU_MEM", "0.30"))
    print(f"model={model} enforce_eager={eager} (CUDA graphs {'OFF' if eager else 'ON'}) gpu_mem={gpu_mem}")
    kw = dict(model=model, enforce_eager=eager, attention_backend="TRITON_ATTN",
              gpu_memory_utilization=gpu_mem, max_model_len=2048,
              enable_chunked_prefill=True, max_num_batched_tokens=256, max_num_seqs=16)
    if os.environ.get("EKV_GPU_BLOCKS"):                          # else let vLLM size the KV pool
        kw["num_gpu_blocks_override"] = int(os.environ["EKV_GPU_BLOCKS"])
    llm = LLM(**kw)
    llm.generate(prompts, sp)

    m, tot, cs = _or["match"], _or["total"], _or["chunked_steps"]
    print(f"max query-length seen = {_or['max_qlen']} (chunking exercised if >> the query tile)")
    print(f"multi-query-block steps observed (program_id != seq_idx): {cs}")
    print(f"index_lookup slot->blocks == engine block_table (per active seq, at those steps): {m}/{tot}")
    for s, ql, t, p in _or["mism"]:
        print(f"  MISMATCH seq {s} (step max_qlen={ql}): engine={t}  pass={p}")
    if cs >= 1 and tot >= 1 and m == tot:
        print("PASS(chunked-prefill): the seq_idx-indexed index_lookup capture matches the engine's "
              "block_table per sequence under CHUNKED PREFILL (mixed prefill-chunk + decode batches, "
              "program_id != seq_idx) -- the production-default regime, zero-touch.")
    else:
        print(f"CHECK(chunked-prefill): match={m}/{tot} chunked_steps={cs} max_qlen={_or['max_qlen']}")


if __name__ == "__main__":
    main()
