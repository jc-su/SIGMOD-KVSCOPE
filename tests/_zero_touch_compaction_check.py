"""Compaction-robust request identity. When requests finish at different steps, vLLM
COMPACTS the batch (a later request is shifted into the freed row), so the prefill-order
seq->request tie goes stale. A request's FIRST physical block is allocated at prefill and
never moves, so it identifies the request across compaction.

This validates the tracking LOGIC (observe.zero_touch.anchor_first_blocks /
slot_to_request_via_anchor) on REAL per-step block tables from a live, compacting vLLM:
  * the per-step block_table == what the pass captures (validated exact elsewhere);
  * step-0 seq->request == the eBPF input_ids tie (validated 8/8 elsewhere);
here we feed those real values to the first-block tracker and check it stays correct at
EVERY step, vs the naive one-shot prefill map.

  CUDA_VISIBLE_DEVICES=1 python tests/_zero_touch_compaction_check.py
"""
import os
import sys

os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from observe import zero_touch  # noqa: E402

_steps = []   # per pure-decode step: (slot_blocks {slot:[blocks]}, slot_to_request {slot:rid})


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
            for attr in ("np", "cpu"):
                t = getattr(cu, attr, None)
                if t is not None:
                    lens = [int(t[i + 1]) - int(t[i]) for i in range(n)]
                    break
            if n > 0 and lens is not None and all(x == 1 for x in lens):   # pure decode
                bt = ib.block_table[0].get_device_tensor(n).detach().to("cpu")
                rids = [str(r) for r in list(ib.req_ids)[:n]]
                sb = {s: [int(x) for x in bt[s].tolist() if int(x) != 0] for s in range(n)}
                s2r = {s: rids[s] for s in range(n)}
                _steps.append((sb, s2r))
        except Exception:  # noqa: BLE001
            pass
        return out
    GR.execute_model = execute_model


def main():
    _install_oracle()
    from vllm import LLM, SamplingParams
    llm = LLM(model="facebook/opt-125m", enforce_eager=True, attention_backend="TRITON_ATTN",
              gpu_memory_utilization=0.30, max_model_len=2048, enable_prefix_caching=True,
              enable_chunked_prefill=False, max_num_seqs=64, num_gpu_blocks_override=4096)
    prompts = [f"Distinct prompt number {i} " + ("alpha beta gamma delta " * (4 + i))
               for i in range(8)]
    # STAGGERED lengths -> requests finish at different steps -> the batch compacts.
    sps = [SamplingParams(max_tokens=6 + 5 * i, temperature=0.0, ignore_eos=True)
           for i in range(8)]
    llm.generate(prompts, sps)

    if len(_steps) < 3:
        print(f"FAIL: too few decode steps ({len(_steps)})")
        return
    # step-0 tie (== eBPF input_ids tie) -> anchor first_block -> request.
    sb0, s2r0 = _steps[0]
    anchor = zero_touch.anchor_first_blocks(sb0, s2r0)
    print(f"decode steps={len(_steps)}  step0 reqs={len(s2r0)}  anchored first-blocks={len(anchor)}")

    compacted_steps = robust_ok = naive_ok = checked = 0
    for sb, truth in _steps:
        # did rows move vs step 0? (a slot now holds a different request)
        moved = any(s in truth and s in s2r0 and truth[s] != s2r0[s] for s in truth)
        compacted_steps += 1 if moved else 0
        robust = zero_touch.slot_to_request_via_anchor(sb, anchor)
        for s, rid in truth.items():
            checked += 1
            if robust.get(s) == rid:
                robust_ok += 1
            if s2r0.get(s) == rid:           # naive one-shot prefill map
                naive_ok += 1
    print(f"steps with compaction (rows moved): {compacted_steps}/{len(_steps)}")
    print(f"slot->request correct  ROBUST (first-block): {robust_ok}/{checked}")
    print(f"slot->request correct  NAIVE  (prefill map):  {naive_ok}/{checked}")

    ok = robust_ok == checked and compacted_steps > 0 and naive_ok < checked
    print("PASS: first-block tracking keeps request identity correct across batch "
          "compaction (100%), where the naive prefill-order map goes stale — the "
          "compaction caveat is resolved." if ok else
          f"CHECK: robust={robust_ok}/{checked} naive={naive_ok}/{checked} "
          f"compacted={compacted_steps} (need robust=all, naive<all, compaction>0)")


if __name__ == "__main__":
    main()
