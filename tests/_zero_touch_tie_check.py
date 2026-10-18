"""ZERO-TOUCH request<->block, FULL tie — eBPF + pass only, no engine hook in the data path:
  * seq -> REQUEST : eBPF EKV_META captures the engine's input_ids H2D copy; the prompt
    token sequences (what http_label+net_label recover from the network) are located in
    it, their offsets giving the seq order -> seq -> network request.
  * seq -> BLOCK   : the eKV pass (auto-installed) captures block_table GPU-side; read
    externally via /proc/mem.
  * JOIN by seq    -> request -> physical blocks.

Validated vs an in-process oracle (engine req_ids + block_table) — TEST SCAFFOLDING only.
The prompts here stand in for http_label's output (proven separately on a served engine);
everything in the data path (input_ids via eBPF, blocks via pass+external-read) is zero-touch.

  bash tests/_zero_touch_tie_run.sh
"""
import json
import os
import sys

os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"
os.environ["EKV_PASS_AUTO"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from observe import net_label, zero_touch  # noqa: E402

PROMPTS = [
    "Zebra quantum waterfall marble engine drifting",
    "Maple thunder violin orchestra cobalt morning",
    "Falcon glacier ribbon tunnel velvet harbor",
    "Cinnamon nebula compass lantern prairie echo",
    "Obsidian meadow trumpet circuit walnut beacon",
    "Saffron comet anchor pixel meadowlark tundra",
    "Crimson abacus willow magnet thunderbolt cove",
    "Indigo pelican domino quartz lantern cascade",
]


from ekv_vllm_plugin import install_pass  # noqa: E402

install_pass()
_oracle = {"n": 0, "bt": None, "rids": None}


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
                _oracle.update(n=n, bt=bt,
                               rids=[str(r) for r in list(ib.req_ids)[:n] if r is not None])
        except Exception:  # noqa: BLE001
            pass
        return out
    GR.execute_model = execute_model


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("facebook/opt-125m")
    # net_reqs: exactly what http_label + net_label would hand us (prompt -> tokens).
    net_reqs = [net_label.NetRequest(fd=0, t_ns=i, method="POST", path="/v1/completions",
                                     prompt=p, arrival_idx=i,
                                     token_ids=list(tok(p)["input_ids"]))
                for i, p in enumerate(PROMPTS)]

    _install_oracle()
    from vllm import LLM, SamplingParams
    llm = LLM(model="facebook/opt-125m", enforce_eager=True, attention_backend="TRITON_ATTN",
              gpu_memory_utilization=0.30, max_model_len=2048, enable_prefix_caching=True,
              enable_chunked_prefill=False,   # single in-order prefill -> seq order == decode
              max_num_seqs=int(os.environ.get("EKV_MAX_ROWS", "128")),
              num_gpu_blocks_override=int(os.environ.get("EKV_GPU_BLOCKS", "2048")))
    # ignore_eos: all requests run the full length -> none finish early -> no batch
    # compaction -> the decode seq order stays equal to the prefill (input_ids) order.
    llm.generate(PROMPTS, SamplingParams(max_tokens=24, temperature=0.0, ignore_eos=True))

    pid = os.getpid()
    # --- seq -> REQUEST from the eBPF input_ids copy (zero touch) ---
    meta = []
    for ln in open("/tmp/ekv_meta.jsonl"):
        try:
            r = json.loads(ln)
            if r.get("type") == "meta":
                meta.append(r)
        except json.JSONDecodeError:
            pass
    seq2req = net_label.reconstruct_seq_to_request(meta, net_reqs)
    # --- seq -> BLOCK from the pass (zero touch, external readout) ---
    seq2block = zero_touch.read_pass_slots(pid)
    print(f"meta records={len(meta)}  seq->request recovered={len(seq2req)}  "
          f"seq->block slots={len(seq2block)}")

    bt, n, rids = _oracle["bt"], _oracle["n"], _oracle["rids"]
    if bt is None or not seq2req:
        print(f"FAIL: oracle empty={bt is None} or no tie (seq2req={len(seq2req)})")
        return

    # (A) the eBPF tie maps each seq to the right request: vLLM assigns req_id "i-..." in
    # submission order, so seq s's request index == seq2req[s] (the matched arrival_idx).
    tie_ok = sum(1 for s in range(n)
                 if s in seq2req and rids[s].split("-")[0].isdigit()
                 and seq2req[s] == int(rids[s].split("-")[0]))
    print(f"seq->request (eBPF input_ids) correct on {tie_ok}/{n} seqs")

    # (B) full join request -> blocks, vs the engine's own block_table per request.
    req2block = zero_touch.attribute(seq2block, seq2req)
    ncap = zero_touch.find_published(pid)[0].ncap
    join_ok = 0
    for s in range(n):
        aidx = seq2req.get(s)
        truth = [int(x) for x in bt[s][:ncap].tolist()]
        if aidx is not None and aidx in req2block:
            # req2block[aidx] is the deduped/sorted set; compare as sets vs truth row
            if set(req2block[aidx]) == set(x for x in truth):
                join_ok += 1
    print(f"request->block (joined) == engine block_table on {join_ok}/{n} requests")
    ex = next(((s, seq2req[s], seq2block.get(s, [])) for s in range(n) if s in seq2req), None)
    if ex:
        print(f"example: seq {ex[0]} -> request#{ex[1]} -> blocks "
              f"{[b for b in ex[2] if b != 0][:8]}")

    # (C) per-request COMPUTE: the pass timer (clock64) read externally, attributed via
    # the SAME seq->request tie -> request -> attention cycles. (EKV_TIMER=1)
    cycles = zero_touch.read_pass_cycles(pid)
    timed = 0
    if cycles:
        per_req = {}
        for s, c in cycles.items():
            aidx = seq2req.get(s)
            if aidx is not None:
                per_req[aidx] = per_req.get(aidx, 0) + c
        timed = len(per_req)
        top = sorted(per_req.items(), key=lambda kv: -kv[1])[:3]
        print(f"per-request compute (clock64) for {timed}/{n} requests; "
              f"top cycles: {[(f'req#{r}', c) for r, c in top]}")

    if tie_ok == n and join_ok == n and timed == n:
        print("PASS: zero-touch per-request plane — eBPF input_ids gave seq->request, the "
              "pass gave seq->block AND per-request clock64 compute (external /proc read), "
              "joined to request->block matching the engine exactly. No engine hook.")
    else:
        print(f"CHECK: tie={tie_ok}/{n} join={join_ok}/{n} timed={timed}/{n} (seq2req={seq2req})")


if __name__ == "__main__":
    main()
