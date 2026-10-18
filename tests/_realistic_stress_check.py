"""#0 REAL-WORLD STRESS TEST of the request<->request-identity tie under the production
settings the 8/8 validation dodged: chunked prefill ON, prefix caching ON, natural
finishing (no ignore_eos -> compaction), duplicates + shared prefixes + long prompts.

It captures, per step (in-process oracle = stand-in for the eBPF input_ids capture, which
is byte-exact-validated separately), the engine's input_ids + seq boundaries + req_ids
(ground truth), then measures:
  (current) does each request's FULL prompt appear in a SINGLE step's input_ids? — the
            assumption reconstruct_seq_to_request relies on; and that fn's tie accuracy.
  (robust)  if we ACCUMULATE per-slot tokens across steps/chunks, do we recover identity?

No eBPF/sudo/pass needed — default attention backend; this isolates the identity logic.
  CUDA_VISIBLE_DEVICES=1 python tests/_realistic_stress_check.py
"""
import os
import sys

os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_steps = []   # per step: {"rids":[...], "cu":[...], "toks":[...]}


def _install_oracle():
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner as GR
    orig = GR.execute_model

    def execute_model(runner, *a, **k):
        out = orig(runner, *a, **k)
        try:
            ib = runner.input_batch
            n = int(ib.num_reqs)
            if n <= 0:
                return out
            rids = [str(r) for r in list(ib.req_ids)[:n] if r is not None]
            cu = getattr(runner, "query_start_loc", None)
            cua = None
            for attr in ("np", "cpu"):
                t = getattr(cu, attr, None)
                if t is not None:
                    cua = t.tolist() if hasattr(t, "tolist") else list(t)
                    break
            if cua is None:
                return out
            ntok = int(cua[n])
            ii = getattr(runner, "input_ids", None)
            if hasattr(ii, "gpu"):          # CpuGpuBuffer: host side is .cpu
                toks = ii.cpu[:ntok].tolist()
            else:                            # plain tensor
                toks = ii[:ntok].detach().to("cpu").tolist()
            _steps.append({"rids": rids[:n], "cu": [int(x) for x in cua[:n + 1]], "toks": toks})
        except Exception as e:  # noqa: BLE001
            if not _steps:
                print(f"[oracle] capture error: {type(e).__name__}: {e}", file=sys.stderr)
        return out
    GR.execute_model = execute_model


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("facebook/opt-125m")
    # realistic prompt mix: long (chunked), shared-prefix (prefix cache), duplicates, short.
    long_pre = "Detailed corporate policy document. " + " ".join(f"section{i} clause text" for i in range(120))
    prompts = []
    for i in range(6):
        prompts.append(long_pre + f" Question {i}: summarize section {i}.")     # shared long prefix -> cache + chunk
    for i in range(6):
        prompts.append(f"Short distinct query number {i} about topic {i*7}.")    # short distinct
    prompts += prompts[:4]                                                        # duplicates
    prompts.append(long_pre + " Question 0: summarize section 0.")               # exact dup of a long one
    import random
    random.seed(0)
    net = [{"arrival": i, "tokens": list(tok(p)["input_ids"])} for i, p in enumerate(prompts)]

    _install_oracle()
    from vllm import LLM, SamplingParams
    llm = LLM(model="facebook/opt-125m", enforce_eager=True,
              enable_chunked_prefill=True, max_num_batched_tokens=256,   # force chunking
              enable_prefix_caching=True, max_num_seqs=16,
              gpu_memory_utilization=0.30, max_model_len=2048)
    # natural finishing (no ignore_eos) -> requests end at different steps -> compaction
    sp = [SamplingParams(max_tokens=8 + (i * 5) % 40, temperature=0.0) for i in range(len(prompts))]
    llm.generate(prompts, sp)

    print(f"captured {len(_steps)} steps; {len(prompts)} requests "
          f"({len(set(map(tuple, (n['tokens'] for n in net))))} distinct token-seqs)")

    # ---- (current) does each request's FULL prompt appear in ONE step's input_ids? ----
    def contiguous_in(hay, needle):
        if not needle or len(needle) > len(hay):
            return False
        for s in range(len(hay) - len(needle) + 1):
            if hay[s:s + len(needle)] == needle:
                return True
        return False

    step_tok = [st["toks"] for st in _steps]
    full_in_one = 0
    for nr in net:
        if any(contiguous_in(t, nr["tokens"]) for t in step_tok):
            full_in_one += 1
    print(f"(current assumption) full prompt present in a single step's input_ids: "
          f"{full_in_one}/{len(net)} requests")

    # ---- (robust) accumulate per-slot tokens across steps, per prefill episode ----
    # a slot's sequence grows while the same request occupies it; a new request in the slot
    # (its accumulated len resets when req_ids[slot] changes) starts a fresh episode.
    slot_seq = {}        # slot -> {"req": rid, "toks": [...]}
    episodes = []        # (rid, accumulated_tokens)
    for st in _steps:
        rids, cu, toks = st["rids"], st["cu"], st["toks"]
        for s in range(len(rids)):
            seg = toks[cu[s]:cu[s + 1]] if s + 1 < len(cu) else []
            cur = slot_seq.get(s)
            if cur is None or cur["req"] != rids[s]:
                if cur is not None:
                    episodes.append((cur["req"], cur["toks"]))
                cur = {"req": rids[s], "toks": []}
                slot_seq[s] = cur
            cur["toks"].extend(seg)
    for cur in slot_seq.values():
        episodes.append((cur["req"], cur["toks"]))
    acc = {}
    for rid, toks in episodes:
        acc.setdefault(rid, toks)

    # a request's captured prefill tokens = the UNCACHED suffix of its prompt (prefix
    # caching hides the cached prefix), chunk-accumulated, then decode tokens. So some
    # SUFFIX of the prompt P[c:] should be a PREFIX of the accumulated A. min_len checks
    # against a too-short (non-distinguishing) match.
    def best_suffix_match(A, P, min_len=4):
        for c in range(len(P)):
            suf = P[c:]
            if len(suf) >= min_len and A[:len(suf)] == suf:
                return len(suf)               # longest first (c=0 => full prompt)
        return 0
    robust_ok = ambiguous = tooshort = 0
    for rid, A in acc.items():
        matched = {tuple(nr["tokens"]) for nr in net if best_suffix_match(A, nr["tokens"]) > 0}
        if len(matched) == 1:
            robust_ok += 1
        elif len(matched) > 1:
            ambiguous += 1
        else:
            tooshort += 1                     # nothing distinguishing captured (fully cached / decode-only)
    print(f"(robust suffix-match) uniquely identified {robust_ok}/{len(acc)} episodes; "
          f"{ambiguous} ambiguous, {tooshort} unidentifiable (too few prompt tokens captured)")

    print("VERDICT:")
    print(f"  - CURRENT (full-prompt-in-one-copy): {full_in_one}/{len(net)}")
    print(f"  - ROBUST (suffix-of-prompt = prefix-of-accumulated): {robust_ok}/{len(acc)} unique")
    print("  - residual = requests whose UNCACHED part is too short / fully cached: input_ids "
          "alone can't identify them (a fundamental prefix-caching limit).")


if __name__ == "__main__":
    main()
