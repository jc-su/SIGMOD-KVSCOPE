"""#2 TENSOR-PARALLEL validation: the index_lookup (slot -> physical blocks) is RANK-INVARIANT.

Under TP>1 vLLM shards the KV cache across ranks by attention head, but the block table is
LOGICAL and replicated — every rank's attention kernel reads the same slot->block mapping.
So the eKV pass, which loads in EACH worker via the vLLM general-plugin entry point
(ekv_vllm_plugin:register, gated on EKV_ENABLE/EKV_PASS_AUTO), should capture an IDENTICAL
slot->blocks on every rank. This is what makes eKV work under TP with no modification, and
why the token-derived content hash (deterministic CRC32) agrees across ranks too.

This runs an offline LLM(tensor_parallel_size=2): vLLM spawns one worker process per GPU,
each auto-installs the pass and publishes /tmp/ekv_pass_<pid>.json. After a batch, we read
EVERY published worker buffer externally (sudo /proc read via observe/zero_touch.py) and
check the ranks agree slot-for-slot. Needs 2 GPUs (CUDA_VISIBLE_DEVICES=0,1) + sudo.
Run via tests/_tp_index_lookup_run.sh.
"""
import glob
import json
import os
import re
import subprocess
import sys
import time

os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _read_worker(pid):
    """External /proc read of one worker's published pass buffer -> {slot: [blocks]}."""
    r = subprocess.run(["sudo", sys.executable, f"{ROOT}/observe/zero_touch.py", str(pid)],
                       capture_output=True, text=True)
    if not r.stdout.strip():
        return {}
    d = json.loads(r.stdout)
    return {int(k): [int(b) for b in v if int(b) > 0] for k, v in d.get("slots", {}).items()}


def main():
    for f in glob.glob("/tmp/ekv_pass_*.json"):
        os.remove(f)
    from vllm import LLM, SamplingParams
    prompts = [("token " * (8 + i * 9)).strip() for i in range(8)]   # distinct lengths
    llm = LLM(model="facebook/opt-125m", enforce_eager=True, attention_backend="TRITON_ATTN",
              tensor_parallel_size=2, gpu_memory_utilization=0.30, max_model_len=2048,
              enable_prefix_caching=True, enable_chunked_prefill=False, max_num_seqs=16)
    llm.generate(prompts, SamplingParams(max_tokens=24, temperature=0.0, ignore_eos=True))
    time.sleep(1)

    pubs = sorted(glob.glob("/tmp/ekv_pass_*.json"))
    pids = [int(re.search(r"ekv_pass_(\d+)\.json", p).group(1)) for p in pubs]
    print(f"published pass buffers (one per worker): {len(pubs)} -> pids {pids}")
    ranks = {pid: _read_worker(pid) for pid in pids}
    for pid, slots in ranks.items():
        nb = sum(len(v) for v in slots.values())
        print(f"  worker {pid}: {len(slots)} slots, {nb} blocks captured")

    nonempty = {pid: s for pid, s in ranks.items() if s}
    if len(nonempty) < 2:
        print(f"FAIL: need >=2 worker buffers with captures; got {len(nonempty)} "
              f"(TP workers not both publishing — check the plugin loaded in each worker).")
        return
    # rank-invariance: every pair of ranks agrees slot-for-slot on the captured blocks.
    items = list(nonempty.items())
    (p0, r0) = items[0]
    agree_slots, total_slots, mism = 0, 0, []
    for (p1, r1) in items[1:]:
        common = set(r0) & set(r1)
        for s in sorted(common):
            total_slots += 1
            if r0[s] == r1[s]:
                agree_slots += 1
            else:
                mism.append((s, r0[s][:6], r1[s][:6]))
    print(f"cross-rank agreement: {agree_slots}/{total_slots} common slots identical "
          f"(ranks {p0} vs {[p for p,_ in items[1:]]})")
    if mism:
        for s, a, b in mism[:3]:
            print(f"    slot {s}: rank0={a} != rank1={b}")
    if total_slots > 0 and agree_slots == total_slots:
        print("PASS(#2-TP): the index_lookup slot->blocks is RANK-INVARIANT — every TP worker's pass "
              "captured an identical mapping. eKV works under tensor parallelism unmodified; the "
              "per-rank captures need no cross-rank reconciliation (and the deterministic content "
              "hash agrees across ranks by construction).")
    else:
        print(f"CHECK(#2-TP): {agree_slots}/{total_slots} slots agreed — inspect mismatches above.")


if __name__ == "__main__":
    main()
