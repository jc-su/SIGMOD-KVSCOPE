"""#4 RECOMPUTE vs RELOAD via content-hash identity, on a live engine.

The same reuse workload (8 long shared prefixes cycled through a small GPU pool, so earlier
prefixes are evicted and later reused) is run in two regimes:
  * EKV_OFFLOAD=1 : SimpleCPUOffloadConnector ON  -> a reused evicted prefix is RELOADED from
                    host (an H2D the eBPF copy face captures).
  * EKV_OFFLOAD=0 : no connector, prefix caching ON -> a reused evicted prefix is RECOMPUTED
                    (re-run prefill); there is NO copy at all.

In BOTH, the prefix's content reappears in the resident set. eKV derives each block's content
hash from its tokens (content_hashes_from_tokens, zero-touch) and tracks residency over the
scheduler steps; classify_rematerializations then calls a reappearance a RELOAD iff a captured
H2D landed in its block just before, else a RECOMPUTE. The claim: offload-ON is reload-dominated
(and H2D>0); offload-OFF is recompute-dominated (and H2D~0 — recompute is invisible to copies,
yet eKV still sees it via the content hash). Run via tests/_recompute_run.sh (both regimes).
"""
import json
import os
import sys
import time

os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from observe import (BlockAddressMap, classify_rematerializations,  # noqa: E402
                     content_hashes_from_tokens, read_copies_file)

# Three regimes that FORCE one class each, to validate the query-length signal end to end:
#   offload   : connector ON,  prefix-cache ON  -> reuse RELOADS from host (H2D copy)
#   cachehit  : connector OFF, prefix-cache ON  -> reuse is a free GPU CACHE-HIT (no copy/compute)
#   recompute : connector OFF, prefix-cache OFF -> reuse RECOMPUTES (full prefill, large query_len)
MODE = os.environ.get("EKV_MODE", "offload")
COPY_JSONL = os.environ.get("EKV_COPY_FILE", "/tmp/ekv_copy.jsonl")
RESULT = os.environ.get("EKV_RECOMPUTE_OUT", f"/tmp/ekv_recompute_{MODE}.json")
_or = {"timeline": [], "compute": [], "ranges": None, "nblk": 0}


def _slot_qlen(runner, n):
    """{slot -> query length computed this step} = cu_seqlens_q[s+1]-cu_seqlens_q[s], the
    kernel's own cur_batch_query_len (what the Triton pass can capture). A recompute re-runs
    the prefill over its prefix -> large query_len; a cache-hit/decode reads from cache ->
    query_len ~1. This is the EXACT compute signal that separates recompute from cache-hit."""
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


def _slot_tokens(ib, n):
    """{slot -> token_ids} for the n active requests (defensive across vLLM versions). The
    prefix blocks (full block_size chunks from token 0) hash identically regardless of the
    exact tail count, which is all the reload/recompute tracking needs."""
    tok = getattr(ib, "token_ids_cpu", None)
    if tok is None:
        return {}
    # the FULL resident context per slot (prompt + generated so far), not just this step's
    # scheduled tokens — otherwise decode steps (1 scheduled token) hash nothing and the
    # snapshot misses what is actually resident, so content "leaving" is never recorded.
    nt = getattr(ib, "num_computed_tokens_cpu", None)
    if nt is None:
        nt = getattr(ib, "num_tokens", None)
    out = {}
    for s in range(n):
        try:
            k = int(nt[s]) if nt is not None else 0
            row = tok[s]
            out[s] = [int(x) for x in (row[:k] if k > 0 else row)]
        except Exception:  # noqa: BLE001
            pass
    return out


def _install_oracle():
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner as GR
    orig = GR.execute_model

    def execute_model(runner, *a, **k):
        out = orig(runner, *a, **k)
        try:
            now = time.monotonic_ns()
            ib = runner.input_batch
            n = int(ib.num_reqs)
            if n > 0:
                bt = ib.block_table[0].get_device_tensor(n).detach().to("cpu").tolist()
                toks = _slot_tokens(ib, n)
                qlen = _slot_qlen(runner, n)                     # per-slot computed-token count
                snap, comp = {}, {}
                for s in range(min(n, len(bt))):
                    sb = {s: [int(b) for b in bt[s] if int(b) > 0]}
                    bh = content_hashes_from_tokens(sb, {s: toks.get(s, [])})
                    for blk, h in bh.items():
                        snap[h] = blk                            # {content hash -> block}
                        comp[h] = qlen.get(s, 0)                 # {content hash -> its slot's query_len}
                if snap:
                    _or["timeline"].append((now, snap))
                    _or["compute"].append(comp)
            if _or["ranges"] is None:
                rng = []
                for t in (getattr(runner, "kv_caches", None) or []):
                    try:
                        nb = int(t.shape[0]) if t.ndim and t.shape[0] > 1 else int(t.shape[1])
                        stride = int(t.stride(0) * t.element_size()) if t.shape[0] == nb \
                            else int(t.stride(1) * t.element_size())
                        rng.append((int(t.data_ptr()), nb, stride))
                    except Exception:  # noqa: BLE001
                        pass
                if rng:
                    _or["ranges"], _or["nblk"] = rng, rng[0][1]
        except Exception:  # noqa: BLE001
            pass
        return out
    GR.execute_model = execute_model


def _workload():
    from vllm import LLM, SamplingParams
    # distinct prompts, reused across TWO sequential rounds: round 1 = first compute; round 2 =
    # re-materialization (the content fully left when round 1 finished), which the regime forces
    # to be a reload / cache-hit / recompute. A clean gap is what makes a PREFILL recompute (large
    # query_len) detectable — interleaved cycling keeps prefixes resident (only decode-block churn).
    prompts = [f"Report number {i}: " + " ".join(f"item{i}field{k}" for k in range(70))
               for i in range(8)]
    # offload must EVICT round 1 before round 2 (small pool -> connector offloads -> round 2 reloads);
    # cache-hit must RETAIN round 1 (big pool, no eviction); recompute disables the cache entirely.
    pool = {"offload": 64, "cachehit": 4096, "recompute": 4096}.get(MODE, 256)
    kw = dict(model="facebook/opt-125m", enforce_eager=True, attention_backend="TRITON_ATTN",
              gpu_memory_utilization=0.30, max_model_len=1024,
              enable_prefix_caching=(MODE != "recompute"),   # OFF -> every reuse re-derives
              enable_chunked_prefill=False, max_num_seqs=8,
              num_gpu_blocks_override=int(os.environ.get("EKV_GPU_BLOCKS", str(pool))))
    if MODE == "offload":
        from vllm.config import KVTransferConfig
        kw["kv_transfer_config"] = KVTransferConfig(
            kv_connector="SimpleCPUOffloadConnector", kv_role="kv_both",
            kv_connector_extra_config={"cpu_bytes_to_use": 4_000_000_000})
    llm = LLM(**kw)
    sp = SamplingParams(max_tokens=16, temperature=0.0, ignore_eos=True)
    llm.generate(prompts, sp)                                # round 1: first compute
    llm.generate(prompts, sp)                                # round 2: re-materialization


def _rematerialization_qlens(timeline, loads, compute):
    """For each re-materialization (a hash that left the resident set and reappeared with NO
    H2D copy), the per-slot query_len observed at its reappearance — to see whether recomputes
    actually carry a large query_len in this regime (debug for the classification)."""
    import bisect
    lt = {}
    for blk, t in loads:
        lt.setdefault(int(blk), []).append(int(t))
    for v in lt.values():
        v.sort()
    seen, present, out = set(), set(), []
    for idx, (t, snap) in enumerate(timeline):
        cur = set(snap)
        for h in cur - present:
            if h not in seen:
                continue
            blk = int(snap[h])
            ts = lt.get(blk, [])
            i = bisect.bisect_right(ts, t)
            if i > 0 and (t - ts[i - 1]) <= 500_000_000:
                continue                                   # reload (copy) — not a no-copy event
            out.append(compute[idx].get(h, 0))
        seen |= cur
        present = cur
    return out


def _addr_map():
    m = BlockAddressMap()
    for (base, nblk, stride) in (_or["ranges"] or []):
        for b in range(nblk):
            try:
                m.add(b, base + b * stride, stride)
            except ValueError:
                pass
    return m


def _resolve_loads(m):
    """H2D KV loads -> [(block_id, finished_at_ns)], resolved through the KV pool addr map."""
    if not os.path.exists(COPY_JSONL):
        return []
    loads = []
    for c in read_copies_file(COPY_JSONL):
        a = m.resolve(c.addr)
        o = m.resolve(c.other_addr) if c.other_addr is not None else None
        if c.direction == "H2D":
            blk = a or o
        elif c.direction == "D2H":
            blk = None                       # offload out, not a re-materialization
        else:
            blk = o if (o is not None and a is None) else None
        if blk is not None and c.finished_at is not None:
            loads.append((blk.block_id, c.finished_at))
    return loads


def main():
    _install_oracle()
    _workload()
    tl, ranges = _or["timeline"], _or["ranges"]
    print(f"mode={MODE}  steps={len(tl)}  "
          f"kv layers={len(ranges or [])}  blocks/layer={_or['nblk']}")
    if not tl or not ranges:
        print(f"FAIL: timeline={len(tl)} ranges={ranges is not None}")
        return
    loads = _resolve_loads(_addr_map())
    heur = classify_rematerializations(tl, loads)                       # same-block heuristic
    exact = classify_rematerializations(tl, loads, compute=_or["compute"])  # query-length EXACT
    nrl = len(exact["reload"])
    hrc, hch = len(heur["recompute"]), len(heur["cache_hit"])
    xrc, xch = len(exact["recompute"]), len(exact["cache_hit"])
    print(f"captured H2D loads (resolved to KV blocks) = {len(loads)}")
    print(f"re-materializations: reload={nrl}  first_compute={exact['first_compute']}")
    print(f"  recompute/cache-hit  (same-block heuristic): {hrc} / {hch}")
    print(f"  recompute/cache-hit  (query-length EXACT)   : {xrc} / {xch}")
    allq = [q for comp in _or["compute"] for q in comp.values()]
    rematq = sorted((exact_qlen for exact_qlen in _rematerialization_qlens(tl, loads, _or["compute"])),
                    reverse=True)
    print(f"  query_len: max-per-step={max(allq) if allq else 0}  "
          f"re-materialization query_lens (top): {rematq[:8]}")
    json.dump({"mode": MODE, "h2d_loads": len(loads),
               "reload": nrl, "recompute": xrc, "cache_hit": xch,
               "recompute_heur": hrc, "cache_hit_heur": hch,
               "first_compute": exact["first_compute"]}, open(RESULT, "w"))
    if MODE == "offload":
        ok = nrl > 0 and nrl >= xrc                          # reload dominant (H2D copy present)
        print(f"{'OK' if ok else 'CHECK'}(offload): {nrl} RELOADS, {len(loads)} H2D — reuse comes "
              f"back via a captured copy.")
    elif MODE == "cachehit":
        ok = xch > 0 and xrc == 0 and len(loads) == 0        # cache-hit dominant (no copy, no compute)
        print(f"{'OK' if ok else 'CHECK'}(cachehit): {xch} CACHE-HITS, {xrc} recompute, {len(loads)} "
              f"H2D by query-length — prefix cache serves reuse with NO copy and NO compute. "
              f"(same-block heuristic wrongly called {hrc} of these 'recompute'.)")
    else:  # recompute
        ok = xrc > 0 and len(loads) == 0                     # recompute dominant (large query_len, no copy)
        print(f"{'OK' if ok else 'CHECK'}(recompute): {xrc} RECOMPUTES, {xch} cache-hit, {len(loads)} "
              f"H2D by query-length — prefix cache OFF, so reuse re-derives (large cur_batch_query_len). "
              f"Invisible to a transfer-only view; the pass's compute signal makes it exact.")


if __name__ == "__main__":
    main()
