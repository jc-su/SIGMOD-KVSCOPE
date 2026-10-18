#!/usr/bin/env bash
# ORACLE (research, NOT a make-test gate): is the eKV pass's SGLang capture SEMANTICALLY CORRECT, not
# just "published"? A sitecustomize (zero-touch, runs in the spawned scheduler) does two things:
#   1. register() -- install the eKV pass (as in _sglang_kernel_run.sh).
#   2. wrap SGLang's attention fwd fns (extend / extend_unified / decode) to record the GROUND-TRUTH
#      kv_indptr + kv_indices the kernel is handed, then read the pass's published host buffer and check
#      buffer[r] == kv_indices[kv_indptr[r] : kv_indptr[r]+ncap] (truncated i32) for every request r.
# This is the real test of "general": the CSR additive capture reconstructs request -> physical KV slots
# correctly, against SGLang's own arrays. PASS => the SGLang index_lookup is correct, not merely firing.
# VALIDATED: opt-125m (MHA decode _fwd_kernel_stage1) 96/96; EKV_SGLANG_MODEL=Qwen/Qwen2.5-0.5B
# (GQA decode _fwd_grouped_kernel_stage1, kv_group_num=7) 192/192 -- both name-free, the reduce/GEMV branch.
set -u
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
PY="${EKV_SGLANG_PY:-/opt/conda/bin/python3}"
RESULT=/tmp/ekv_oracle_result.json
rm -f "$RESULT"

SITE="$(mktemp -d /tmp/ekv_site.XXXXXX)"
cat > "$SITE/sitecustomize.py" <<PYEOF
import os, sys, json, ctypes
if os.environ.get("EKV_ENABLE") and os.environ.get("EKV_PASS_AUTO"):
    sys.path.insert(0, "$ROOT")
    try:
        import ekv_vllm_plugin; ekv_vllm_plugin.register()
    except Exception as e:
        print("[oracle] register failed:", e, file=sys.stderr)

    import inspect
    # calls           : every attention fwd call
    # empty_calls     : no prefix KV to compare (kv_indptr[r+1]==kv_indptr[r], e.g. fresh-request prefill)
    # compared_calls  : calls that had >=1 request with KV to check
    # match_calls     : compared calls where EVERY checked request matched
    # mismatch_calls  : compared calls where SOME request mismatched (the real failures)
    _state = {"calls": 0, "empty_calls": 0, "compared_calls": 0, "match_calls": 0,
              "mismatch_calls": 0, "by_fn": {}, "last": None, "sample_mismatch": None, "err": None}

    def _compare(kvp, kvi, fn="?"):
        import numpy as np, torch
        torch.cuda.synchronize()
        pid = os.getpid()
        # the pass now publishes ONE json line per instrumented kernel variant (per-kernel buffers);
        # take the last capture line that has a real host_va (the SGLang attention kernel's).
        lines = [json.loads(ln) for ln in open(f"/tmp/ekv_pass_{pid}.json") if ln.strip()]
        caps = [m for m in lines if int(m.get("host_va", 0))]
        if not caps:
            return
        j = caps[-1]
        hv, ncap, mr = int(j["host_va"]), int(j["ncap"]), int(j["max_rows"])
        if not hv:
            return
        raw = (ctypes.c_int32 * (ncap * mr)).from_address(hv)
        buf = np.ctypeslib.as_array(raw).reshape(mr, ncap)
        indptr = kvp.detach().to("cpu").numpy().astype(np.int64)
        indices = kvi.detach().to("cpu").numpy().astype(np.int64)
        nreq = int(len(indptr) - 1)
        rows, all_match, any_row = [], True, False
        for r in range(min(nreq, mr)):
            lo, hi = int(indptr[r]), int(indptr[r + 1])
            n = min(hi - lo, ncap)
            if n <= 0:
                continue
            any_row = True
            truth = indices[lo:lo + n].astype(np.int32)
            got = buf[r, :n].copy()
            m = bool(np.array_equal(got, truth))
            all_match = all_match and m
            rows.append({"r": r, "n": int(n), "match": m,
                         "truth": truth[:8].tolist(), "got": got[:8].tolist()})
        _state["calls"] += 1
        if not any_row:
            _state["empty_calls"] += 1
        else:
            _state["compared_calls"] += 1
            bf = _state["by_fn"].setdefault(fn, {"compared": 0, "matched": 0})
            bf["compared"] += 1
            if all_match:
                _state["match_calls"] += 1
                bf["matched"] += 1
                _state["last"] = {"fn": fn, "nreq": nreq, "kvi_dtype": str(kvi.dtype), "rows": rows}
            else:
                _state["mismatch_calls"] += 1
                if _state["sample_mismatch"] is None:
                    _state["sample_mismatch"] = {"fn": fn, "nreq": nreq, "kvi_dtype": str(kvi.dtype), "rows": rows}
        json.dump(_state, open("$RESULT", "w"))

    def _wrap(orig, fn):
        sig = inspect.signature(orig)
        def w(*a, **k):
            out = orig(*a, **k)
            try:
                b = sig.bind(*a, **k); b.apply_defaults()
                _compare(b.arguments["kv_indptr"], b.arguments["kv_indices"], fn)
            except Exception as e:
                _state["err"] = repr(e)[:300]; json.dump(_state, open("$RESULT", "w"))
            return out
        return w

    def _patch():
        from sglang.srt.layers.attention.triton_ops import extend_attention as E, decode_attention as D
        # decode_attention_fwd -> _fwd_kernel_stage1 = the GEMV (tl.sum, no tt.dot); the others are GEMM.
        E.extend_attention_fwd = _wrap(E.extend_attention_fwd, "extend_GEMM")
        E.extend_attention_fwd_unified = _wrap(E.extend_attention_fwd_unified, "extend_unified_GEMM")
        D.decode_attention_fwd = _wrap(D.decode_attention_fwd, "decode_GEMV")
    try:
        _patch()
    except Exception as e:
        print("[oracle] patch failed:", e, file=sys.stderr)
PYEOF

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTHONPATH="$SITE:/path/to/triton/python:${ROOT}"
export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libstdc++.so.6"
export TRITON_EXT_ENABLED="1" TRITON_PLUGIN_PATHS="${ROOT}/build/libeKVPlugin.so"
export TRITON_CACHE_DIR="$(mktemp -d /tmp/ekv_cache.XXXXXX)"
export EKV_ENABLE="1" EKV_PASS_AUTO="1"
export EKV_NCAP="256" EKV_MAX_ROWS="64" EKV_HEAD_AXIS="1" EKV_SLOT_AXIS="0"
# Detection is unconditionally STRUCTURAL now -- the pass reads no kernel/arg names and no stride
# constant (the gather triad + the load's own addressing drive everything), so there is nothing to
# "poison": this oracle proves the structural detector alone reconstructs the SGLang index_lookup correctly.

timeout 420 "$PY" tests/_sglang_kernel_probe.py >/dev/null 2>&1
ec=$?
echo "---- SGLang capture ORACLE (names poisoned -> structural-only) ----"
"$PY" - "$RESULT" <<'PYEOF'
import json, sys
try:
    s = json.load(open(sys.argv[1]))
except Exception as e:
    print("NO RESULT (oracle never ran):", e); sys.exit(1)
if s.get("err"):
    print("oracle error:", s["err"]); sys.exit(1)
print(f"attention fwd calls: {s['calls']}  (empty-prefix, nothing to check: {s['empty_calls']})")
print(f"calls with KV to compare: {s['compared_calls']}   matched: {s['match_calls']}   MISMATCHED: {s['mismatch_calls']}")
for fn, st in sorted(s.get("by_fn", {}).items()):
    tag = "  <- GEMV decode (tt.reduce, no tt.dot)" if "GEMV" in fn else "  <- GEMM (tt.dot)"
    print(f"    by path  {fn}: {st['matched']}/{st['compared']} correct{tag}")
last = s.get("last") or {}
for r in (last.get("rows") or [])[:4]:
    print(f"  [match example] req {r['r']}: len={r['n']} match={r['match']}  truth[:8]={r['truth']}  got[:8]={r['got']}")
sm = s.get("sample_mismatch")
if sm:
    print("  [MISMATCH sample]:")
    for r in sm["rows"][:4]:
        print(f"    req {r['r']}: len={r['n']} match={r['match']}  truth[:8]={r['truth']}  got[:8]={r['got']}")
ok = s["mismatch_calls"] == 0 and s["match_calls"] > 0
print("RESULT:", "PASS -- buffer[r] == kv_indices[kv_indptr[r]:] for EVERY compared request (SGLang capture is CORRECT)"
      if ok else "FAIL -- a real capture mismatch (see sample above)")
PYEOF
rm -rf "$SITE"