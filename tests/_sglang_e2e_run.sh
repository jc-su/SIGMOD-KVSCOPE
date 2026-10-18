#!/usr/bin/env bash
# SGLang SERVING e2e (gap 2): a real sglang.Engine serves a batch; a zero-touch sitecustomize (in the
# spawned scheduler) registers the eKV pass and wraps the attention fwd to ACCUMULATE, per request,
# eKV's pass-captured KV slots AND SGLang's own kv_indices ground truth over the whole run. The check
# validates the per-request reconstruction is exact and computes the accounting use case. Detection is
# unconditionally STRUCTURAL (no names). GPU 1 ONLY (GPU 0 is a co-tenant).
set -u
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
DEFAULT_SGLANG_PY=/path/to/sglang-env/bin/python
if [ ! -x "$DEFAULT_SGLANG_PY" ]; then
  DEFAULT_SGLANG_PY=/opt/conda/bin/python3
fi
PY="${EKV_SGLANG_PY:-$DEFAULT_SGLANG_PY}"
RESULT=/tmp/ekv_sglang_e2e.json
COVERAGE=/tmp/ekv_sglang_coverage.json
rm -f "$RESULT" "$COVERAGE"

[ -f "$ROOT/build/libeKVPlugin.so" ] || { echo "FAIL: build the pass (bash scripts/build_plugin.sh)"; exit 1; }
SITE="$(mktemp -d /tmp/ekv_e2e_site.XXXXXX)"
cat > "$SITE/sitecustomize.py" <<PYEOF
import os, sys, json, ctypes
if os.environ.get("EKV_ENABLE") and os.environ.get("EKV_PASS_AUTO"):
    sys.path.insert(0, "$ROOT")
    try:
        import ekv_vllm_plugin; ekv_vllm_plugin.register()
    except Exception as e:
        print("[e2e] register failed:", e, file=sys.stderr)
    import inspect
    _acc = {}   # anchor(first slot) -> {"ekv": set, "truth": set}
    _st = {"calls": 0, "err": None}

    def _accum(kvp, kvi, fn="?"):
        import numpy as np, torch
        torch.cuda.synchronize()                       # visibility: read the buffer after completion
        try:
            lines = [json.loads(l) for l in open(f"/tmp/ekv_pass_{os.getpid()}.json") if l.strip()]
        except OSError:
            return
        caps = [m for m in lines if int(m.get("host_va", 0))]
        if not caps:
            return
        j = caps[-1]                                   # per-compile lines; last capture = decode variant
        hv, ncap, mr = int(j["host_va"]), int(j["ncap"]), int(j["max_rows"])
        raw = (ctypes.c_int32 * (ncap * mr)).from_address(hv)
        buf = np.ctypeslib.as_array(raw).reshape(mr, ncap)
        indptr = kvp.detach().to("cpu").numpy().astype(np.int64)
        indices = kvi.detach().to("cpu").numpy().astype(np.int64)
        _st["calls"] += 1
        for r in range(min(int(len(indptr) - 1), mr)):
            lo, hi = int(indptr[r]), int(indptr[r + 1])
            n = min(hi - lo, ncap)
            if n <= 0:
                continue
            truth = [int(x) for x in indices[lo:lo + n]]
            got = [int(x) for x in buf[r, :n] if x >= 0]     # eKV pass capture (positive slots)
            anchor = truth[0]                          # first slot = content-addressed request anchor
            a = _acc.setdefault(anchor, {"ekv": set(), "truth": set()})
            a["ekv"].update(got)
            a["truth"].update(truth)
        json.dump({str(k): {"ekv": sorted(v["ekv"]), "truth": sorted(v["truth"])}
                   for k, v in _acc.items()}, open("$RESULT", "w"))
        if os.environ.get("EKV_COVERAGE"):
            from observe.coverage import report
            json.dump(report(os.getpid()), open("$COVERAGE", "w"))

    def _wrap(orig, fn):
        sig = inspect.signature(orig)
        def w(*a, **k):
            out = orig(*a, **k)
            try:
                b = sig.bind(*a, **k); b.apply_defaults()
                _accum(b.arguments["kv_indptr"], b.arguments["kv_indices"], fn)
            except Exception as e:
                _st["err"] = repr(e)[:300]
            return out
        return w

    def _patch():
        import sglang.srt.layers.attention.triton_ops.extend_attention as E
        import sglang.srt.layers.attention.triton_ops.decode_attention as D
        if hasattr(E, "extend_attention_fwd"):
            E.extend_attention_fwd = _wrap(E.extend_attention_fwd, "extend")
        if hasattr(E, "extend_attention_fwd_unified"):
            E.extend_attention_fwd_unified = _wrap(E.extend_attention_fwd_unified, "extend_unified")
        if hasattr(D, "decode_attention_fwd"):
            D.decode_attention_fwd = _wrap(D.decode_attention_fwd, "decode")
    try:
        _patch()
    except Exception as e:
        print("[e2e] patch failed:", e, file=sys.stderr)
PYEOF

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTHONPATH="$SITE:/path/to/triton/python:${ROOT}"
export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libstdc++.so.6"
export TRITON_EXT_ENABLED="1" TRITON_PLUGIN_PATHS="${ROOT}/build/libeKVPlugin.so"
CACHE="$(mktemp -d /tmp/ekv_e2e_cache.XXXXXX)"
export TRITON_CACHE_DIR="$CACHE"
export EKV_ENABLE="1" EKV_PASS_AUTO="1" EKV_COVERAGE="1"
export EKV_NCAP="512" EKV_MAX_ROWS="64" EKV_HEAD_AXIS="1" EKV_SLOT_AXIS="0"

timeout 500 "$PY" tests/_sglang_e2e_probe.py >/tmp/ekv_sglang_e2e.out 2>&1
grep -E "GEN_OK|ENGINE_INIT_FAIL|Traceback" /tmp/ekv_sglang_e2e.out | head -3
echo "---- SGLang serving e2e: eKV per-request KV reconstruction + accounting ----"
"$PY" tests/_sglang_e2e_check.py "$RESULT"
rc=$?
if [ -s "$COVERAGE" ]; then
  "$PY" - "$COVERAGE" <<'PY'
import json, sys
record = json.load(open(sys.argv[1]))
print("EKV_SGLANG_COVERAGE " + json.dumps(record, sort_keys=True))
PY
else
  echo "FAIL: SGLang coverage record missing"
  rc=1
fi
rm -rf "$SITE" "$CACHE"
exit $rc
