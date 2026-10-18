#!/usr/bin/env bash
# BRING-UP PROBE (research, NOT a passing gate -- excluded from `make test`): how far does the eKV
# Triton pass GENERALIZE past vLLM? Runs SGLang's Triton attention backend under the SAME source-triton
# + pass env the vLLM index_lookup uses, PLUS a zero-touch `sitecustomize.py` bootstrap that runs eKV's
# engine-agnostic register() in every interpreter (incl. SGLang's SPAWNED scheduler subprocess) with
# ZERO edits to SGLang -- the universal form of the plugin register (idiomatic form: an
# `sglang.srt.plugins` setuptools entry point, the analogue of vLLM's `vllm.general_plugins`).
#
# FINDING (2026-06-22, SGLang 0.5.12, GPU1) -- VALIDATED-GENERAL after the eKVTimer.cpp CSR/GEMV/i64 fix:
#   * SUBPROCESS BOOTSTRAP (#3) is ZERO-TOUCH: register() runs in 4 pids (main + scheduler + workers) via
#     sitecustomize -- a PLUGIN REGISTER, not an engine touch. Contribution intact.
#   * The pass detects SGLang's attention kernels NAME-FREE (set EKV_DEBUG=1 to see): decode
#     `_fwd_kernel_stage1` (GEMV: tt.reduce, NO tt.dot -- the DOMINANT path) AND prefill `_fwd_kernel`
#     (GEMM: tt.dot) both select via the contraction rule (dot OR reduce); leaf=kv_indices (i64),
#     base=K_Buffer; `_fwd_kernel_stage2` + token-pool helpers excluded.
#   * CORRECTNESS: tests/_sglang_oracle_run.sh proves buffer[r]==kv_indices[kv_indptr[r]:] for 96/96
#     decode requests vs SGLang's own arrays (names poisoned). The CSR additive capture is masked to each
#     request's KV length (ragged array) and reads i64 indices truncated to i32.
# This script just shows the run fires + publishes; the oracle is the correctness gate. See GENERALIZATION.md S8.
# Run: bash tests/_sglang_kernel_run.sh   (EKV_DEBUG=1 for per-kernel decisions)
set -u
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
PY="${EKV_SGLANG_PY:-/opt/conda/bin/python3}"   # the interp that has sglang 0.5.12

# --- zero-touch bootstrap: a sitecustomize.py that every interpreter (incl. spawned children) auto-runs.
SITE="$(mktemp -d /tmp/ekv_site.XXXXXX)"
cat > "$SITE/sitecustomize.py" <<PYEOF
import os, sys
if os.environ.get("EKV_ENABLE") and os.environ.get("EKV_PASS_AUTO"):
    try:
        sys.path.insert(0, "$ROOT")
        import ekv_vllm_plugin            # register() is engine-agnostic: only installs the Triton hook
        ekv_vllm_plugin.register()
        print(f"[ekv-site] register() ran in pid {os.getpid()}", file=sys.stderr, flush=True)
    except Exception as e:
        print(f"[ekv-site] FAILED pid {os.getpid()}: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
PYEOF

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"        # GPU1 only -- never touch GPU0 (co-tenant)
export PYTHONPATH="$SITE:/path/to/triton/python:${ROOT}"   # sitecustomize dir FIRST
export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libstdc++.so.6"   # source libtriton needs newer libstdc++
export TRITON_EXT_ENABLED="1" TRITON_PLUGIN_PATHS="${ROOT}/build/libeKVPlugin.so"
export TRITON_CACHE_DIR="$(mktemp -d /tmp/ekv_sglang_cache.XXXXXX)"
export EKV_ENABLE="1" EKV_PASS_AUTO="1" EKV_TIMER="1" EKV_DUMP_IR="${EKV_DUMP_IR:-1}"
export EKV_NCAP="128" EKV_MAX_ROWS="64" EKV_HEAD_AXIS="1" EKV_SLOT_AXIS="0"   # stride is structural now

before=$(find /tmp -maxdepth 1 -name 'ekv_pass_*.json' 2>/dev/null | wc -l)
OUT="$(mktemp /tmp/ekv_sglang_run.XXXXXX.out)"
timeout 420 "$PY" tests/_sglang_kernel_probe.py > "$OUT" 2>&1
after=$(find /tmp -maxdepth 1 -name 'ekv_pass_*.json' 2>/dev/null | wc -l)

echo "---- SGLang generalization probe ----"
grep -E "GEN_OK|ENGINE_INIT_FAIL" "$OUT" | head -1
echo "register() ran in $(grep -c '\[ekv-site\] register() ran' "$OUT") pids (main + spawned scheduler/workers => #3 bootstrap is zero-touch)"
echo "SGLang kernels the pass SAW:   $(grep -oE 'tt.func public @[A-Za-z0-9_]+' "$OUT" | sort -u | sed 's/tt.func public //' | paste -sd, -)"
echo "kernels with tt.dot (selectable): $(awk '/tt.func public @/{k=$0; sub(/.*@/,"@",k); sub(/[^A-Za-z0-9_@].*/,"",k)} /tt\.dot/{print k}' "$OUT" | sort -u | paste -sd, -)"
echo "pass buffers published by this run: $((after - before))  (>=1 => PREFILL captured; decode GEMV still excluded -- see GENERALIZATION.md S8)"
rm -rf "$SITE"