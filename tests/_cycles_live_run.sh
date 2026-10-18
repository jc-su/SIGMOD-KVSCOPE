#!/usr/bin/env bash
# LIVE per-request cycles gate (Phase-1 spike, step C): stock vLLM opt-125m on the Triton attention
# backend with the pass + EKV_TIMER injected by the registered plugin (EKV_ENABLE). Validates
# per-request keying on the real chunked-prefill kernel and cycles ~ per-request attention area.
# GPU 1 ONLY (GPU 0 is a co-tenant).
set -u
cd "$(dirname "$0")/.."
EKV="$(pwd)"
SO="$EKV/build/libeKVPlugin.so"
LIBSTDCPP=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
TR=/path/to/triton/python
OUT=/tmp/ekv_cycles_live.log

[ -f "$SO" ] || { echo "FAIL: build the pass first (bash scripts/build_plugin.sh)"; exit 1; }
CACHE=$(mktemp -d /tmp/ekv_cyclive_cache.XXXXXX)
TRITON_CACHE_DIR="$CACHE" VLLM_CACHE_ROOT="$CACHE/vllm" \
  EKV_TIMER=1 EKV_NCAP=128 EKV_MAX_ROWS=1024 EKV_DEBUG=1 \
  TRITON_EXT_ENABLED=1 TRITON_PLUGIN_PATHS="$SO" EKV_ENABLE=1 EKV_PASS_AUTO=1 \
  PYTHONPATH="$TR:$EKV" LD_PRELOAD="$LIBSTDCPP" CUDA_VISIBLE_DEVICES=1 \
  timeout 420 python "$EKV/tests/_cycles_live_check.py" >"$OUT" 2>&1
rc=$?
grep -E "prompt token|timer rows|content-addressed|shares:|window:|window_series|whole-step|corr\(|PASS|FAIL|Error|Traceback" "$OUT" | tail -14
exit $rc
