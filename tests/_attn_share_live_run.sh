#!/usr/bin/env bash
# R1 gate: in-kernel attention statistic against the true attention share, on vLLM's real
# unified-attention kernel (2D and split-KV 3D), with real Llama-3.1-8B queries and KV.
# GPU 1 ONLY (GPU 0 is a co-tenant).
set -u
cd "$(dirname "$0")/.."
EKV="$(pwd)"
SO="$EKV/build/libeKVPlugin.so"
LIBSTDCPP=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
TR=/path/to/triton/python
OUT=${EKV_SHARE_OUT:-/tmp/ekv_attn_share.log}

[ -f "$SO" ] || { echo "FAIL: build the pass first (bash scripts/build_plugin.sh)"; exit 1; }
TRITON_CACHE_DIR=$(mktemp -d /tmp/ekv_share_cache.XXXXXX) \
  EKV_ATTN_SHARE=1 EKV_NCAP=512 EKV_MAX_ROWS=64 EKV_DEBUG=1 \
  TRITON_EXT_ENABLED=1 TRITON_PLUGIN_PATHS="$SO" EKV_ENABLE=1 EKV_PASS_AUTO=1 \
  PYTHONPATH="$TR:$EKV" LD_PRELOAD="$LIBSTDCPP" CUDA_VISIBLE_DEVICES=1 \
  timeout 1800 python "$EKV/tests/_attn_share_live_check.py" >"$OUT" 2>&1
rc=$?
grep -aE "^cells|^  (llama|flat|ramp)|Error|error|Traceback|assert" "$OUT" | tail -30
exit $rc
