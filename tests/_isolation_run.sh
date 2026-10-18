#!/usr/bin/env bash
# Request-isolation audit from compiler-captured request rows on GPU 1.
set -euo pipefail
cd "$(dirname "$0")/.."
EKV="$(pwd)"
SO="$EKV/build/libeKVPlugin.so"
[ -f "$SO" ] || { echo "FAIL: build the pass first"; exit 1; }
CACHE=$(mktemp -d /tmp/ekv_isolation_cache.XXXXXX)
trap 'rm -rf "$CACHE"' EXIT
EKV_NCAP=128 EKV_MAX_ROWS=1024 TRITON_CACHE_DIR="$CACHE" VLLM_CACHE_ROOT="$CACHE/vllm" \
  TRITON_EXT_ENABLED=1 TRITON_PLUGIN_PATHS="$SO" EKV_ENABLE=1 EKV_PASS_AUTO=1 \
  PYTHONPATH="/path/to/triton/python:$EKV" \
  LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 CUDA_VISIBLE_DEVICES=1 \
  timeout 400 python "$EKV/tests/_isolation_probe.py"
