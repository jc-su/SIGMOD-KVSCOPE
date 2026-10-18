#!/usr/bin/env bash
# Validate the pass-side cu_seqlens_q (per-seq query length) capture vs the engine's own
# query_start_loc. Needs the source-triton pass env + a GPU.
set -uo pipefail
EKV="$(cd "$(dirname "$0")/.." && pwd)"
export TRITON_EXT_ENABLED=1 TRITON_PLUGIN_PATHS="$EKV/build/libeKVPlugin.so"
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
export PYTHONPATH="/path/to/triton/python:$EKV"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/ekv_qlen_cache}"
export EKV_NCAP=128 EKV_MAX_ROWS=64 EKV_HEAD_AXIS=1 EKV_SLOT_AXIS=0
rm -rf "$TRITON_CACHE_DIR"
timeout 360 "${PY:-/opt/conda/bin/python3}" "$EKV/tests/_qlen_capture_check.py" 2>&1 \
  | grep -vE "WARNING|INFO|Rendering|Processed|Loading|it/s\]$|Transient|^\s*$"
