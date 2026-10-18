#!/usr/bin/env bash
# #2 tensor-parallel index_lookup rank-invariance. Uses BOTH GPUs (the eKV pass loads in each
# TP worker via the vLLM general-plugin entry point). Needs 2 free GPUs + sudo (/proc read).
set -uo pipefail
EKV="$(cd "$(dirname "$0")/.." && pwd)"
export TRITON_EXT_ENABLED=1 TRITON_PLUGIN_PATHS="$EKV/build/libeKVPlugin.so"
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
export PYTHONPATH="/path/to/triton/python:$EKV"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/ekv_tp_cache}"
# the plugin auto-installs the pass in EVERY vLLM process (incl. each TP worker)
export EKV_ENABLE=1 EKV_PASS_AUTO=1
export EKV_NCAP=128 EKV_MAX_ROWS=64 EKV_HEAD_AXIS=1 EKV_SLOT_AXIS=0
rm -rf "$TRITON_CACHE_DIR"
timeout 540 "${PY:-/opt/conda/bin/python3}" "$EKV/tests/_tp_index_lookup_check.py" 2>&1 \
  | grep -vE "WARNING|INFO|Rendering|Processed|Loading|it/s\]$|Transient|^\s*$"
