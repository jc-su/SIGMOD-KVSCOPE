#!/usr/bin/env bash
# End-to-end: the pass-captured query length drives the recompute/cache-hit + park classification
# on one live run, read per-step via /proc, cross-validated vs the engine. Needs triton-src + GPU.
set -uo pipefail
EKV="$(cd "$(dirname "$0")/.." && pwd)"
export TRITON_EXT_ENABLED=1 TRITON_PLUGIN_PATHS="$EKV/build/libeKVPlugin.so"
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
export PYTHONPATH="/path/to/triton/python:$EKV"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/ekv_e2e_cache}"
export EKV_NCAP=128 EKV_MAX_ROWS=64 EKV_HEAD_AXIS=1 EKV_SLOT_AXIS=0
export EKV_GPU_BLOCKS="${EKV_GPU_BLOCKS:-96}"
rm -rf "$TRITON_CACHE_DIR"
timeout 420 "${PY:-/opt/conda/bin/python3}" "$EKV/tests/_recompute_e2e_check.py" 2>&1 \
  | grep -vE "WARNING|INFO|Rendering|Processed|Loading|it/s\]$|Transient|^\s*$"
