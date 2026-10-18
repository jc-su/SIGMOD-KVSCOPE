#!/usr/bin/env bash
# Validate the seq_idx-indexed index_lookup capture under CHUNKED PREFILL (vLLM production default):
# the pass's slot->blocks must equal the engine's block_table per sequence at multi-query-block
# (program_id != seq_idx) steps. Needs the source-triton pass env + a GPU.
set -uo pipefail
EKV="$(cd "$(dirname "$0")/.." && pwd)"
export TRITON_EXT_ENABLED=1 TRITON_PLUGIN_PATHS="$EKV/build/libeKVPlugin.so"
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
export PYTHONPATH="/path/to/triton/python:$EKV"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/ekv_chunked_cache}"
export EKV_NCAP=128 EKV_MAX_ROWS=64 EKV_HEAD_AXIS=1 EKV_SLOT_AXIS=0
export EKV_GPU_BLOCKS="${EKV_GPU_BLOCKS:-2048}"
rm -rf "$TRITON_CACHE_DIR"
timeout 480 python "$EKV/tests/_chunked_prefill_check.py" 2>&1 \
  | grep -vE "WARNING|INFO|Rendering|Processed|Loading|it/s\]$|Transient|^\s*$"
