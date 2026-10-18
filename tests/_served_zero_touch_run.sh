#!/usr/bin/env bash
# CAPSTONE: whole zero-touch per-request plane on a real `vllm serve` (multi-process).
# The check self-orchestrates serve + the two sudo eBPF collectors; it runs under the
# DEFAULT python (just reads files + /proc + tokenizer), while the serve SUBPROCESS gets
# the triton-src pass env (set inside the check).
set -u
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
timeout 720 python tests/_served_zero_touch_check.py 2>&1 \
  | grep -vE "WARNING|INFO|Rendering|Processed|Loading|it/s\]$|^\s*$"
