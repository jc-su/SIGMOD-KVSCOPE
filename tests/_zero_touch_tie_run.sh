#!/usr/bin/env bash
# ZERO-TOUCH request<->block, full tie: eBPF EKV_META (input_ids -> seq->request) + the
# eKV pass (block_table -> seq->block, external /proc read), joined. eBPF + pass only.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=/path/to/triton/python
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
export TRITON_EXT_ENABLED=1
export TRITON_PLUGIN_PATHS=$(cd "$(dirname "$0")/.." && pwd)/build/libeKVPlugin.so
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export VLLM_ATTENTION_BACKEND=TRITON_ATTN VLLM_USE_FLASHINFER=0 VLLM_USE_FLASHINFER_SAMPLER=0
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export EKV_PASS_AUTO=1 EKV_TIMER=1 EKV_NCAP="${EKV_NCAP:-128}"
export EKV_MAX_ROWS="${EKV_MAX_ROWS:-128}" EKV_HEAD_AXIS="${EKV_HEAD_AXIS:-1}" EKV_SLOT_AXIS=0
export EKV_GPU_BLOCKS="${EKV_GPU_BLOCKS:-2048}"
LIBCUDART=/opt/conda/lib/python3.12/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12
rm -f /tmp/ekv_meta.jsonl /tmp/ekv_pass_*.json; rm -rf ~/.triton/cache

sudo EKV_DRIVER_ONLY=1 EKV_META=1 ./observe/ebpf/kv_copy "$LIBCUDART" >/tmp/ekv_meta.jsonl 2>/dev/null &
COL=$!
cleanup() { sudo kill "$COL" 2>/dev/null; sudo pkill -9 -x kv_copy 2>/dev/null; }
trap cleanup EXIT
sleep 2.5

timeout 590 python tests/_zero_touch_tie_check.py 2>&1 \
  | grep -vE "WARNING|INFO|Rendering|Processed|Loading|it/s\]$|^\s*$|^\[eKV\] self-alloc"
echo "(eBPF meta lines: $(wc -l </tmp/ekv_meta.jsonl 2>/dev/null || echo 0))"
