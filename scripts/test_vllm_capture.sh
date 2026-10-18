#!/usr/bin/env bash
# Go/no-go: attach the eBPF KV-copy collector to a REAL vLLM process and confirm
# it captures real CUDA copy/completion traffic -- and which completion path vLLM
# actually uses. Result on this box: vLLM uses BOTH stream-sync AND CUDA events,
# so a stream-sync-only collector would miss the event-path completions.
#
# Env notes: use the Triton attention backend to avoid flashinfer's nvcc JIT, and
# point CPATH at the CUDA headers (flashinfer JIT needs curand.h). opt-125m,
# offline, engine in-process. The eBPF collector is Triton-version-independent
# (it attaches to libcudart), so this runs on stock vLLM 0.21 / pip-triton 3.6.
set -uo pipefail
EKV="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-/opt/conda/bin/python3}"
OUT=/tmp/ekv_vllm.jsonl ERR=/tmp/ekv_vllm.err

CU=$($PY -c "import torch; torch.randn(4,device='cuda'); print([l.split()[-1] for l in open('/proc/self/maps') if 'cu13' in l and 'libcudart' in l][0])")
make -s -C "$EKV/observe/ebpf" kv_copy
: > "$OUT"; : > "$ERR"
sudo "$EKV/observe/ebpf/kv_copy" "$CU" > "$OUT" 2> "$ERR" &
for _ in $(seq 1 4000); do grep -q "collector running" "$ERR" && break; done
grep -q "collector running" "$ERR" || { echo "FAIL: collector not ready"; cat "$ERR"; sudo pkill -f "$EKV/observe/ebpf/kv_copy"; exit 1; }

export HF_HUB_OFFLINE=1 EKV_MODEL="${EKV_MODEL:-facebook/opt-125m}"
export VLLM_ATTENTION_BACKEND=TRITON_ATTN VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_USE_FLASHINFER=0
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.9}"
export CPATH="${CPATH:-/usr/local/cuda-12.9/targets/x86_64-linux/include}"
timeout 280 "$PY" "$EKV/tests/_vllm_gen.py" 2>&1 | grep -iE "generated|error|fatal|curand" | head -5
sudo pkill -f "$EKV/observe/ebpf/kv_copy" 2>/dev/null

ENQ=$(grep -c '"type":"enqueue"' "$OUT"); ENQ=${ENQ:-0}
SS=$(grep '"type":"completion"' "$OUT" | grep -c '"via":"stream-sync"'); SS=${SS:-0}
EV=$(grep '"type":"completion"' "$OUT" | grep -c '"via":"event-sync"'); EV=${EV:-0}
echo "captured from real vLLM: enqueues=$ENQ  completions stream-sync=$SS event-sync=$EV"
grep -i drop "$ERR" | tail -1
if [ "$ENQ" -ge 50 ] && [ "$EV" -ge 1 ]; then
  echo "PASS: collector captures real vLLM traffic; vLLM uses the CUDA-event completion path (stream-sync-only would miss it)"
else
  echo "FAIL"; exit 1
fi