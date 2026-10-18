#!/usr/bin/env bash
# Verify the collector captures the DRIVER-API cuMemcpyBatchAsync (vLLM's KV
# offload path) — which the runtime cudaMemcpyAsync probe cannot see — and that
# the fuser resolves the batch elements to KV blocks as D2H edges.
set -uo pipefail
EKV="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-/opt/conda/bin/python3}"
OUT=/tmp/ekv_copy.jsonl ERR=/tmp/ekv_copy.err

CU=$($PY -c "import torch; torch.randn(4,device='cuda'); print([l.split()[-1] for l in open('/proc/self/maps') if 'cu13' in l and 'libcudart' in l][0])")
make -s -C "$EKV/observe/ebpf" kv_copy
: > "$OUT"; : > "$ERR"
sudo "$EKV/observe/ebpf/kv_copy" "$CU" > "$OUT" 2> "$ERR" &
for _ in $(seq 1 4000); do grep -q "collector running" "$ERR" && break; done
grep -q "collector running" "$ERR" || { echo "FAIL: not ready"; cat "$ERR"; sudo pkill -f "$EKV/observe/ebpf/kv_copy"; exit 1; }
grep -q "driver-API.*attached" "$ERR" || echo "WARN: driver-API probes not attached"

PYTHONPATH="$EKV" "$PY" "$EKV/tests/_driver_offload_workload.py"
sudo pkill -f "$EKV/observe/ebpf/kv_copy" 2>/dev/null

PYTHONPATH="$EKV" "$PY" "$EKV/tests/_driver_offload_check.py"