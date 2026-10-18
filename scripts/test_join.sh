#!/usr/bin/env bash
# End-to-end test of the address->block join (the load-bearing correlation):
# real KV tensor + real D2H cudaMemcpyAsync + real eBPF capture -> resolve each
# copy back to its KV block. Needs sudo (uprobe) + the built kv_copy collector.
set -uo pipefail
EKV="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-/opt/conda/bin/python3}"
OUT=/tmp/ekv_copy.jsonl ERR=/tmp/ekv_copy.err

CU=$($PY -c "import torch; torch.randn(4,device='cuda'); print([l.split()[-1] for l in open('/proc/self/maps') if 'cu13' in l and 'libcudart' in l][0])")
echo "libcudart : $CU"
make -s -C "$EKV/observe/ebpf" kv_copy
: > "$OUT"; : > "$ERR"
sudo "$EKV/observe/ebpf/kv_copy" "$CU" > "$OUT" 2> "$ERR" &
for _ in $(seq 1 4000); do grep -q "collector running" "$ERR" && break; done
grep -q "collector running" "$ERR" || { echo "FAIL: collector not ready"; cat "$ERR"; sudo pkill -f "$EKV/observe/ebpf/kv_copy"; exit 1; }

PYTHONPATH="$EKV" "$PY" "$EKV/tests/_join_workload.py"
sudo pkill -f "$EKV/observe/ebpf/kv_copy" 2>/dev/null

PYTHONPATH="$EKV" "$PY" "$EKV/tests/_join_check.py"