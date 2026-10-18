#!/usr/bin/env bash
# End-to-end test for the eBPF KV-copy collector: attach to torch's libcudart,
# run a workload of 5x4MB D2H cudaMemcpyAsync copies, assert they are captured.
# Needs sudo (passwordless here) for the uprobe.
set -uo pipefail
EKV="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-/opt/conda/bin/python3}"
OUT=/tmp/ekv_ebpf.jsonl ERR=/tmp/ekv_ebpf.err

CU=$($PY -c "import torch; torch.randn(4,device='cuda'); print([l.split()[-1] for l in open('/proc/self/maps') if 'cu13' in l and 'libcudart' in l][0])")
echo "libcudart : $CU"

make -s -C "$EKV/observe/ebpf"
: > "$OUT"; : > "$ERR"
sudo "$EKV/observe/ebpf/kv_copy" "$CU" > "$OUT" 2> "$ERR" &
# wait (bounded, no foreground sleep) for the collector to be ready
for _ in $(seq 1 3000); do grep -q "collector running" "$ERR" && break; done
if ! grep -q "collector running" "$ERR"; then echo "FAIL: collector not ready"; cat "$ERR"; sudo pkill -f "$EKV/observe/ebpf/kv_copy"; exit 1; fi

"$PY" "$EKV/tests/_ebpf_workload.py"
sudo pkill -f "$EKV/observe/ebpf/kv_copy" 2>/dev/null

echo "=== collector stderr ==="; cat "$ERR"
echo "=== captured events ==="; cat "$OUT"
ENQ=$(grep -c '"type":"enqueue".*"kind":"D2H"' "$OUT" 2>/dev/null || echo 0)
COMP=$(grep -c '"type":"completion"' "$OUT" 2>/dev/null || echo 0)
COMP_LAT=$(grep '"type":"completion"' "$OUT" | grep -c '"latency_ns":[1-9]' 2>/dev/null || echo 0)
echo "enqueue(D2H)=$ENQ  completion=$COMP  completion-with-latency>0=$COMP_LAT"
if [ "$ENQ" -ge 5 ] && [ "$COMP" -ge 5 ] && [ "$COMP_LAT" -ge 5 ]; then
  echo "PASS: eBPF captured D2H enqueue AND completion latency (no CUPTI)"
else
  echo "FAIL"; exit 1
fi
