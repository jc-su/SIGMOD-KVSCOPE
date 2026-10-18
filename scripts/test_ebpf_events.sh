#!/usr/bin/env bash
# Verify the EVENT-based completion path (the real vLLM/LMCache offload pattern:
# cudaEventRecord + cudaEventSynchronize, not cudaStreamSynchronize). This is the
# fix for "completion never fires on real vLLM because it uses events."
set -uo pipefail
EKV="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-/opt/conda/bin/python3}"
OUT=/tmp/ekv_ev.jsonl ERR=/tmp/ekv_ev.err

CU=$($PY -c "import torch; torch.randn(4,device='cuda'); print([l.split()[-1] for l in open('/proc/self/maps') if 'cu13' in l and 'libcudart' in l][0])")
make -s -C "$EKV/observe/ebpf" kv_copy
: > "$OUT"; : > "$ERR"
sudo "$EKV/observe/ebpf/kv_copy" "$CU" > "$OUT" 2> "$ERR" &
for _ in $(seq 1 4000); do grep -q "collector running" "$ERR" && break; done
grep -q "collector running" "$ERR" || { echo "FAIL: not ready"; cat "$ERR"; sudo pkill -f "$EKV/observe/ebpf/kv_copy"; exit 1; }
grep -q "attached cudaEventSynchronize \[ret\]" "$ERR" || echo "WARN: event-sync uretprobe not attached"

"$PY" "$EKV/tests/_ebpf_events_workload.py"
sudo pkill -f "$EKV/observe/ebpf/kv_copy" 2>/dev/null

echo "=== events ==="; cat "$OUT"; echo "--- collector ---"; tail -2 "$ERR"
ENQ=$(grep '"type":"enqueue"' "$OUT" | grep -c '"kind":"D2H"'); ENQ=${ENQ:-0}
EVC=$(grep '"type":"completion"' "$OUT" | grep -c '"via":"event-sync"'); EVC=${EVC:-0}
echo "enqueue(D2H)=$ENQ  event-sync completions=$EVC"
if [ "$ENQ" -ge 5 ] && [ "$EVC" -ge 5 ]; then
  echo "PASS: completion captured via the CUDA-event path (real vLLM offload pattern)"
else
  echo "FAIL"; exit 1
fi