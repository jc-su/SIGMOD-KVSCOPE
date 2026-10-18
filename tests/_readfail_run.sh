#!/usr/bin/env bash
# Exercise the kv_copy batch READ-FAULT mask (do_chunk). A real bpf_probe_read_user fault can't
# be injected safely (the driver faults on the same bogus descriptor array), so EKV_FORCE_READFAIL
# forces the read-fail branch on a normal 5-block cuMemcpyBatchAsync. Expect: 0 BATCH records
# emitted (the chunk's ring buffer reservation is discarded -> no garbage records), all 5 blocks
# audited in stats[ST_BATCH_READFAIL], collector survives. Needs sudo + GPU1.
set -uo pipefail
EKV="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-/opt/conda/bin/python3}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
OUT=/tmp/ekv_rf.jsonl ERR=/tmp/ekv_rf.err

make -s -C "$EKV/observe/ebpf" kv_copy
CU=$($PY -c "import torch; torch.randn(4,device='cuda'); print([l.split()[-1] for l in open('/proc/self/maps') if 'cu13' in l and 'libcudart' in l][0])")
: > "$OUT"; : > "$ERR"
sudo env EKV_FORCE_READFAIL=1 "$EKV/observe/ebpf/kv_copy" "$CU" > "$OUT" 2> "$ERR" &
for _ in $(seq 1 8000); do grep -q "collector running" "$ERR" && break; done
grep -q "collector running" "$ERR" || { echo "FAIL: collector not ready"; cat "$ERR"; sudo pkill -9 -x kv_copy; exit 1; }

PYTHONPATH="$EKV" "$PY" "$EKV/tests/_driver_offload_workload.py" >/dev/null 2>&1
alive=$(pgrep -x kv_copy >/dev/null && echo YES || echo NO)
sudo pkill -TERM -x kv_copy 2>/dev/null; sleep 1

batch=$(grep -c '"kind":"BATCH"' "$OUT" || true)
rf=$(grep -oE "batch read faults \(chunk skipped\): [0-9]+" "$ERR" | grep -oE "[0-9]+$" || echo "?")
echo "collector survived=$alive ; BATCH records emitted=$batch ; read-faults audited=$rf"
[ "$alive" = YES ] && [ "$batch" = 0 ] && [ "$rf" = 5 ] \
  && echo "PASS: read-fault mask skips the chunk (0 garbage records), audits all 5 blocks, survives" \
  || { echo "FAIL: expected survived=YES batch=0 rf=5"; exit 1; }
