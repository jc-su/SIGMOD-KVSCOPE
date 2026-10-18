#!/usr/bin/env bash
# Sustained PRODUCTION soak: the SUPERVISED collector (scripts/run_collector.sh -- auto-restart on
# death + a gap marker into the stream) under a long paced offload workload. Validates the eBPF data
# plane is lossless + stable over a sustained run AND that a restart (if any) is FLAGGED as a capture
# gap, not silent. This is the production deployment config, not a bare collector. Needs sudo + GPU1.
set -uo pipefail
EKV="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-/opt/conda/bin/python3}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
OUT=/tmp/ekv_copy.jsonl ERR=/tmp/ekv_sustain.err

make -s -C "$EKV/observe/ebpf" kv_copy
CU=$($PY -c "import torch; torch.randn(4,device='cuda'); print([l.split()[-1] for l in open('/proc/self/maps') if 'cu13' in l and 'libcudart' in l][0])")
: > "$OUT"; : > "$ERR"
sudo -E env EKV_DRIVER_ONLY=1 "$EKV/scripts/run_collector.sh" "$CU" > "$OUT" 2> "$ERR" &
for _ in $(seq 1 8000); do grep -q "collector running" "$ERR" && break; done
grep -q "collector running" "$ERR" || { echo "FAIL: collector not ready"; cat "$ERR"; \
  sudo pkill -9 -f run_collector.sh; sudo pkill -9 -x kv_copy; exit 1; }
echo "supervised collector up; sustained paced workload (EKV_CYCLES=${EKV_CYCLES:-100} EKV_PACED=${EKV_PACED:-4})"

PYTHONPATH="$EKV" EKV_PACED="${EKV_PACED:-4}" EKV_CYCLES="${EKV_CYCLES:-100}" EKV_GPU_BLOCKS="${EKV_GPU_BLOCKS:-96}" \
  "$PY" "$EKV/tests/_diag_check.py" 2>&1 | grep -iE "captured copies|H2D loads|OK:|OK\(#3\)|NO LOADS|FAIL" | head -6
sudo pkill -TERM -f run_collector.sh 2>/dev/null; sleep 4; sudo pkill -9 -x kv_copy 2>/dev/null; true

echo "=== collector loss report (sustained) ==="
grep -iE "batch truncation|read faults|ringbuf drops" "$ERR" | tail -4
echo "=== supervisor restarts (death mitigation) ==="
grep -c "CAPTURE GAP" "$ERR" | sed 's/^/  restarts=/'
echo "=== capture completeness over the run ==="
$PY -c "import sys; sys.path.insert(0,'$EKV'); from observe.ebpf_reader import read_copies_file_report as r; \
rep=r('$OUT'); print(f'  copies={len(rep.copies)} gaps={rep.gaps} complete={rep.complete} fully_costed={rep.fully_costed}')"
