#!/usr/bin/env bash
# REAL multi-tier (LMCache host-DRAM + local-disk) reload-cost measurement with the eBPF kv_copy
# collector. Needs sudo (uprobe) + GPU. Writes the LMCache config + resets the disk tier.
set -uo pipefail
EKV="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-/opt/conda/bin/python3}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
OUT=/tmp/ekv_copy.jsonl ERR=/tmp/ekv_mt.err
READS=/tmp/ekv_reads.jsonl RERR=/tmp/ekv_reads.err

rm -rf /tmp/lmcache_disk; mkdir -p /tmp/lmcache_disk
cat > /tmp/lmcache_multitier.yaml <<'YAML'
chunk_size: 256
local_cpu: True
max_local_cpu_size: 0.2
local_disk: "file:///tmp/lmcache_disk/"
max_local_disk_size: 4.0
YAML

make -s -C "$EKV/observe/ebpf" kv_copy storage_read
CU=$($PY -c "import torch; torch.randn(4,device='cuda'); print([l.split()[-1] for l in open('/proc/self/maps') if 'cu13' in l and 'libcudart' in l][0])")
echo "libcudart : $CU"
: > "$OUT"; : > "$ERR"; : > "$READS"; : > "$RERR"
sudo env EKV_DRIVER_ONLY=1 "$EKV/observe/ebpf/kv_copy" "$CU" > "$OUT" 2> "$ERR" &
# storage_read: syscall tracepoints (read/pread64), all PIDs, >=1MiB (KV chunks); analysis
# filters to the workload pid + the reload window. This makes the disk-tier read measurable.
sudo "$EKV/observe/ebpf/storage_read" 0 -m 1048576 > "$READS" 2> "$RERR" &
for _ in $(seq 1 6000); do grep -q "collector running" "$ERR" && break; done
grep -q "collector running" "$ERR" || { echo "FAIL: collector not ready"; cat "$ERR"; sudo pkill -f "$EKV/observe/ebpf/kv_copy"; sudo pkill -f "$EKV/observe/ebpf/storage_read"; exit 1; }
for _ in $(seq 1 6000); do grep -q "storage_read:" "$RERR" && break; done
grep -q "storage_read:" "$RERR" || echo "WARN: storage_read not ready (tier-read measurement will be empty); $(cat "$RERR")"
echo "collectors running; starting LMCache multi-tier workload"

EKV_COPY_FILE="$OUT" EKV_READS_FILE="$READS" PYTHONPATH="$EKV" "$PY" "$EKV/tests/_multitier_check.py"
RC=$?
sudo pkill -f "$EKV/observe/ebpf/kv_copy" 2>/dev/null
sudo pkill -f "$EKV/observe/ebpf/storage_read" 2>/dev/null
sleep 2
echo "storage_read $(grep -E '^ringbuf drops:' "$RERR" || echo 'ringbuf drops: MISSING (collector did not report)')"
exit $RC
