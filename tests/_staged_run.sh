#!/usr/bin/env bash
# Validate the STAGED KV fetch path (storage -> DRAM -> GPU). The workload reads a 4 MiB KV file into
# pinned DRAM then H2Ds it to GPU1; storage_read captures the read() (storage->DRAM) and kv_copy
# captures the H2D (DRAM->GPU), both SCOPED to the workload pid. PASS = both legs captured AND on the
# SAME tid -- the join key the analysis uses to fuse them into the full tier_read+H2D miss cost.
set -u
cd "$(dirname "$0")/.."
EKV="$(pwd)"
NVCC="${NVCC:-/opt/conda/bin/nvcc}"
LIBCUDART="$(ldconfig -p | grep 'libcudart.so' | head -1 | awk '{print $NF}')"
[ -z "$LIBCUDART" ] && LIBCUDART=/usr/local/cuda-12.9/targets/x86_64-linux/lib/libcudart.so
SR_OUT=/tmp/ekv_staged_sr.out KV_OUT=/tmp/ekv_staged_kv.out TOUT=/tmp/ekv_staged.test

make -s -C "$EKV/observe/ebpf" storage_read kv_copy 2>/dev/null || { echo "FAIL build collectors"; exit 1; }
"$NVCC" -o /tmp/ekv_staged "$EKV/tests/_staged_read.cu" 2>/dev/null || { echo "FAIL build workload (need nvcc)"; exit 1; }

CUDA_VISIBLE_DEVICES=1 LD_LIBRARY_PATH="$(dirname "$LIBCUDART")" /tmp/ekv_staged >"$TOUT" 2>&1 &
TESTPID=$!
PID=""
for i in $(seq 1 40); do PID=$(awk '/EKV_TEST_PID/{print $2; exit}' "$TOUT" 2>/dev/null); [ -n "$PID" ] && break; sleep 0.1; done
[ -z "$PID" ] && { echo "FAIL: workload did not announce pid"; cat "$TOUT"; kill $TESTPID 2>/dev/null; exit 1; }

sudo "$EKV/observe/ebpf/storage_read" "$PID" -m 1048576 >"$SR_OUT" 2>/dev/null &
sudo "$EKV/observe/ebpf/kv_copy" "$LIBCUDART" "$PID" >"$KV_OUT" 2>/dev/null &
sleep 2

wait $TESTPID
grep -E "read|H2D|done" "$TOUT"
sleep 1; sudo pkill -f observe/ebpf/storage_read 2>/dev/null; sudo pkill -f observe/ebpf/kv_copy 2>/dev/null; sleep 1

echo "=== storage_read (storage->DRAM leg) ==="; grep '"bytes":4194304' "$SR_OUT" | head -2
echo "=== kv_copy H2D (DRAM->GPU leg) ===";       grep '"kind":"H2D"' "$KV_OUT" | grep '"bytes":4194304' | head -2
sr_tid=$(grep '"bytes":4194304' "$SR_OUT" | head -1 | grep -oE '"tid":[0-9]+' | head -1 | cut -d: -f2)
kv_tid=$(grep '"kind":"H2D"' "$KV_OUT" | grep '"bytes":4194304' | head -1 | grep -oE '"tid":[0-9]+' | head -1 | cut -d: -f2)
echo "storage_read tid=$sr_tid   kv_copy H2D tid=$kv_tid"
rm -f /tmp/ekv_staged /tmp/ekv_staged_kv.bin
[ -n "$sr_tid" ] && [ "$sr_tid" = "$kv_tid" ] \
  && echo "PASS: staged path observed -- read(storage->DRAM) join H2D(DRAM->GPU) on the same tid ($sr_tid), fusable into tier_read+H2D" \
  || echo "FAIL: both legs not captured on a shared tid (sr=$sr_tid kv=$kv_tid)"
