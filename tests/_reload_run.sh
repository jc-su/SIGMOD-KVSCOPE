#!/usr/bin/env bash
# Reload-aware use case. Offload workload (small GPU pool -> evict to host -> reload on re-access).
# kv_alloc captures the KV-pool device allocations (attached BEFORE the pool cudaMalloc); kv_copy
# captures the batched KV offload/reload copies + the index_lookup metadata. The check attributes the
# per-request reload bytes. GPU 1 ONLY.
set -u
cd "$(dirname "$0")/.."
PY=/opt/conda/bin/python3
KV=/tmp/reload.kv; ALLOC=/tmp/reload.alloc; TOUT=/tmp/reload.out
LIBCUDART=$("$PY" - <<'P'
import glob
for d in ("/usr/local/cuda/targets/x86_64-linux/lib", "/usr/local/cuda/lib64"):
    g = sorted(glob.glob(d + "/libcudart.so.12*"))
    if g:
        print(g[0]); break
P
)
make -s -C observe/ebpf kv_copy kv_alloc || { echo "FAIL build"; exit 1; }
rm -f "$KV" "$ALLOC" "$TOUT"
EKV_MODEL="${EKV_MODEL:-facebook/opt-125m}" EKV_GPU_BLOCKS="${EKV_GPU_BLOCKS:-160}" EKV_ALLOC_WAIT=8 \
  CUDA_VISIBLE_DEVICES=1 "$PY" tests/_reload_probe.py >"$TOUT" 2>&1 &
PROBE=$!
PID=""
for _ in $(seq 1 60); do
  PID=$(awk '/EKV_PID/{print $2; exit}' "$TOUT" 2>/dev/null)
  [ -n "$PID" ] && break
  kill -0 "$PROBE" 2>/dev/null || { echo "FAIL: probe died early"; tail -15 "$TOUT"; exit 1; }
  sleep 0.3
done
[ -z "$PID" ] && { echo "FAIL: no PID"; exit 1; }
echo "probe PID $PID; attaching kv_alloc (before the KV-pool cudaMalloc)"
LIBCUDA="${EKV_LIBCUDA:-/usr/lib/x86_64-linux-gnu/libcuda.so.1}"
sudo ./observe/ebpf/kv_alloc "$PID" "$LIBCUDA:cuMemAlloc_v2" -m 1048576 >"$ALLOC" 2>/dev/null &
for _ in $(seq 1 600); do grep -q EKV_READY "$TOUT" 2>/dev/null && break; kill -0 "$PROBE" 2>/dev/null || { echo "FAIL: died"; tail -15 "$TOUT"; exit 1; }; sleep 0.5; done
echo "ready; attaching kv_copy EKV_META"
sudo EKV_META=1 ./observe/ebpf/kv_copy "$LIBCUDART" "$PID" >"$KV" 2>/dev/null &
sleep 3
for _ in $(seq 1 180); do grep -q EKV_GEN_DONE "$TOUT" 2>/dev/null && break; kill -0 "$PROBE" 2>/dev/null || break; sleep 1; done
sleep 1
sudo pkill -f "observe/ebpf/kv_copy" 2>/dev/null; sudo pkill -f "observe/ebpf/kv_alloc" 2>/dev/null
wait "$PROBE" 2>/dev/null
grep -E "EKV_GEN_DONE" "$TOUT"
echo "kv_alloc device allocations captured: $(grep -c addr "$ALLOC" 2>/dev/null)"
echo "=== per-request reload attribution ==="
"$PY" tests/_reload_check.py "$KV" "$ALLOC" /tmp/reload_oracle.json "${EKV_GPU_BLOCKS:-160}"
