#!/usr/bin/env bash
# Validate the event-query READY-BRACKET (the precise GPU-ready instant, zero-touch). A 256MiB H2D
# (fetch) is still running when tests/_event_ready.cu starts polling cudaEventQuery, so the last
# not-ready poll and the first success bracket GPU-done to the poll gap. kv_copy is attached SCOPED
# to the workload pid (no system-wide libcudart uprobe -> no GPU0 co-tenant overhead).
# PASS = a completion via=event-query with ready_lo_ns>0 AND ready_window_ns>0 (a real bracket).
set -u
cd "$(dirname "$0")/.."
EKV="$(pwd)"
NVCC="${NVCC:-/opt/conda/bin/nvcc}"
LIBCUDART="$(ldconfig -p | grep 'libcudart.so' | head -1 | awk '{print $NF}')"
[ -z "$LIBCUDART" ] && LIBCUDART=/usr/local/cuda-12.9/targets/x86_64-linux/lib/libcudart.so
OUT=/tmp/ekv_evtready.out ERR=/tmp/ekv_evtready.err TOUT=/tmp/ekv_evtready.test

make -s -C "$EKV/observe/ebpf" kv_copy 2>/dev/null || { echo "FAIL build kv_copy"; exit 1; }
"$NVCC" -o /tmp/ekv_evtready "$EKV/tests/_event_ready.cu" 2>/dev/null || { echo "FAIL build test (need nvcc)"; exit 1; }

# launch the workload: it loads libcudart, prints its pid, then sleeps 4s for us to attach
CUDA_VISIBLE_DEVICES=1 LD_LIBRARY_PATH="$(dirname "$LIBCUDART")" /tmp/ekv_evtready >"$TOUT" 2>&1 &
TESTPID=$!
PID=""
for i in $(seq 1 40); do PID=$(awk '/EKV_TEST_PID/{print $2; exit}' "$TOUT" 2>/dev/null); [ -n "$PID" ] && break; sleep 0.1; done
[ -z "$PID" ] && { echo "FAIL: workload did not announce its pid"; cat "$TOUT"; kill $TESTPID 2>/dev/null; exit 1; }

sudo "$EKV/observe/ebpf/kv_copy" "$LIBCUDART" "$PID" >"$OUT" 2>"$ERR" &
sleep 2
grep -q "cudaEventQuery \[ret\]" "$ERR" || { echo "FAIL: uret_event_query not attached"; grep -i query "$ERR" | head; sudo pkill -f observe/ebpf/kv_copy; kill $TESTPID 2>/dev/null; exit 1; }

wait $TESTPID
grep -E "iter|done" "$TOUT"
sleep 1; sudo pkill -f observe/ebpf/kv_copy 2>/dev/null; sleep 1

echo "=== event-query completions (pid=$PID) ==="
grep "\"via\":\"event-query\"" "$OUT" | grep "\"pid\":$PID" | head -3
ok=$(grep "\"via\":\"event-query\"" "$OUT" | grep "\"pid\":$PID" | grep -cE "\"ready_window_ns\":[1-9]")
echo "event-query completions with a ready bracket (ready_window_ns>0): $ok"
rm -f /tmp/ekv_evtready
[ "$ok" -ge 1 ] && echo "PASS: ready-bracket recovered -- GPU-done bracketed to the poll gap, zero-touch" \
                || echo "FAIL: no bracketed completion"