#!/usr/bin/env bash
# End-to-end serving overhead of the full eKV collector (data-plane copy capture + metadata
# capture), as interleaved off-vs-on pairs so co-tenant drift hits both halves of a pair.
# Reports paired overhead + 95% CI AND the off-run noise floor (CV). On a SHARED box the
# overhead is typically below the noise floor; this harness is built to give a tight number on
# ISOLATED hardware: EKV_LOCK_CLOCKS="<minMHz,maxMHz>" pins the GPU clock, and the co-tenant
# check below warns when the floor will be high.
set -u
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
LIBCUDART=/opt/conda/lib/python3.12/site-packages/torch/../nvidia/cuda_runtime/lib/libcudart.so.12
K=${EKV_PAIRS:-3}
OUT=/tmp/ekv_e2e.txt; : > "$OUT"

echo "=== isolated-HW readiness check ==="
echo "co-tenant compute processes on this GPU (each inflates the noise floor):"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null | head
if [ -n "${EKV_LOCK_CLOCKS:-}" ]; then
  sudo nvidia-smi -pm 1 >/dev/null 2>&1
  sudo nvidia-smi -lgc "$EKV_LOCK_CLOCKS" >/dev/null 2>&1 && echo "GPU clock locked to $EKV_LOCK_CLOCKS MHz" \
    || echo "clock lock failed (need permission / supported GPU)"
else
  echo "clocks NOT locked (set EKV_LOCK_CLOCKS=minMHz,maxMHz on isolated HW for a tight number)"
fi
echo

tps() {  # one fresh-process run; retries the shared-GPU init race
  for a in 1 2 3; do
    v=$(timeout 400 python tests/_serving_workload.py 2>/dev/null | grep -oE 'TPS=[0-9.]+' | cut -d= -f2)
    [ -n "$v" ] && { echo "$v"; return; }
  done
  echo ""
}

sudo pkill -9 -x kv_copy 2>/dev/null; sleep 1
echo "pair  off_tps  on_tps  overhead_%"
for k in $(seq 1 "$K"); do
  off=$(tps)
  sudo EKV_DRIVER_ONLY=1 EKV_META=1 ./observe/ebpf/kv_copy "$LIBCUDART" >/dev/null 2>/tmp/oe.err &
  COL=$!; sleep 4.0
  grep -qi "load failed" /tmp/oe.err && echo "  (collector load failed pair $k)"
  on=$(tps)
  sudo kill "$COL" 2>/dev/null; sleep 1; sudo pkill -9 -x kv_copy 2>/dev/null
  [ -z "$off" ] || [ -z "$on" ] && { echo "  pair $k: a run failed, skipping"; continue; }
  ov=$(python3 -c "print(f'{($off-$on)/$off*100:.2f}')")
  printf "%4d  %7s  %6s  %9s\n" "$k" "$off" "$on" "$ov"
  echo "$off $on" >> "$OUT"
done

[ -n "${EKV_LOCK_CLOCKS:-}" ] && sudo nvidia-smi -rgc >/dev/null 2>&1
echo ""; echo "=== end-to-end serving overhead (full eKV: data-plane + metadata capture) ==="
python3 - "$OUT" <<'PY'
import sys, statistics as st
pairs = [tuple(map(float, l.split())) for l in open(sys.argv[1]) if l.split()]
if len(pairs) < 1:
    print("no completed pairs"); sys.exit()
off = [o for o, _ in pairs]; on = [n for _, n in pairs]
ov = [(o - n) / o * 100 for o, n in pairs]
n = len(ov); mean = st.mean(ov); sd = st.pstdev(ov) if n > 1 else 0.0
ci = 1.96 * sd / (n ** 0.5) if n > 1 else float('nan')
floor = st.pstdev(off) / st.mean(off) * 100 if n > 1 else float('nan')   # off-run CV = noise floor
print(f"pairs={n}  off-tps mean={st.mean(off):.1f}  on-tps mean={st.mean(on):.1f}")
print(f"overhead = {mean:.2f}%   (95% CI +/-{ci:.2f})")
print(f"off-run noise floor (CV) = {floor:.2f}%")
if n > 1 and abs(mean) <= floor:
    print(f"=> overhead is BELOW the {floor:.1f}% noise floor of this (shared) testbed: an upper "
          f"bound, not a tight number. Re-run on isolated HW with EKV_LOCK_CLOCKS for the clean value.")
else:
    print("=> overhead exceeds the noise floor; reportable as measured.")
PY
