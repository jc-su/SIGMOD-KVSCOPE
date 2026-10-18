#!/usr/bin/env bash
# REPRODUCE (paper, Llama-3.2-3B): EKV_MODEL=meta-llama/Llama-3.2-3B-Instruct EKV_GPU_BLOCKS=160 \
#   VLLM_ATTENTION_BACKEND=TRITON_ATTN bash tests/_reload_ctrl_run.sh  (grouped reloads 0 across pools 80-160)
# Reload-aware controller. eKV's per-prefix reload attribution says which prefixes pay the reload tax;
# the controller reorders the request stream to group each prefix's requests (so its KV stays resident)
# instead of the cache-blind interleaved order. We compare reload bytes (eKV) and generate wall time
# under a FIXED small pool: baseline (interleaved) vs eKV-guided (grouped). GPU 1 ONLY.
set -u
cd "$(dirname "$0")/.."
PY=/opt/conda/bin/python3
LIBCUDA="${EKV_LIBCUDA:-/usr/lib/x86_64-linux-gnu/libcuda.so.1}"
LIBCUDART=$("$PY" - <<'P'
import glob
for d in ("/usr/local/cuda/targets/x86_64-linux/lib", "/usr/local/cuda/lib64"):
    g = sorted(glob.glob(d + "/libcudart.so.12*"))
    if g:
        print(g[0]); break
P
)
make -s -C observe/ebpf kv_copy kv_alloc || { echo "FAIL build"; exit 1; }

run_order(){  # $1 = interleaved|grouped ; echoes "order reload_MiB wall_s"
  local ord="$1" KV=/tmp/reload_$1.kv ALLOC=/tmp/reload_$1.alloc TOUT=/tmp/reload_$1.out ORC=/tmp/reload_$1.json
  rm -f "$KV" "$ALLOC" "$TOUT" "$ORC"
  EKV_ORDER="$ord" EKV_RELOAD_ORACLE="$ORC" EKV_MODEL="${EKV_MODEL:-facebook/opt-125m}" \
    EKV_GPU_BLOCKS="${EKV_GPU_BLOCKS:-160}" EKV_ALLOC_WAIT=8 CUDA_VISIBLE_DEVICES=1 \
    "$PY" tests/_reload_probe.py >"$TOUT" 2>&1 &
  local PROBE=$! PID=""
  for _ in $(seq 1 60); do PID=$(awk '/EKV_PID/{print $2;exit}' "$TOUT" 2>/dev/null); [ -n "$PID" ] && break
    kill -0 $PROBE 2>/dev/null || { echo "$ord DIED"; return 1; }; sleep 0.3; done
  sudo ./observe/ebpf/kv_alloc "$PID" "$LIBCUDA:cuMemAlloc_v2" -m 1048576 >"$ALLOC" 2>/dev/null &
  for _ in $(seq 1 600); do grep -q EKV_READY "$TOUT" 2>/dev/null && break; kill -0 $PROBE 2>/dev/null || { echo "$ord DIED"; return 1; }; sleep 0.5; done
  sudo EKV_META=1 ./observe/ebpf/kv_copy "$LIBCUDART" "$PID" >"$KV" 2>/dev/null &
  sleep 3
  for _ in $(seq 1 180); do grep -q EKV_ORACLE_DONE "$TOUT" 2>/dev/null && break; kill -0 $PROBE 2>/dev/null || break; sleep 1; done
  sleep 1; sudo pkill -f observe/ebpf/kv_copy 2>/dev/null; sudo pkill -f observe/ebpf/kv_alloc 2>/dev/null
  wait $PROBE 2>/dev/null
  local wall reload
  wall=$(grep -oE "wall=[0-9.]+" "$TOUT" | head -1 | cut -d= -f2)
  reload=$("$PY" tests/_reload_check.py "$KV" "$ALLOC" "$ORC" "${EKV_GPU_BLOCKS:-160}" 2>/dev/null | grep -oE "reload [0-9]+ MiB" | grep -oE "[0-9]+" | head -1)
  echo "RESULT $ord reload=${reload:-NA} wall=${wall:-NA}"
  pkill -f 'tests/_reload_probe' 2>/dev/null; sleep 3
}

echo "=== RELOAD-AWARE CONTROLLER: ${EKV_MODEL:-facebook/opt-125m}, pool=${EKV_GPU_BLOCKS:-160} blocks ==="
run_order interleaved | tee /tmp/reload_ctrl_i.txt
run_order grouped | tee /tmp/reload_ctrl_g.txt
echo "=== compare (grouped = eKV-guided reorder vs interleaved baseline) ==="
"$PY" - <<'PY'
import re
def parse(p):
    s=open(p).read(); m=re.search(r"reload=(\d+) wall=([0-9.]+)",s)
    return (int(m.group(1)), float(m.group(2))) if m else (None, None)
ri,wi=parse("/tmp/reload_ctrl_i.txt"); rg,wg=parse("/tmp/reload_ctrl_g.txt")
if None in (ri,rg,wi,wg): print("parse error", ri,rg,wi,wg)
else:
    print(f"reload bytes: interleaved {ri} MiB -> grouped {rg} MiB  ({(rg-ri)/ri*100:+.0f}%)")
    print(f"generate wall: interleaved {wi:.2f}s -> grouped {wg:.2f}s  ({(wg-wi)/wi*100:+.0f}%)")
    print(f"WIN: eKV-guided reorder cuts reload {(1-rg/ri)*100:.0f}% and wall {(1-wg/wi)*100:.0f}%")
PY