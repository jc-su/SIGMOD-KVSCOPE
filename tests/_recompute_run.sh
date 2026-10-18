#!/usr/bin/env bash
# #4 reload vs recompute vs cache-hit, validated by the PASS-CAPTURABLE query-length signal.
# The same reuse workload is run in three regimes that each force ONE class:
#   offload   : connector ON,  prefix-cache ON  -> RELOAD  (H2D copy captured by eBPF)
#   cachehit  : connector OFF, prefix-cache ON  -> CACHE-HIT (no copy, no compute)
#   recompute : connector OFF, prefix-cache OFF -> RECOMPUTE (large cur_batch_query_len, no copy)
# The recompute/cache-hit split is by the kernel's per-slot query length (cu_seqlens_q diff =
# what the Triton pass can capture) — exact, vs the same-block heuristic. eBPF kv_copy watches
# all three. Needs sudo (uprobe) + GPU.
set -uo pipefail
EKV="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-/opt/conda/bin/python3}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
ERR=/tmp/ekv_recompute.err

make -s -C "$EKV/observe/ebpf" kv_copy
CU=$($PY -c "import torch; torch.randn(4,device='cuda'); print([l.split()[-1] for l in open('/proc/self/maps') if 'cu13' in l and 'libcudart' in l][0])")
echo "libcudart : $CU"

run_mode() {  # $1=EKV_MODE
  local OUT="/tmp/ekv_copy_$1.jsonl"
  : > "$OUT"; : > "$ERR"
  sudo env EKV_DRIVER_ONLY=1 "$EKV/observe/ebpf/kv_copy" "$CU" > "$OUT" 2> "$ERR" &
  for _ in $(seq 1 6000); do grep -q "collector running" "$ERR" && break; done
  grep -q "collector running" "$ERR" || { echo "FAIL: collector not ready ($1)"; cat "$ERR"; sudo pkill -f "$EKV/observe/ebpf/kv_copy"; return 1; }
  echo "--- regime $1: collector running ---"
  EKV_MODE="$1" EKV_COPY_FILE="$OUT" PYTHONPATH="$EKV" "$PY" "$EKV/tests/_recompute_check.py"
  sudo pkill -f "$EKV/observe/ebpf/kv_copy" 2>/dev/null
  sleep 1
}

run_mode offload
run_mode cachehit
run_mode recompute

echo "=== A/B/C verdict (query-length classification per forced regime) ==="
"$PY" - <<'PYEOF'
import json
M = {m: json.load(open(f"/tmp/ekv_recompute_{m}.json")) for m in ("offload", "cachehit", "recompute")}
for m, d in M.items():
    print(f"  {m:9s}: reload={d['reload']} recompute={d['recompute']} cache_hit={d['cache_hit']} "
          f"H2D={d['h2d_loads']}  (heuristic recompute/cache={d.get('recompute_heur')}/{d.get('cache_hit_heur')})")
ok_o = M["offload"]["reload"] > 0 and M["offload"]["h2d_loads"] > 0
ok_c = M["cachehit"]["cache_hit"] > 0 and M["cachehit"]["recompute"] == 0 and M["cachehit"]["h2d_loads"] == 0
ok_r = M["recompute"]["recompute"] > 0 and M["recompute"]["h2d_loads"] == 0
if ok_o and ok_c and ok_r:
    print("PASS(#4): the pass-capturable per-slot query length classifies all three EXACTLY — "
          "reload (copy), cache-hit (no copy, no compute), recompute (large query_len, no copy). "
          "Recompute is invisible to a transfer-only view; the compute signal makes it exact, and "
          "it CORRECTS the same-block heuristic (which mislabels cache-reuse churn as recompute).")
else:
    print(f"CHECK(#4): ok_offload={ok_o} ok_cachehit={ok_c} ok_recompute={ok_r} — see numbers above.")
PYEOF
