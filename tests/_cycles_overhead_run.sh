#!/usr/bin/env bash
# Pass+timer overhead A/B: identical decode-heavy generation, stock Triton (OFF) vs the pass with
# capture + per-request timer (ON), interleaved OFF/ON x3 so drift hits both arms. The instrumented
# cost is per CTA: two clock64 reads + one posted red.add to host memory at exit, plus the masked
# capture row store -- expected within run noise; a tiny model is the CONSERVATIVE case (least GPU
# work to hide behind). Reports medians; the gate is completion + capture present, the numbers speak.
# GPU 1 ONLY (GPU 0 is a co-tenant).
set -u
cd "$(dirname "$0")/.."
EKV="$(pwd)"
SO="$EKV/build/libeKVPlugin.so"
LIBSTDCPP=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
TR=/path/to/triton/python
OUT=/tmp/ekv_cycles_overhead.log
: >"$OUT"

[ -f "$SO" ] || { echo "FAIL: build the pass first (bash scripts/build_plugin.sh)"; exit 1; }
export EKV_MODEL="${EKV_MODEL:-facebook/opt-125m}"
export EKV_GPU_UTIL="${EKV_GPU_UTIL:-0.35}"
echo "model=$EKV_MODEL util=$EKV_GPU_UTIL"
for i in 1 2 3; do
  for MODE in off on; do
    CACHE=$(mktemp -d /tmp/ekv_ovh_cache.XXXXXX)
    if [ "$MODE" = off ]; then
      EKV_LABEL="off$i" TRITON_CACHE_DIR="$CACHE" \
        PYTHONPATH="$TR:$EKV" LD_PRELOAD="$LIBSTDCPP" CUDA_VISIBLE_DEVICES=1 \
        timeout 600 python "$EKV/tests/_cycles_overhead_probe.py" >>"$OUT" 2>&1
    else
      EKV_LABEL="on$i" TRITON_CACHE_DIR="$CACHE" \
        EKV_TIMER=1 EKV_NCAP=128 EKV_MAX_ROWS=1024 \
        TRITON_EXT_ENABLED=1 TRITON_PLUGIN_PATHS="$SO" EKV_ENABLE=1 EKV_PASS_AUTO=1 \
        PYTHONPATH="$TR:$EKV" LD_PRELOAD="$LIBSTDCPP" CUDA_VISIBLE_DEVICES=1 \
        timeout 600 python "$EKV/tests/_cycles_overhead_probe.py" >>"$OUT" 2>&1
    fi
    rm -rf "$CACHE"
    sleep 5   # let the previous process's GPU teardown settle (vLLM's memory profiling asserts if
              # free memory shifts mid-profile on a shared box)
  done
done
grep EKV_RESULT "$OUT"
python3 - "$OUT" <<'P'
import re, statistics, sys
rows = {}
for m in re.finditer(r"label=(\w+?)(\d) wall_s=([\d.]+) tokens=(\d+) toks_per_s=([\d.]+)",
                     open(sys.argv[1]).read()):
    rows.setdefault(m.group(1), []).append(float(m.group(5)))
if set(rows) != {"off", "on"} or any(len(v) < 2 for v in rows.values()):
    print("FAIL: too few completed runs per arm", rows); sys.exit(1)
off, on = statistics.median(rows["off"]), statistics.median(rows["on"])
note = "" if all(len(v) == 3 for v in rows.values()) else \
    f"  (UNEVEN: {len(rows['off'])} off / {len(rows['on'])} on runs completed on the shared box)"
print(f"median toks/s: off={off:.1f} (runs {sorted(rows['off'])})  "
      f"on={on:.1f} (runs {sorted(rows['on'])}){note}")
print(f"pass+timer overhead: {100*(off-on)/off:+.1f}% of baseline throughput "
      f"(conservative tiny-model case; medians over completed runs)")
print("PASS: A/B complete; overhead reported above")
P
