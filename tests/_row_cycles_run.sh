#!/usr/bin/env bash
# Per-request EXACT cycles gate (Phase-1 spike). A chunked-prefill-shaped kernel (QB q-block CTAs per
# sequence x NH heads) with planted per-request context: the pass timer must accumulate every CTA of a
# sequence into that SEQUENCE's row (per-request, not per-CTA), cycles must be linear in the planted
# work, and the capture+timer must join on the same row. GPU 1 ONLY (GPU 0 is a co-tenant).
set -u
cd "$(dirname "$0")/.."
EKV="$(pwd)"
SO="$EKV/build/libeKVPlugin.so"
LIBSTDCPP=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
TR=/path/to/triton/python
OUT=/tmp/ekv_row_cycles.log

[ -f "$SO" ] || { echo "FAIL: build the pass first (bash scripts/build_plugin.sh)"; exit 1; }
TRITON_CACHE_DIR=$(mktemp -d /tmp/ekv_rowcyc_cache.XXXXXX) EKV_TIMER=1 EKV_DEBUG=1 \
  TRITON_EXT_ENABLED=1 TRITON_PLUGIN_PATHS="$SO" EKV_ENABLE=1 EKV_PASS_AUTO=1 \
  PYTHONPATH="$TR:$EKV" LD_PRELOAD="$LIBSTDCPP" CUDA_VISIBLE_DEVICES=1 \
  timeout 240 python "$EKV/tests/_row_cycles_check.py" >"$OUT" 2>&1
rc=$?
grep -E "timer rows lit|cycles vs planted|row0|join|PASS|FAIL" "$OUT" | tail -8
exit $rc
