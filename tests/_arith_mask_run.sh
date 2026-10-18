#!/usr/bin/env bash
# The no-silent-fallback check: forcing the arithmetic-index path (EKV_FORCE_ARITH, test-only) onto an
# INDEX-based (paged) kernel must REJECT it. The kernel has a vector-integer index load, so
# findArithmeticIndex records sawIndexGather and the pass SKIPS LOUDLY ("detection gap; NOT silently
# re-bound as arithmetic"), publishing NO row_form=arith. PASS = the loud skip fires AND no arith
# capture is published -- proving a findIndexLookup false-negative can never be mis-bound as a formula.
set -u
cd "$(dirname "$0")/.."
EKV="$(pwd)"
SO="$EKV/build/libeKVPlugin.so"
LIBSTDCPP=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
TR=/path/to/triton/python
OUT=/tmp/ekv_arith_mask.log

[ -f "$SO" ] || { echo "FAIL: build the pass first (cmake --build build)"; exit 1; }
TRITON_CACHE_DIR=$(mktemp -d /tmp/ekv_mask_cache.XXXXXX) EKV_FORCE_ARITH=1 EKV_DEBUG=1 \
  TRITON_EXT_ENABLED=1 TRITON_PLUGIN_PATHS="$SO" EKV_ENABLE=1 EKV_PASS_AUTO=1 \
  PYTHONPATH="$TR" LD_PRELOAD="$LIBSTDCPP" CUDA_VISIBLE_DEVICES=1 \
  timeout 200 python "$EKV/tests/_pass_autoconfig_check.py" >"$OUT" 2>&1

mask=$(grep -c "NOT silently re-bound as arithmetic" "$OUT")
arith=$(grep -cE "self-alloc ARITH|row_form.:.arith" "$OUT")
echo "mask-fired=$mask  arith-published=$arith"
grep -E "NOT silently re-bound|self-alloc ARITH" "$OUT" | head -2
[ "$mask" -ge 1 ] && [ "$arith" -eq 0 ] \
  && echo "PASS: the arithmetic path REJECTED an index-based kernel (loud skip, no mis-bind) -- no silent fallback" \
  || echo "FAIL: mask did not fire / arith wrongly published (mask=$mask arith=$arith)"
