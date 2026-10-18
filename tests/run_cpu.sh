#!/usr/bin/env bash
# CPU-pure test gate: every test that needs no GPU/sudo/serve, run with CUDA hidden so a stray
# device init can't mask a real failure. The GPU/live harnesses (test_blocktable_capture,
# test_timer_kernel, and the _*_run.sh scripts) are EXCLUDED -- run those on a GPU box.
# Exits non-zero if any test fails (CI gate). Override the interpreter with PY=... (needs the
# project's deps: observe/ + grpcio + opentelemetry for the OTel round-trip check).
set -u
cd "$(dirname "$0")/.."
PY="${PY:-/opt/conda/bin/python3}"

# all test_*.py except the two that require a real GPU, plus the pure underscore checks
TESTS=""
for f in tests/test_*.py; do
  t=$(basename "$f")
  case "$t" in
    test_blocktable_capture.py|test_timer_kernel.py|test_sglang_view.py) continue ;;  # need a GPU
  esac
  TESTS="$TESTS $t"
done
TESTS="$TESTS _binary_transport_check.py"

# the eval harness is not imported by any test: a syntax error there broke every RQ runner
# once without this gate noticing. Compile it (and observe/) before counting anything.
if ! CUDA_VISIBLE_DEVICES="" "$PY" -m compileall -q observe runtime ekv_vllm_plugin.py >/tmp/ekv_compileall.log 2>&1; then
  echo "  FAIL  compileall (observe/ runtime/)"; cat /tmp/ekv_compileall.log | tail -5
  echo "---- CPU suite: 0 passed, 1 failed ----"; exit 1
fi
echo "  PASS  compileall (observe/ runtime/ plugin)"

pass=0; fail=0; failed=""
PYTEST_FILES=""
for t in $TESTS; do
  # pytest-style files (def test_* with no __main__ runner) do NOTHING when executed as a
  # script: they would count as vacuous passes. Route them through pytest and count real tests.
  if grep -q "def test_" "tests/$t" && ! grep -qE "__main__|sys.exit\(" "tests/$t"; then
    PYTEST_FILES="$PYTEST_FILES tests/$t"
    continue
  fi
  if CUDA_VISIBLE_DEVICES="" PYTHONPATH="$(pwd)" timeout 120 "$PY" "tests/$t" >"/tmp/ekv_test_$t.log" 2>&1; then
    printf "  PASS  %s\n" "$t"; pass=$((pass + 1))
  else
    printf "  FAIL  %s\n" "$t"; fail=$((fail + 1)); failed="$failed $t"
    sed 's/^/        /' <"/tmp/ekv_test_$t.log" | tail -4
  fi
done
if [ -n "$PYTEST_FILES" ]; then
  if CUDA_VISIBLE_DEVICES="" PYTHONPATH="$(pwd)" timeout 300 "$PY" -m pytest -q $PYTEST_FILES \
      >/tmp/ekv_test_pytest.log 2>&1; then
    n=$(grep -oE "[0-9]+ passed" /tmp/ekv_test_pytest.log | grep -oE "[0-9]+" | head -1)
    printf "  PASS  pytest suite (%s tests: %s)\n" "${n:-?}" "$(echo $PYTEST_FILES | tr ' ' ',' )"
    pass=$((pass + ${n:-0}))
  else
    printf "  FAIL  pytest suite\n"; fail=$((fail + 1)); failed="$failed pytest"
    tail -6 /tmp/ekv_test_pytest.log | sed 's/^/        /'
  fi
fi
echo "---- CPU suite: $pass passed, $fail failed ----"
[ "$fail" -eq 0 ] || { echo "FAILED:$failed"; exit 1; }
