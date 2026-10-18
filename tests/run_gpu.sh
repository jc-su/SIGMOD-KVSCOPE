#!/usr/bin/env bash
# The GPU gate suite, one entry point .
# Each gate is independently runnable; this
# just sequences them and summarizes. CPU-only invariants: tests/run_cpu.sh.
set -u
cd "$(dirname "$0")/.."
declare -a RESULTS

gate() {   # $1 = name, $2 = script
  echo; echo "════ $1 ════"
  if "$2" >/tmp/ekv_gate_last.log 2>&1; then
    RESULTS+=("ok    $1")
    tail -2 /tmp/ekv_gate_last.log | sed 's/^/  /'
  else
    RESULTS+=("FAIL  $1  (log: /tmp/ekv_gate_last.log -> /tmp/ekv_gate_$1.log)")
    cp /tmp/ekv_gate_last.log "/tmp/ekv_gate_$1.log"
    tail -5 /tmp/ekv_gate_last.log | sed 's/^/  /'
  fi
}

# the regression ladder: observation -> attribution -> statistic
gate attn_share     tests/_attn_share_live_run.sh    # tile log-sum-exp equals the true attention share
gate row_cycles     tests/_row_cycles_run.sh         # per-request device timer, controlled
gate cycles_live    tests/_cycles_live_run.sh        # cycles on real vLLM, 3 phases
gate isolation      tests/_isolation_run.sh          # no cross-request private-block reads

echo; echo "════ summary ════"
printf '%s\n' "${RESULTS[@]}"
[[ " ${RESULTS[*]} " != *FAIL* ]]
