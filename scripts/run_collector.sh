#!/usr/bin/env bash
# Supervise the kv_copy collector: if it dies UNEXPECTEDLY, log a timestamped capture-gap marker and
# restart it (bounded). MITIGATION, not a root cause -- one unexplained loader death was seen once
# under heavy system stress and never reproduced (a 1.78M-record microbench + two clean soaks all
# survived); supervision keeps the observation plane resilient to it and to any future crash, and
# makes the gap auditable (never a silent stop). SIGINT/SIGTERM -> clean stop (kills the collector).
#
# Run as ROOT (the collector's BPF load needs it); running the supervisor itself as root keeps
# $kid == the collector's own pid so signals are direct (an inner `sudo` would not forward cleanly).
#   sudo -E EKV_DRIVER_ONLY=1 scripts/run_collector.sh <libcudart.so> [pid]
set -u
EKV="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$EKV/observe/ebpf/kv_copy"
MAX_RESTARTS="${EKV_MAX_RESTARTS:-50}"     # give up after this many (a crash loop, not a blip)
BACKOFF="${EKV_RESTART_BACKOFF:-1}"        # seconds between restarts

running=1; kid=""
trap 'running=0; [ -n "$kid" ] && kill -TERM "$kid" 2>/dev/null' INT TERM

n=0
while [ "$running" = 1 ]; do
  "$BIN" "$@" &                            # already root -> $kid is the collector itself
  kid=$!
  wait "$kid"; rc=$?
  [ "$running" = 1 ] || break              # clean shutdown requested -> do NOT restart
  n=$((n + 1))
  printf 'WARN: kv_copy exited rc=%s at %s -- CAPTURE GAP, restart #%d/%d\n' \
         "$rc" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$n" "$MAX_RESTARTS" >&2
  # Emit a gap marker INTO the record stream so the reader/health flag the window incomplete
  # (not just the operator-facing stderr line). JSONL only -- a JSON line would corrupt EKV_BINARY.
  [ -z "${EKV_BINARY:-}" ] && printf '{"type":"gap","reason":"collector restart %d"}\n' "$n"
  if [ "$n" -ge "$MAX_RESTARTS" ]; then
    echo "FATAL: kv_copy restarted $n times -- crash loop, giving up" >&2
    exit 1
  fi
  sleep "$BACKOFF"
done
echo "collector supervisor: clean shutdown after $n restart(s)" >&2
