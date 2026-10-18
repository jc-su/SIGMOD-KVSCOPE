#!/usr/bin/env bash
# Identity under TLS, one method. http_label reads the request after decryption via an SSL_read_ex
# uprobe (one layer above the encrypted socket), so net_label recovers the prompt from a TLS-only
# vllm serve with no engine hook and no off-wire telemetry. PASS = both distinct HTTPS prompts are
# recovered from the encrypted traffic. GPU 1 ONLY.
set -u
cd "$(dirname "$0")/.."
PY=/opt/conda/bin/python3
VLLM=/opt/conda/bin/vllm
PORT="${EKV_PORT:-8766}"
MODEL="${EKV_MODEL:-facebook/opt-125m}"
LIBSSL="${EKV_LIBSSL:-/opt/conda/lib/libssl.so.3}"
KEY=/tmp/ekv_key.pem CERT=/tmp/ekv_cert.pem
HTTP=/tmp/ekv_tls_http.jsonl; SOUT=/tmp/ekv_tls_serve.out; ERR=/tmp/ekv_tls_http.err
make -s -C observe/ebpf http_label || { echo "FAIL build"; exit 1; }
rm -f "$HTTP" "$ERR" "$SOUT"
openssl req -x509 -newkey rsa:2048 -keyout "$KEY" -out "$CERT" -days 1 -nodes -subj "/CN=localhost" >/dev/null 2>&1

CUDA_VISIBLE_DEVICES=1 "$VLLM" serve "$MODEL" --port "$PORT" --gpu-memory-utilization 0.30 \
  --max-model-len 2048 --ssl-keyfile "$KEY" --ssl-certfile "$CERT" > "$SOUT" 2>&1 &
SRV=$!
for _ in $(seq 1 400); do curl -sk "https://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  kill -0 $SRV 2>/dev/null || { echo "FAIL: serve died"; tail -15 "$SOUT"; exit 1; }; sleep 0.5; done
curl -sk "https://127.0.0.1:$PORT/health" >/dev/null 2>&1 || { echo "FAIL: not healthy"; tail -15 "$SOUT"; kill $SRV; exit 1; }
APIPID=$(ss -ltnp 2>/dev/null | grep ":$PORT" | grep -oE 'pid=[0-9]+' | head -1 | cut -d= -f2)
[ -z "$APIPID" ] && APIPID=$(sudo ss -ltnp 2>/dev/null | grep ":$PORT" | grep -oE 'pid=[0-9]+' | head -1 | cut -d= -f2)
echo "TLS API server PID=$APIPID; attaching http_label (SSL_read_ex on $LIBSSL)"
sudo EKV_LIBSSL="$LIBSSL" ./observe/ebpf/http_label "$APIPID" > "$HTTP" 2>"$ERR" &
sleep 2
grep "uprobe on" "$ERR"
curl -sk "https://127.0.0.1:$PORT/v1/completions" -H 'content-type: application/json' \
  -d '{"model":"'"$MODEL"'","prompt":"TLS encrypted prompt about quantum computing alpha one","max_tokens":5}' >/dev/null
curl -sk "https://127.0.0.1:$PORT/v1/completions" -H 'content-type: application/json' \
  -d '{"model":"'"$MODEL"'","prompt":"A different encrypted request concerning distributed databases beta","max_tokens":5}' >/dev/null
sleep 1
sudo pkill -f observe/ebpf/http_label 2>/dev/null
kill $SRV 2>/dev/null; pkill -f 'vllm serve' 2>/dev/null; pkill -f EngineCore 2>/dev/null
echo "=== net_label parse of the TLS-decrypted capture ==="
"$PY" - "$HTTP" <<'PYEOF'
import sys
sys.path.insert(0, ".")
from observe import net_label as nl
reqs = nl.parse_http_recs(sys.argv[1])
prompts = [r.prompt for r in reqs]
print(f"recovered {len(reqs)} request(s) from TLS traffic:")
for p in prompts:
    print("  ", repr(p[:70]))
want = ["quantum computing alpha one", "distributed databases beta"]
ok = all(any(w in p for p in prompts) for w in want)
print("PASS: TLS request identity recovered from decrypted reads, no engine hook" if ok
      else "FAIL: prompts not recovered from TLS traffic")
sys.exit(0 if ok else 1)
PYEOF
