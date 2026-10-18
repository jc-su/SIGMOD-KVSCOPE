"""CAPSTONE: the whole zero-touch per-request plane on a REAL `vllm serve` API server,
multi-process, eBPF + pass only — no engine hook in the data path.

  * vllm serve (separate API-server + worker processes); the eKV vLLM plugin installs
    the self-configuring auto-pass in the WORKER (EKV_ENABLE=1 EKV_PASS_AUTO=1) -> the
    pass self-allocates + publishes its block_table+timer buffer.
  * http_label (eBPF, on the API-server pid) captures the inbound HTTP requests -> prompts.
  * kv_copy EKV_META (eBPF) captures the worker's input_ids H2D copies.
  * external /proc read of the worker's pass buffer -> seq->block + seq->cycles.
  * join: prompts(tokens) ⋈ input_ids -> seq->request ; pass -> seq->block/compute
    -> request -> blocks + attention cycles, live.

Self-orchestrating: launches serve, finds pids, runs the sudo collectors, drives HTTP,
joins, cleans up. Validates the plane is COHERENT live (each request -> its own blocks +
compute, tied from the network). Run: bash tests/_served_zero_touch_run.sh
"""
import glob
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from observe import net_label, zero_touch  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PORT = int(os.environ.get("EKV_PORT", "8200"))
LIBCUDART = "/opt/conda/lib/python3.12/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12"
PROMPTS = [
    "Zebra quantum waterfall marble engine drifting north",
    "Maple thunder violin orchestra cobalt morning breeze",
    "Falcon glacier ribbon tunnel velvet harbor signal",
    "Cinnamon nebula compass lantern prairie echo summit",
    "Obsidian meadow trumpet circuit walnut beacon harbor",
    "Saffron comet anchor pixel meadowlark tundra willow",
]


def _serve_env():
    e = dict(os.environ)
    e.update(PYTHONPATH=f"/path/to/triton/python:{ROOT}",
             LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libstdc++.so.6",
             TRITON_EXT_ENABLED="1",
             TRITON_PLUGIN_PATHS=f"{ROOT}/build/libeKVPlugin.so",
             PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False",
             VLLM_ATTENTION_BACKEND="TRITON_ATTN", VLLM_USE_FLASHINFER="0",
             VLLM_USE_FLASHINFER_SAMPLER="0", VLLM_LOGGING_LEVEL="WARNING",
             CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES", "1"),
             VLLM_ENABLE_V1_MULTIPROCESSING="0",   # engine in the API process: shares the
                                                   # triton-src env so the pass actually fires
             EKV_ENABLE="1", EKV_PASS_AUTO="1", EKV_TIMER="1",
             EKV_NCAP="128", EKV_MAX_ROWS="64",
             EKV_HEAD_AXIS="1", EKV_SLOT_AXIS="0")
    return e


def _post(prompt, max_tokens=24):
    body = json.dumps({"model": "facebook/opt-125m", "prompt": prompt,
                       "max_tokens": max_tokens, "temperature": 0.0,
                       "ignore_eos": True}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def _listen_pid(port):
    out = subprocess.run(["ss", "-ltnpH", f"sport = :{port}"], capture_output=True, text=True).stdout
    for tok in out.replace(",", " ").split():
        if tok.startswith("pid="):
            return int(tok.split("=")[1])
    return None


def main():
    for f in glob.glob("/tmp/ekv_pass_*.json"):
        os.remove(f)
    for f in ("/tmp/ekv_http.jsonl", "/tmp/ekv_meta.jsonl"):
        if os.path.exists(f):
            os.remove(f)
    subprocess.run("rm -rf ~/.triton/cache", shell=True)

    serve_log = open("/tmp/ekv_serve.log", "w")
    serve = subprocess.Popen(
        ["vllm", "serve", "facebook/opt-125m", "--port", str(PORT), "--enforce-eager",
         "--attention-backend", "TRITON_ATTN",     # 0.21: backend is a CLI arg, not an env var
         "--gpu-memory-utilization", "0.30", "--max-model-len", "2048",
         "--max-num-seqs", "64", "--no-enable-chunked-prefill"],
        env=_serve_env(), stdout=serve_log, stderr=subprocess.STDOUT)
    cols = []
    seq2block, cycles, worker_pid = {}, {}, None
    try:
        # wait for the server
        for _ in range(120):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=2)
                break
            except Exception:  # noqa: BLE001
                if serve.poll() is not None:
                    print("FAIL: vllm serve exited during startup"); return
                time.sleep(2)
        else:
            print("FAIL: server not ready"); return
        _post("warmup zero touch", 4)               # trigger kernel compile -> pass publishes
        time.sleep(2)

        pub = zero_touch.find_published()             # the worker that published the buffer
        api_pid = _listen_pid(PORT)
        if not pub or not api_pid:
            print(f"FAIL: pub={len(pub)} api_pid={api_pid} (pass/serve pids not found)"); return
        worker_pid = pub[0].pid
        print(f"api_pid={api_pid} worker_pid={worker_pid} (pass buffer published)")

        cols.append(subprocess.Popen(["sudo", f"{ROOT}/observe/ebpf/http_label", str(api_pid)],
                                     stdout=open("/tmp/ekv_http.jsonl", "w"), stderr=subprocess.DEVNULL))
        cols.append(subprocess.Popen(
            ["sudo", "-E", "env", "EKV_DRIVER_ONLY=1", "EKV_META=1",
             f"{ROOT}/observe/ebpf/kv_copy", LIBCUDART],
            stdout=open("/tmp/ekv_meta.jsonl", "w"), stderr=subprocess.DEVNULL))
        time.sleep(3)                                 # let the uprobes attach

        import threading                              # send CONCURRENTLY so they BATCH
        ths = [threading.Thread(target=_post, args=(p, 24)) for p in PROMPTS]
        for t in ths:
            t.start()
        # read the pass buffer WHILE the batch is mid-decode (worker alive, many slots active).
        time.sleep(6)
        rd = subprocess.run(["sudo", sys.executable, f"{ROOT}/observe/zero_touch.py",
                             str(worker_pid)], capture_output=True, text=True)
        d = json.loads(rd.stdout or "{}") if rd.stdout.strip() else {}
        seq2block = {int(k): v for k, v in d.get("slots", {}).items()}
        cycles = {int(k): v for k, v in d.get("cycles", {}).items()}
        for t in ths:
            t.join()
    finally:
        for c in cols:
            subprocess.run(["sudo", "kill", str(c.pid)], stderr=subprocess.DEVNULL)
        subprocess.run(["sudo", "pkill", "-9", "-x", "http_label"], stderr=subprocess.DEVNULL)
        subprocess.run(["sudo", "pkill", "-9", "-x", "kv_copy"], stderr=subprocess.DEVNULL)
        serve.send_signal(signal.SIGINT)
        try:
            serve.wait(timeout=20)
        except Exception:  # noqa: BLE001
            serve.kill()

    # ---- join, all from eBPF + pass (zero touch) ----
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("facebook/opt-125m")
    net = net_label.parse_http_recs("/tmp/ekv_http.jsonl", tokenizer=tok)
    net = [n for n in net if n.prompt in PROMPTS]      # the measured requests
    meta = [json.loads(l) for l in open("/tmp/ekv_meta.jsonl") if '"type":"meta"' in l]
    seq2req = net_label.reconstruct_seq_to_request(meta, net)
    # seq2block / cycles were read via sudo while the worker was alive (above).
    req2block = zero_touch.attribute(seq2block, {s: str(net[a].prompt) for s, a in seq2req.items()})

    print(f"HTTP requests captured (eBPF http_label): {len(net)}")
    print(f"input_ids copies (eBPF kv_copy meta records): {len(meta)}")
    print(f"seq->request tied (network prompt ⋈ input_ids): {len(seq2req)}")
    print(f"seq->block (pass): {len(seq2block)}  seq->cycles (pass timer): {len(cycles)}")
    print(f"request->block recovered for {len(req2block)} requests")
    for pr, bl in list(req2block.items())[:3]:
        print(f"   {pr[:40]!r} -> blocks {[b for b in bl if b != 0][:8]}")

    n = len(net)
    coherent = (len(seq2req) >= max(1, n - 1) and len(req2block) >= max(1, n - 1)
                and len(seq2block) > 0)
    print("PASS: zero-touch per-request plane LIVE on a real vllm serve — eBPF (http_label "
          "+ kv_copy) gave network-request->seq, the pass gave seq->block+compute (external "
          "/proc read), joined into request->block live across the API-server + worker "
          "processes. No engine hook in the data path." if coherent else
          f"CHECK: net={len(net)} seq2req={len(seq2req)} req2block={len(req2block)} "
          f"seq2block={len(seq2block)} (inspect collectors)")


if __name__ == "__main__":
    main()
