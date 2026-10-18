"""kv_copy must see a real vLLM engine's CUDA runtime copies through every libcudart it maps.

Runs ``tests/_ebpf_runtime_copy_probe.py`` (vLLM 0.21, in process, native CPU KV offload) on one GPU
under ``EbpfSession``, which gives the collector every libcudart in ``/proc/<pid>/maps``. Passes if
the collector lost no record, recorded runtime cudaMemcpyAsync copies from the engine, and recorded
the offload's cuMemcpyBatchAsync blocks. Prints the counts per API and direction.

Needs sudo and a GPU. Usage: python tests/_ebpf_runtime_copy_check.py [--gpu 1] [--out DIR]
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from observe import parse_collector_report
from observe.ebpf_attach import EbpfSession, mapped_libcudarts

PROBE = ROOT / "tests" / "_ebpf_runtime_copy_probe.py"
RUNTIME_APIS = ("cudaMemcpyAsync", "cudaMemcpyAsync_ptsz")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    out = args.out or Path(tempfile.mkdtemp(prefix="ekv_runtime_copy_"))
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run(["make", "-s", "-C", str(ROOT / "observe/ebpf"), "kv_copy"], check=True)

    start = out / "start"
    log_path = out / "probe.log"
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=str(args.gpu),
        EKV_START=str(start),
        PYTHONPATH=f"{ROOT}:{os.environ.get('PYTHONPATH', '')}",
    )
    with open(log_path, "w") as log:
        proc = subprocess.Popen([sys.executable, str(PROBE)], env=env, stdout=log, stderr=subprocess.STDOUT)
    pid = None
    while pid is None:
        match = re.search(r"EKV_PID (\d+)", log_path.read_text(errors="replace"))
        if match:
            pid = int(match.group(1))
        elif proc.poll() is not None:
            raise SystemExit(f"FAIL: the engine exited before the collector attached; see {log_path}")
        else:
            time.sleep(0.2)
    runtimes = mapped_libcudarts(Path(f"/proc/{pid}/maps").read_text())
    print(f"engine pid {pid} maps {len(runtimes)} CUDA runtimes: {runtimes}")

    with EbpfSession(pid, out, alloc=False, copy=True) as session:
        time.sleep(0.5)
        start.touch()
        if proc.wait(timeout=900):
            raise SystemExit(f"FAIL: the engine failed; see {log_path}")
    loss = parse_collector_report(session.kv_copy_report.read_text())

    by_api: Counter[tuple[str, str]] = Counter()
    runtime_copies: dict[tuple[int, int, int, int], list[int]] = {}
    driver_copies: list[tuple[tuple[int, int, int, int], int]] = []
    with open(session.kv_copy) as capture:
        for line in capture:
            record = json.loads(line)
            if record["type"] != "enqueue" or record["pid"] != pid:
                continue
            by_api[(record["api"], record["kind"])] += 1
            key = (record["tid"], record["dst"], record["src"], record["bytes"])
            if record["api"] in RUNTIME_APIS:
                runtime_copies.setdefault(key, []).append(record["t_ns"])
            elif record["api"] == "cuMemcpy(driver)":
                driver_copies.append((key, record["t_ns"]))
    # A runtime copy calls the driver's cuMemcpy*Async_v2 5-25 us after its entry; the collector
    # must record that copy once, not again as a driver copy.
    duplicates = sum(
        1 for key, at in driver_copies if any(0 <= at - t < 1_000_000 for t in runtime_copies.get(key, ()))
    )
    for (api, kind), n in sorted(by_api.items()):
        print(f"  {api:24s} {kind:5s} {n}")
    runtime = sum(n for (api, _), n in by_api.items() if api in RUNTIME_APIS)
    batch = sum(n for (api, _), n in by_api.items() if api == "cuMemcpyBatchAsync")
    print(
        f"runtime cudaMemcpyAsync copies {runtime}; offload batch blocks {batch}; "
        f"driver copies recorded twice {duplicates}; collector loss {dict(loss)}"
    )
    if any(loss.values()):
        raise SystemExit(f"FAIL: the collector lost records: {loss}")
    if runtime == 0:
        raise SystemExit("FAIL: no runtime cudaMemcpyAsync copy from the engine was recorded")
    if duplicates:
        raise SystemExit(f"FAIL: {duplicates} runtime copies were also recorded as driver copies")
    if batch == 0:
        raise SystemExit("FAIL: no offload cuMemcpyBatchAsync block was recorded")
    print(f"PASS: kv_copy recorded the engine's runtime and offload copies (capture in {out})")


if __name__ == "__main__":
    main()
