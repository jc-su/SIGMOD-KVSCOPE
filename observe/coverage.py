"""Measured compiler-pass coverage for one live serving process.

The pass records one mutually exclusive classification for every compiled Triton function it
examines.  Instrumented variants optionally expose an exact device launch counter under
``EKV_COVERAGE=1``.  Unsupported functions remain in the report with an explicit reason; their
launch count is reported as unknown unless an external profiler supplies it.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

import numpy as np

from observe.cuda import Cuda
from observe.zero_touch import pub_path


CLASSES = ("observation_control", "observation_only", "recognized_declined", "untouched")


def _json_lines(path: str) -> list[dict]:
    records = []
    try:
        for line in open(path):
            try:
                records.append(json.loads(line))
            except ValueError:
                continue
    except OSError:
        pass
    return records


def report(pid: int, *, external: bool = False) -> dict:
    descriptors = _json_lines(pub_path(pid))
    by_host = {int(item.get("host_va", 0) or 0): item for item in descriptors
               if int(item.get("host_va", 0) or 0)}
    coverage_path = pub_path(pid).replace("ekv_pass_", "ekv_coverage_")
    variants = _json_lines(coverage_path)
    cuda = Cuda()
    rows = []
    for item in variants:
        row = dict(item)
        host = int(item.get("host_va", 0) or 0)
        descriptor = by_host.get(host, {})
        counter = 0
        try:
            if external and descriptor.get("coverage_launch_ipc"):
                counter = cuda.ipc_open(descriptor["coverage_launch_ipc"])
            elif not external:
                counter = int(descriptor.get("coverage_launch_dev", 0) or 0)
            row["launches"] = (int(np.frombuffer(cuda.d2h(counter, 8), dtype=np.uint64)[0])
                               if counter else None)
        except RuntimeError as error:
            row["launches"] = None
            row["launch_count_error"] = str(error)
        rows.append(row)
    compiled = Counter(row.get("coverage_class", "untouched") for row in rows)
    known_launches = Counter()
    for row in rows:
        if row.get("launches") is not None:
            known_launches[row.get("coverage_class", "untouched")] += int(row["launches"])
    return {
        "pid": pid,
        "abi_version": 2,
        "compiled_variants": {name: compiled.get(name, 0) for name in CLASSES},
        "known_launches": {name: known_launches.get(name, 0) for name in CLASSES},
        "variants": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--external", action="store_true",
                        help="open launch counters through CUDA IPC from another process")
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    result = report(args.pid, external=args.external)
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output:
        with open(args.output, "w") as output:
            output.write(rendered + "\n")


if __name__ == "__main__":
    main()
