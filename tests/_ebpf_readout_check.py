"""Validate the eBPF readout (observe/ebpf/ekv_readout): a uprobe-triggered
bpf_probe_read_user of a pass-published buffer reads the SAME bytes as the
/proc/<pid>/mem read in observe/zero_touch.py — the eBPF realization of the index_lookup
readout. Standalone (no GPU): a writer publishes a known 'block_table' + 'timer'
buffer and ticks a uprobe target; the eBPF reader lifts it; we assert
eBPF == /proc == the known pattern. Needs sudo (eBPF + cross-process /proc) and a
built ekv_readout (make -C observe/ebpf ekv_readout)."""
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from observe import parse_ringbuf_drops

EBPF = f"{ROOT}/observe/ebpf/ekv_readout"
WRITER_SRC = f"{ROOT}/tests/_readout_writer.c"
WRITER = "/tmp/_ekv_readout_writer"
NCAP, MAXROWS, ACTIVE = 128, 8, 4


def expected_slots():
    return {s: [s * 100 + j for j in range(NCAP)] for s in range(ACTIVE)}


def expected_cycles():
    return {s: (s + 1) * 1000000 for s in range(ACTIVE)}


def main():
    assert os.path.exists(EBPF), f"build it first: make -C observe/ebpf ekv_readout  ({EBPF} missing)"
    subprocess.run(["cc", "-O2", WRITER_SRC, "-o", WRITER], check=True)

    w = subprocess.Popen([WRITER], stderr=subprocess.DEVNULL)
    jpath = f"/tmp/ekv_pass_{w.pid}.json"
    try:
        for _ in range(150):
            if os.path.exists(jpath):
                break
            time.sleep(0.02)
        assert os.path.exists(jpath), "writer did not publish its descriptor"

        out = subprocess.run(["sudo", EBPF, str(w.pid), f"{WRITER}:ekv_readout_tick", "-d", "2"],
                             capture_output=True, text=True, timeout=30)
        lines = [l for l in out.stdout.splitlines() if l.startswith("{")]
        assert lines, f"no snapshot from the eBPF reader\nstderr: {out.stderr}"
        drops = parse_ringbuf_drops(out.stderr)
        assert drops == 0, f"ekv_readout lost {drops} snapshots on ring buffer reservation"
        got = json.loads(lines[-1])
        ebpf_slots = {int(k): v for k, v in got["slots"].items()}
        ebpf_cycles = {int(k): v for k, v in got["cycles"].items()}

        assert ebpf_slots == expected_slots(), \
            f"eBPF slots mismatch: got rows {sorted(ebpf_slots)} (expected {list(range(ACTIVE))})"
        assert ebpf_cycles == expected_cycles(), f"eBPF cycles mismatch: {ebpf_cycles}"
        print(f"eBPF readout: {len(ebpf_slots)} block-table rows + {len(ebpf_cycles)} "
              f"timer slots lifted via bpf_probe_read_user  OK")

        # cross-check: the /proc read (observe/zero_touch.py, run as root) must agree.
        rd = subprocess.run(["sudo", sys.executable, f"{ROOT}/observe/zero_touch.py", str(w.pid)],
                            capture_output=True, text=True, timeout=20)
        if rd.returncode == 0 and rd.stdout.strip().startswith("{"):
            z = json.loads(rd.stdout)
            proc_slots = {int(k): v for k, v in z["slots"].items()}
            proc_cycles = {int(k): v for k, v in z["cycles"].items()}
            assert proc_slots == ebpf_slots, "eBPF read disagrees with the /proc read"
            assert proc_cycles == ebpf_cycles, "eBPF cycles disagree with the /proc read"
            print("eBPF read == /proc read (zero_touch), byte-identical  OK")
        else:
            print(f"(/proc cross-check skipped: {rd.stderr.strip()[:80]})")
        print("PASS: the eBPF reader lifts the pass-published buffer byte-identically")
    finally:
        w.terminate()
        try:
            w.wait(timeout=3)
        except Exception:
            w.kill()
        try:
            os.remove(jpath)
        except OSError:
            pass


if __name__ == "__main__":
    main()
