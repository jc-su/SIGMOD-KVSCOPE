"""Validate the eBPF KV-pool allocation capture (observe/ebpf/kv_alloc): a uprobe +
uretprobe on a cudaMalloc-shaped function recovers each allocation's (addr, size) from
the data plane — the base BlockAddressMap.from_alloc needs for zero-touch addr->block,
with no in-process KV-tensor read. Standalone (no CUDA): a writer calls
fake_alloc(&p, SIZE) setting *p to a known device address; the probe must capture
(KNOWN_ADDR, SIZE), and from_alloc must then resolve addresses within that pool.
Needs sudo + a built kv_alloc."""
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from observe import BlockAddressMap, parse_ringbuf_drops

EBPF = f"{ROOT}/observe/ebpf/kv_alloc"
WRITER_SRC = f"{ROOT}/tests/_alloc_writer.c"
WRITER = "/tmp/_ekv_alloc_writer"
KNOWN_ADDR = 0x0000700000000000
KNOWN_SIZE = 268435456


def main():
    assert os.path.exists(EBPF), f"build it first: make -C observe/ebpf kv_alloc  ({EBPF} missing)"
    subprocess.run(["cc", "-O2", WRITER_SRC, "-o", WRITER], check=True)

    w = subprocess.Popen([WRITER], stderr=subprocess.DEVNULL)
    try:
        time.sleep(0.3)
        out = subprocess.run(
            ["sudo", "timeout", "-s", "INT", "4", EBPF, str(w.pid),
             f"{WRITER}:fake_alloc", "-m", "1048576"],
            capture_output=True, text=True, timeout=25)
        recs = [json.loads(l) for l in out.stdout.splitlines() if l.startswith("{")]
        assert recs, f"no alloc records captured\nstderr: {out.stderr}"
        drops = parse_ringbuf_drops(out.stderr)
        assert drops == 0, f"kv_alloc lost {drops} records on ring buffer reservation"
        hit = [r for r in recs if r["size"] == KNOWN_SIZE and r["addr"] == KNOWN_ADDR]
        assert hit, f"the known allocation was not captured; first records: {recs[:3]}"
        print(f"kv_alloc: captured {len(recs)} allocs; the known "
              f"(addr={KNOWN_ADDR:#x}, {KNOWN_SIZE >> 20} MB) present via uprobe+uretprobe  OK")

        # the eBPF-captured base feeds from_alloc -> zero-touch addr->block (no tensors)
        base = hit[0]["addr"]
        nblocks, stride = 64, KNOWN_SIZE // 64
        m = BlockAddressMap.from_alloc(base, nblocks, stride)
        b = m.resolve(base + 7 * stride + 3)
        assert b is not None and b.block_id == 7, b
        print("from_alloc on the eBPF-captured base: addr->block resolves (block 7)  OK")
        print("PASS: KV-pool base captured via eBPF -> zero-touch addr->block (no in-process read)")
    finally:
        w.terminate()
        try:
            w.wait(timeout=3)
        except Exception:
            w.kill()


if __name__ == "__main__":
    main()
