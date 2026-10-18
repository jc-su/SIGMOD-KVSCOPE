"""Binary transport round-trip: the binary parser must yield the SAME Copies as the
JSON path (shared policy), and honor the same H2H-skip. No eBPF/CUDA needed —
constructs the exact wire bytes the C collector emits in EKV_BINARY mode."""
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from observe.ebpf_reader import (_BIN_HDR, _BIN_REC, read_copies,  # noqa: E402
                                 read_copies_binary)


def rec(typ, kind, n, ts, handle, dst, src, nbytes=4096, pid=100):
    # kv_copy_evt: type,kind,api,pid,tid,n | ts_ns,bytes,first_ns,wait_ns,handle,dst,src
    return _BIN_REC.pack(typ, kind, 0, pid, pid, n, ts, nbytes, 0, 0, handle, dst, src)


HDR = _BIN_HDR.pack(b"eKV1", 1, _BIN_REC.size)

# two D2H enqueues on stream 7 (src=device addr), then a stream-sync completion n=2
binary = HDR + rec(0, 2, 1, 1000, 7, 1, 0xA000) + rec(0, 2, 1, 1001, 7, 1, 0xB000) \
             + rec(1, 0, 2, 5000, 7, 0, 0)
copies = read_copies_binary(io.BytesIO(binary))
assert len(copies) == 2 and {c.addr for c in copies} == {0xA000, 0xB000}
assert all(c.finished_at == 5000 for c in copies), copies
print("binary: 2 enqueues + completion ->", [(hex(c.addr), c.cost) for c in copies])

# the SAME events as JSON must give identical Copies
jsonl = [
    '{"type":"enqueue","t_ns":1000,"kind":"D2H","stream":7,"src":40960,"dst":1,"bytes":4096,"pid":100}',
    '{"type":"enqueue","t_ns":1001,"kind":"D2H","stream":7,"src":45056,"dst":1,"bytes":4096,"pid":100}',
    '{"type":"completion","via":"stream-sync","t_ns":5000,"n":2,"handle":7}',
]
cj = read_copies(jsonl)
key = lambda cs: [(c.addr, c.finished_at, c.direction, c.cost) for c in cs]
assert key(cj) == key(copies), (key(cj), key(copies))
print("binary == json (identical Copies)  OK")

# H2H (kind 0 on an enqueue) is skipped in binary too
h2h = HDR + rec(0, 0, 1, 1, 9, 20, 10, nbytes=64)
assert read_copies_binary(io.BytesIO(h2h)) == []
print("binary H2H skipped  OK")

# a short final record (collector stopped mid-write) ends cleanly, not a crash
truncated = binary[:-10]
read_copies_binary(io.BytesIO(truncated))
print("binary truncated tail handled cleanly  OK")
print("PASS: binary transport parser matches JSON + shares policy + robust tail")
