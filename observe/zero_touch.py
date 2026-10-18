"""Zero-touch readout + join for the eKV pass — reads the pass's published GPU-capture
buffer from OUTSIDE the engine process, with no in-process Python.

The pass (EKV_PASS_AUTO) self-allocates a host-pinned, device-mapped buffer, writes the
block_table GPU-side into it, and publishes its address to /tmp/ekv_pass_<pid>.json. Here
we read that host RAM via /proc/<pid>/mem — exactly the user-memory read eBPF's
bpf_probe_read_user performs — turning the device-side block_table capture into host-side
per-slot block lists. (The eBPF realization of this same read is observe/ebpf/ekv_readout,
which uprobes a per-step engine call and bpf_probe_read_user's the published host_va,
printing the identical {slots, cycles}; this module is the pure-stdlib /proc equivalent.)
Joined with the eBPF network label (net_label: prompts -> tokens, matched to the engine's
input_ids), every block is attributed to a network request with NO engine hook.

Pure stdlib so the reader is a lightweight external process (no torch/triton/CUDA):
    sudo python observe/zero_touch.py <engine_pid>      # -> {slot: [block ids]} JSON
"""
from __future__ import annotations

import array
import glob
import json
from dataclasses import dataclass

NULL_BLOCK_ID = 0
UNWRITTEN = -1   # the pass memsets the buffer to 0xff (-1) so unused rows are detectable


@dataclass
class PassBuffer:
    pid: int
    host_va: int
    bytes: int
    ncap: int
    max_rows: int
    timer_host_va: int = 0     # u64[max_rows] per-slot clock64 cycles (0 if timer off)
    timer_bytes: int = 0
    kv_ptr_host_va: int = 0    # u64: the KV-cache base ptr the kernel was handed (pool id)
    kv_ptr_bytes: int = 0
    qsl_host_va: int = 0       # i32[qsl_count]: cu_seqlens_q snapshot -> per-seq query length
    qsl_count: int = 0
    row_form: str = "paged"    # "paged" | "csr" | "arith" -- arith has no capture array, only the formula
    arith_stride_host_va: int = 0  # u64: per-request KV stride (row_form=="arith"; region=seq*stride)
    kernel: str = ""           # the instrumented kernel's name (per-kernel buffers; reader-side filter)
    timer_dev: int = 0         # DEVICE ptr of the timer accumulator (device-resident since the
    timer_ipc: str = ""        # host-mapped per-CTA atomic was a PCIe storm); read via D2H / IPC


def runtime_dir() -> str:
    """Where the pass publishes: $EKV_RUNTIME_DIR (production: /run/ekv, shared by engine and
    daemon -- see docs/OPERATIONS.md) or /tmp. Must match the ENGINE's setting."""
    import os
    d = os.environ.get("EKV_RUNTIME_DIR", "").strip()
    return d or "/tmp"


def pub_path(pid: int) -> str:
    return f"{runtime_dir()}/ekv_pass_{pid}.json"


def read_json_lines(path: str) -> list[dict]:
    """The records of one of the pass's append-only JSON-lines files."""
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def find_published(pid: int | None = None, out_dir: str | None = None) -> list[PassBuffer]:
    """The pass's published buffer descriptor(s), PAGED entries first. One process can publish
    SEVERAL lines (one per instrumented path): the paged/CSR capture for the attention kernel,
    plus the arithmetic closed form, which a foreign kernel (e.g. a torch.inductor GEMV) can also
    legitimately match. Each read_pass_* helper picks the FIRST entry carrying the field it needs,
    so the attention kernel's buffers win whenever both exist. pid=None -> all on the box."""
    base = out_dir or runtime_dir()
    pat = f"{base}/ekv_pass_{pid}.json" if pid else f"{base}/ekv_pass_*.json"
    res = []
    for f in glob.glob(pat):
        try:
            lines = [json.loads(ln) for ln in open(f) if ln.strip()]
        except (ValueError, OSError):
            continue
        for m in lines:
            try:
                res.append(PassBuffer(int(m["pid"]), int(m.get("host_va", 0)),
                                      int(m.get("bytes", 0)),
                                      int(m.get("ncap", 0)), int(m["max_rows"]),
                                      int(m.get("timer_host_va", 0)), int(m.get("timer_bytes", 0)),
                                      int(m.get("kv_ptr_host_va", 0)), int(m.get("kv_ptr_bytes", 0)),
                                      qsl_host_va=int(m.get("qsl_host_va", 0)),
                                      qsl_count=int(m.get("qsl_count", 0)),
                                      row_form=str(m.get("row_form", "paged")),
                                      arith_stride_host_va=int(m.get("arith_stride_host_va", 0)),
                                      kernel=str(m.get("kernel", "")),
                                      timer_dev=int(m.get("timer_dev", 0)),
                                      timer_ipc=str(m.get("timer_ipc", ""))))
            except (ValueError, KeyError):
                pass
    res.sort(key=lambda b: b.row_form != "paged")      # paged first (stable), arith after
    return res


def _first(bufs: list[PassBuffer], pred, kernel: str | None = None) -> PassBuffer | None:
    """First buffer satisfying pred, optionally restricted to kernels whose name contains
    `kernel`. With per-kernel buffer sets several kernels can publish (the attention kernel,
    its variants, and foreign inductor kernels that satisfy the same structural criteria); the name filter is
    reader-side METADATA disambiguation (detection itself stays name-free). Structural
    disambiguation across kernels is the open next step."""
    return next((b for b in bufs if (kernel is None or kernel in b.kernel) and pred(b)), None)


def read_proc_mem(pid: int, va: int, n: int) -> bytes:
    """Read n bytes at user VA `va` from process `pid` (needs ptrace perms / root). The
    without an engine hook readout — the same memory read eBPF bpf_probe_read_user does, externally."""
    with open(f"/proc/{pid}/mem", "rb", 0) as f:
        f.seek(va)
        return f.read(n)


_cudart = None


def _cudart_lib():
    global _cudart
    if _cudart is None:
        import ctypes
        for n in ("libcudart.so.12", "libcudart.so"):
            try:
                _cudart = ctypes.CDLL(n)
                break
            except OSError:
                pass
    return _cudart


def read_dev_mem(dev_ptr: int, n: int) -> bytes:
    """Read n bytes from a DEVICE pointer via cudaMemcpy D2H. In-process only (same CUDA context as
    the pass) -- the timer accumulator is device-resident because a host-mapped per-CTA clock64
    atomic is a non-posted PCIe round trip each (~24% on decode-heavy 3B). A cross-process daemon
    instead opens the exported CUDA-IPC handle (timer_ipc) and D2Hs from that."""
    import ctypes
    cu = _cudart_lib()
    if cu is None:
        raise RuntimeError("libcudart not found for device timer readback")
    buf = (ctypes.c_char * n)()
    rc = cu.cudaMemcpy(buf, ctypes.c_void_p(dev_ptr), ctypes.c_size_t(n), 2)   # 2 = D2H
    if rc != 0:
        raise RuntimeError(f"cudaMemcpy D2H rc={rc} (device timer readback)")
    return bytes(buf)


def parse_slots(data: bytes, ncap: int, max_rows: int) -> dict[int, list[int]]:
    # Block id 0 is treated as padding throughout the stack. This is guaranteed correct on vLLM
    # V1, which reserves physical block 0 as the never-allocated null_block (BlockPool.__init__
    # pops it before any request can receive it). SGLang's token-pool indices have no such
    # documented reservation; if a target engine can hand out slot 0, this filter would hide it.
    """{slot -> ncap-wide block_table row} for ACTIVE slots (row[0] != -1 sentinel).
    Trailing NULL_BLOCK_ID padding is kept; trim with seq_lens downstream if wanted."""
    a = array.array("i")
    a.frombytes(data[: max_rows * ncap * 4])
    out: dict[int, list[int]] = {}
    for s in range(min(max_rows, len(a) // ncap)):
        row = a[s * ncap:(s + 1) * ncap]
        if len(row) == ncap and row[0] != UNWRITTEN:
            out[s] = list(row)
    return out


def read_pass_slots(pid: int | None = None, kernel: str | None = None) -> dict[int, list[int]]:
    """find + read + parse in one call. slot -> block_table row."""
    b = _first(find_published(pid), lambda x: x.host_va, kernel)
    if b is None:                              # no capture array (e.g. row_form=="arith") -> use read_pass_region
        return {}
    return parse_slots(read_proc_mem(b.pid, b.host_va, b.bytes), b.ncap, b.max_rows)


def read_pass_region(pid: int | None = None, kernel: str | None = None) -> dict[int, int]:
    """{seq -> KV-region START offset (elements from kv_base)} for an ARITHMETIC-mu kernel
    (row_form=="arith"). A non-paged/contiguous attention has no index array, so the
    request->region map is the closed form region(seq) = seq*stride. Combined with
    read_pass_kv_ptr(), request seq's KV lives at kv_base + region[seq] elements. Empty for
    index-based (paged/csr) kernels -- those use read_pass_slots instead."""
    b = _first(find_published(pid), lambda x: x.row_form == "arith" and x.arith_stride_host_va,
               kernel)
    if b is None:
        return {}
    a = array.array("q")                                 # int64 per-request stride
    a.frombytes(read_proc_mem(b.pid, b.arith_stride_host_va, 8))
    stride = int(a[0])
    return {seq: seq * stride for seq in range(b.max_rows)}


def read_pass_cycles(pid: int | None = None, kernel: str | None = None) -> dict[int, int]:
    """{slot -> accumulated clock64 cycles} from the pass's timer buffer (per-request attention
    compute), read externally via /proc/mem. Empty if the timer is off.

    Three properties of this signal the consumer MUST respect (clock64 is a secondary, opt-in
    EKV_TIMER signal; the primary compute classifier is the exact query length, not cycles):

    * UNIT = SM-RESIDENCY, not time and not pure arithmetic work. clock64 is per-SM and a CTA never
      migrates, so its entry->return delta is a valid duration on one clock; the accumulated row is
      the SM time the request's CTAs occupied (the GPU analogue of CPU core-seconds). Cycles are
      DVFS-invariant only for compute-bound phases (a memory stall is fixed ns, so its cycle count
      scales with the clock), and co-residency inflates durations by the contention factor. What IS
      robust: the per-row SHARE within a sync-bounded window -- every CTA in it sees the same clock
      and the same contention regime (symmetric in homogeneous decode) -- applied to the window's
      MEASURED wall time. Summing raw cycles across windows and calling it ns is NOT valid. The
      full contract + the attribution algebra live in observe/cycles.py.
    * VISIBILITY = read AFTER a completion sync. The GPU writes these via a RELAXED, GPU-scope atomic
      to device-mapped host memory; the host sees a consistent value only after a kernel-completion
      happens-before. Read only after an observed cuda{Stream,Event}Synchronize return -- which the
      eBPF tracing already captures, so the happens-before anchor is available without an engine hook (no
      in-process hook). Reading mid-flight can return a partially-flushed value.
    * CUMULATIVE, no step boundary. The buffer ACCUMULATES over the whole run; a slot is reused by
      many requests, so a raw read is the sum over everyone who ever held that slot. For per-request
      attribution, DIFFERENCE across a step boundary (read at two observed sync points and subtract),
      then attribute the delta via the slot->request map of that step. This step signal is, again,
      the observed sync-return -- the honest reason the cumulative timer is not pure one-shot
      without an engine hook: it needs a boundary, but an OBSERVABLE one, not an engine hook."""
    b = _first(find_published(pid), lambda x: x.timer_dev or x.timer_host_va, kernel)
    if b is None:
        return {}
    data = read_dev_mem(b.timer_dev, b.timer_bytes) if b.timer_dev \
        else read_proc_mem(b.pid, b.timer_host_va, b.timer_bytes)
    a = array.array("q")                              # int64 cycle accumulators
    a.frombytes(data[: (b.timer_bytes // 8) * 8])
    return {s: int(a[s]) for s in range(min(b.max_rows, len(a))) if a[s] > 0}


def read_pass_kv_ptr(pid: int | None = None, kernel: str | None = None) -> int | None:
    """The KV-cache base pointer the attention kernel was handed, written into the pass's
    kv_ptr buffer and read externally. Feed to blockmap.identify_kv_pool(kv_ptr=...) to pin
    the KV pool among the eBPF-captured allocations EXACTLY (no size heuristic), on any
    workload. None if the kernel exposed no KV-cache arg / the field is unset."""
    b = _first(find_published(pid), lambda x: x.kv_ptr_host_va and x.kv_ptr_bytes >= 8, kernel)
    if b is None:
        return None
    a = array.array("Q")                              # u64
    try:
        raw = read_proc_mem(b.pid, b.kv_ptr_host_va, 8)
    except PermissionError:
        # cross-process /proc/mem is yama-gated. The daemon stays UNPRIVILEGED (a root daemon
        # is unnecessary attack surface, and the collectors already carry their own sudo); the
        # one 8-byte read it cannot do itself goes through a scoped privileged dd.
        import subprocess
        p = subprocess.run(["sudo", "-n", "dd", f"if=/proc/{b.pid}/mem", "bs=1", "count=8",
                            f"skip={b.kv_ptr_host_va}", "status=none"], capture_output=True)
        if p.returncode != 0 or len(p.stdout) != 8:
            raise
        raw = p.stdout
    a.frombytes(raw)
    return int(a[0]) or None


def read_pass_qlen(pid: int | None = None, kernel: str | None = None) -> dict[int, int]:
    """{seq -> query length computed this step} from the pass's cu_seqlens_q snapshot, read
    externally via /proc/mem: query_len[s] = qsl[s+1] - qsl[s] (the kernel's cur_batch_query_len).
    A LARGE value at a re-materialization means the prefix was RE-DERIVED (recompute); ~1 means it
    was read from cache (cache-hit/decode) — the exact recompute-vs-cache-hit signal. The buffer is
    cu_seqlens_q (cumulative, starts at 0); -1 (0xff) marks the unwritten tail. Empty if EKV_QLEN
    was off (no qsl buffer)."""
    b = _first(find_published(pid), lambda x: x.qsl_host_va and x.qsl_count >= 2, kernel)
    if b is None:
        return {}
    a = array.array("i")
    a.frombytes(read_proc_mem(b.pid, b.qsl_host_va, b.qsl_count * 4))
    cu = []
    for v in a:                                       # cumulative offsets until the -1 sentinel
        if int(v) == UNWRITTEN:
            break
        cu.append(int(v))
    return {s: cu[s + 1] - cu[s] for s in range(len(cu) - 1) if cu[s + 1] - cu[s] > 0}


def attribute(slot_blocks: dict[int, list[int]], slot_to_request: dict[int, str],
              seq_lens: dict[int, int] | None = None, block_size: int = 16
              ) -> dict[str, list[int]]:
    """Join slot->blocks (pass, GPU-side) with slot->request (eBPF network label) ->
    request -> sorted unique blocks, trimming each row to the seq's real block count
    (ceil(seq_len/block_size)) when seq_lens is supplied."""
    out: dict[str, set] = {}
    for s, row in slot_blocks.items():
        rid = slot_to_request.get(s)
        if rid is None:
            continue
        blocks = row
        if seq_lens and s in seq_lens and seq_lens[s] > 0:
            nb = (int(seq_lens[s]) + block_size - 1) // block_size
            blocks = row[:nb]
        out.setdefault(rid, set()).update(int(b) for b in blocks)
    return {r: sorted(v) for r, v in out.items()}


def first_blocks(slot_blocks: dict[int, list[int]]) -> dict[int, int]:
    """slot -> its first physical block (the request's stable identity anchor)."""
    return {s: row[0] for s, row in slot_blocks.items() if row and row[0] >= 0}


def anchor_first_blocks(slot_blocks: dict[int, list[int]],
                        slot_to_request: dict[int, str]) -> dict[int, str]:
    """{first_block -> request} from an UNCOMPACTED step (slot == seq, tied via the eBPF
    input_ids label). A request's first block is allocated at prefill and never moves, so
    it identifies the request across batch COMPACTION (when a finished request frees its
    row and a later request is shifted up into it)."""
    fb = first_blocks(slot_blocks)
    return {b: slot_to_request[s] for s, b in fb.items() if s in slot_to_request}


def slot_to_request_via_anchor(slot_blocks: dict[int, list[int]],
                               fb_to_request: dict[int, str]) -> dict[int, str]:
    """slot -> request at ANY (possibly compacted) step, by looking up each slot's stable
    first block — compaction-robust, vs a one-shot prefill-order map which goes stale."""
    return {s: fb_to_request[row[0]] for s, row in slot_blocks.items()
            if row and row[0] in fb_to_request}


if __name__ == "__main__":
    import sys
    pid = int(sys.argv[1]) if len(sys.argv) > 1 else None
    print(json.dumps({"slots": {str(s): b for s, b in read_pass_slots(pid).items()},
                      "cycles": {str(s): c for s, c in read_pass_cycles(pid).items()},
                      "qlen": {str(s): q for s, q in read_pass_qlen(pid).items()},
                      "kv_ptr": read_pass_kv_ptr(pid)}))
