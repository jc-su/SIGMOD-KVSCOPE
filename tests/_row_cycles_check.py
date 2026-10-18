"""Per-request EXACT device cycles from the pass timer (the Phase-1 spike gate).

The pass timer brackets clock64 per CTA and atomic-adds the delta into timer[row]. This proves the
row is the REQUEST (the kernel's own seq index), not the CTA: the kernel decomposes each sequence
into QB query-block CTAs (pid0 = seq*QB + qb, the chunked-prefill grid shape) x NH head CTAs, and
each sequence's attention loops over its OWN planted context length. PASS requires:
  (1) exactly num_seqs timer rows light up (CTA-indexed timing would light QB x as many);
  (2) per-row cycles track the planted per-row work (linear in context, high R^2, monotone);
  (3) the join: capture row->blocks and timer row->cycles share the SAME row, so each request,
      keyed by its own first block (the content-addressed anchor), gets {blocks, cycles} --
      per-request MEASURED attention compute, no engine metadata and no model.

Env (the runner sets it): TRITON_EXT_ENABLED, TRITON_PLUGIN_PATHS, EKV_PASS_AUTO, EKV_TIMER=1.
"""
import ctypes
import json
import os
import sys

import numpy as np
import torch
import triton
import triton.language as tl

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


from ekv_vllm_plugin import install_pass  # noqa: E402

install_pass()


@triton.jit
def attn_varwork(block_table, seq_nblks, kv_cache, q_ptr, out_ptr,
                 QB: tl.constexpr, W: tl.constexpr, D: tl.constexpr, REPS: tl.constexpr):
    # Paged attention with a chunked-prefill grid: pid0 enumerates QUERY BLOCKS (QB per sequence),
    # so the CTA index is NOT the request index -- seq = pid0 // QB is, and the timer must key on it.
    # REPS multiplies the per-block work so the planted signal dominates the fixed per-CTA overhead
    # (the mid-body capture store) in the cycle counts.
    pid = tl.program_id(0)                                  # q-block CTA: seq*QB + qb
    pid_h = tl.program_id(1)                                # head CTA
    seq = pid // QB                                         # the kernel's OWN request index
    blk = tl.arange(0, W)
    phys = tl.load(block_table + seq * W + blk)             # (1) index map row (paged: seq*W)
    nb = tl.load(seq_nblks + seq)                           # planted per-request context (blocks)
    d = tl.arange(0, D)
    q = tl.load(q_ptr + seq * D + d)
    acc = 0.0
    for rep in range(0, REPS):
        for i in range(0, nb):                              # per-request WORK: nb[seq] gathers
            pb = tl.sum(tl.where(blk == i, phys, 0)) + rep * 0  # phys[i] from the gathered index row
            k = tl.load(kv_cache + pb * D + d)              # (2) KV gather through the index map
            acc += tl.sum(q * k)                            # (3) contraction <q,k>
    tl.store(out_ptr + pid * tl.num_programs(1) + pid_h, acc)


def main():
    assert torch.cuda.is_available()
    # grid = NQ*QB x NH = 64 CTAs <= 84 SMs (A6000): one CTA per SM, so a CTA's clock64 duration is
    # its own work, not co-residency contention. L launches accumulate to average scheduling noise.
    NQ, QB, NH, W, D, NPHYS, MAXROWS, REPS, L = 16, 2, 2, 64, 64, 9000, 256, 8, 16
    os.environ.setdefault("EKV_NCAP", str(W))
    os.environ.setdefault("EKV_MAX_ROWS", str(MAXROWS))
    torch.manual_seed(0)
    # distinct first blocks -> each row's first block is its content-addressed anchor
    bt = torch.randint(1, NPHYS, (NQ, W), dtype=torch.int32, device="cuda")
    bt[:, 0] = torch.arange(1, NQ + 1, dtype=torch.int32)
    nblks = torch.tensor([4 * (r + 1) for r in range(NQ)], dtype=torch.int32, device="cuda")
    kv = torch.randn(NPHYS * D, dtype=torch.float16, device="cuda")
    q = torch.randn(NQ * D, dtype=torch.float16, device="cuda")
    out = torch.zeros(NQ * QB * NH, dtype=torch.float32, device="cuda")
    # per-launch DELTAS, median over launches: the buffer is cumulative, so differencing across an
    # observed completion (sync) isolates one launch -- the same step-differencing the reader
    # documents -- and the median rejects launches distorted by co-tenant bursts on a shared GPU.
    from observe.zero_touch import read_pass_cycles
    attn_varwork[(NQ * QB, NH)](bt, nblks, kv, q, out, QB=QB, W=W, D=D, REPS=REPS)
    torch.cuda.synchronize()                                # visibility: read after completion
    prev = read_pass_cycles(os.getpid())
    deltas = []
    for _ in range(L):
        attn_varwork[(NQ * QB, NH)](bt, nblks, kv, q, out, QB=QB, W=W, D=D, REPS=REPS)
        torch.cuda.synchronize()
        cur = read_pass_cycles(os.getpid())
        deltas.append({s: cur.get(s, 0) - prev.get(s, 0) for s in cur})
        prev = cur

    path = f"/tmp/ekv_pass_{os.getpid()}.json"
    if not os.path.exists(path):
        print("FAIL: pass did not self-allocate/publish")
        sys.exit(1)
    meta = json.load(open(path))
    if not int(meta.get("timer_dev", 0) or meta.get("timer_host_va", 0)):
        print("FAIL: no timer buffer published (EKV_TIMER not honored)")
        sys.exit(1)

    # capture: row -> physical blocks (must be exact, graded by the same request index)
    cap = (ctypes.c_int32 * (int(meta["bytes"]) // 4)).from_address(int(meta["host_va"]))
    rows = np.ctypeslib.as_array(cap).reshape(MAXROWS, W)
    cap_exact = bool((rows[:NQ] == bt.cpu().numpy()).all()) and bool((rows[NQ:] == -1).all())

    # pin the instrumentation UNIT empirically (report-only; shares are invariant to any constant
    # per-CTA multiplier): one event-timed launch vs its cycle delta. multiplier ~= adds per CTA
    # (1 = once per CTA; ~num_warps = once per warp). Max-clock normalization makes this a bound.
    ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ev0.record()
    attn_varwork[(NQ * QB, NH)](bt, nblks, kv, q, out, QB=QB, W=W, D=D, REPS=REPS)
    ev1.record()
    torch.cuda.synchronize()
    cur = read_pass_cycles(os.getpid())
    one = {s: cur.get(s, 0) - prev.get(s, 0) for s in cur}
    clock_hz = torch.cuda.get_device_properties(0).clock_rate * 1e3   # kHz -> Hz (max clock)
    dur_cyc = ev0.elapsed_time(ev1) * 1e-3 * clock_hz
    mult = sum(one.values()) / max(NQ * QB * NH * dur_cyc, 1)
    print(f"instrumentation unit: ~{mult:.2f} atomic adds per CTA "
          f"(launch {ev0.elapsed_time(ev1)*1e3:.0f} us at max clock; shares invariant regardless)")

    # timer: row -> median per-launch cycle delta, via the same external reader the analysis uses
    cyc = {s: float(np.median([d.get(s, 0) for d in deltas])) for s in prev}
    lit = sorted(cyc)
    per_request = len(lit) == NQ and max(lit) == NQ - 1     # CTA-keyed timing would light NQ*QB rows
    print(f"timer rows lit: {len(lit)} (expect {NQ}; per-CTA would be {NQ * QB}) max_row={max(lit)}")

    # per-row cycles must track the planted per-row work: linear in nb (slope>0), high R^2, monotone
    nb = nblks.cpu().numpy().astype(float)
    c = np.array([cyc.get(r, 0) for r in range(NQ)], dtype=float)
    A = np.vstack([nb, np.ones_like(nb)]).T
    (slope, icpt), res, *_ = np.linalg.lstsq(A, c, rcond=None)
    ss_tot = float(((c - c.mean()) ** 2).sum())
    r2 = 1.0 - float(res[0]) / ss_tot if len(res) and ss_tot > 0 else 0.0
    mono = float(np.corrcoef(nb, c)[0, 1])
    print(f"cycles vs planted context: slope={slope:.0f} cyc/blk intercept={icpt:.0f} "
          f"R^2={r2:.4f} corr={mono:.4f}")
    # R^2 answers "is the relationship linear", which is NOT the question a reader of an
    # attribution system has. R^2 = 1.0 would survive a timer reporting double the true cycles,
    # or every request offset by a constant -- both perfectly linear, both wrong. State instead
    # the two quantities that bound what a POLICY can be misled by:
    #   worst_resid  how far any single request's measured cycles sit off the common line,
    #                as a fraction of that request's own cycles -- per-request attribution error;
    #   worst_share  the same error re-expressed in the unit policies actually consume
    #                (cycles_share), in percentage points: how wrong one request's slice of the
    #                batch can be. cycles_share is what CriticalPathTrim and the SLO policies
    #                threshold on, so this is the number that decides whether a policy can pick
    #                the wrong victim.
    pred = slope * nb + icpt
    worst_resid = float(np.max(np.abs(c - pred) / np.maximum(c, 1.0)))
    sm, sp = c / max(c.sum(), 1.0), pred / max(pred.sum(), 1.0)
    worst_share = float(np.max(np.abs(sm - sp))) * 100.0
    print(f"per-request attribution error: worst residual {worst_resid*100:.2f}% of a request's "
          f"cycles; worst cycles_share error {worst_share:.2f} percentage points "
          f"(shares are what policies threshold on)")
    import json as _j
    print("EKV_PAIRS " + _j.dumps({"blocks": nb.tolist(), "cycles": c.tolist()}))
    print(f"  row0 (nb=4): {c[0]:.0f} cyc   row{NQ-1} (nb={int(nb[-1])}): {c[-1]:.0f} cyc "
          f"(x{c[-1]/max(c[0],1):.1f})")

    # the join: request (keyed by its OWN first block) -> {blocks, cycles}; same row keys both buffers
    req = {int(rows[r, 0]): {"blocks": int((rows[r] > 0).sum()), "cycles": int(c[r])}
           for r in range(NQ)}
    join_ok = len(req) == NQ and all(v["cycles"] > 0 for v in req.values())
    show = {k: req[k] for k in sorted(req)[:3]}
    print(f"join (first-block anchor -> blocks+cycles): {len(req)} requests, e.g. {show}")

    ok = cap_exact and per_request and slope > 0 and r2 > 0.95 and mono > 0.98 and join_ok
    print("PASS: timer rows are PER-REQUEST under a q-block grid (QB CTAs/seq + heads accumulate "
          "into one row), cycles are linear in each request's planted context, and the capture and "
          "timer join on the same row -> per-request MEASURED attention compute, no model"
          if ok else
          f"FAIL: cap_exact={cap_exact} per_request={per_request} slope={slope:.0f} "
          f"r2={r2:.3f} corr={mono:.3f} join={join_ok}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
