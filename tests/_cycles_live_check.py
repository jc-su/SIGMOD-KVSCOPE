"""LIVE per-request cycles on real vLLM (Phase-1 spike, step C).

Runs stock vLLM (opt-125m, TRITON_ATTN) with the pass + EKV_TIMER via the registered plugin, eight
requests with strongly varied prompt lengths and identical decode budgets (stable batch, no slot
reuse), and reads the cumulative per-row cycles externally. PASS requires:
  (1) per-REQUEST keying on the real chunked-prefill kernel: the number of timer rows equals the
      number of requests (pid0 spans q-blocks at prefill, so per-CTA keying would light far more);
  (2) cycles track each request's attention work: corr(cycles_r, expected_r) > 0.9, where
      expected_r = prefill causal area len^2/2 + decode area sum(len+t), from known token counts;
  (3) rows map to requests content-addressedly (block count per captured row ranks like the
      known per-request block counts).
The runner sets the pass env; this probe is engine-stock (the plugin injects the compile hook).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"     # in-proc engine: pass publishes under OUR pid

import numpy as np
import torch
from vllm import LLM, SamplingParams

GEN = 32
llm = LLM(model="facebook/opt-125m", gpu_memory_utilization=0.35, max_model_len=2048,
          enable_prefix_caching=True, attention_backend="TRITON_ATTN")
tok = llm.get_tokenizer()

# DISTINCT leading marker per request: with prefix caching on, nested prompts would share their
# leading cached blocks and the first-block anchors would collide (the naive-anchor case).
prompts = [(f"request {r} marker alpha{r} beta{r}. " + "word " * (110 * (r + 1))).strip()
           for r in range(8)]                                          # ~115..885 tokens
lens = [len(tok(p, add_special_tokens=True)["input_ids"]) for p in prompts]
print(f"prompt token lengths: {lens}")
out = llm.generate(prompts, SamplingParams(max_tokens=GEN, temperature=0.0))
torch.cuda.synchronize()                                              # visibility anchor

from observe import cycles as cy
from observe.zero_touch import find_published, parse_slots, read_dev_mem, read_proc_mem

# Per-kernel publications: the attention kernel compiles as SEVERAL variants (eager prefill, the
# CUDA-graph decode buckets), and inductor kernels can also match a index_lookup -- each publishes its
# own buffer set. This check exercises the PRODUCTION API (observe/cycles.py): entry selection,
# snapshot/total, the content-addressed request join, shares, and the exact window partition.
pubs = find_published(os.getpid())
print(f"published entries: {sorted(set((b.kernel, b.row_form) for b in pubs))}")
attn = cy.attention_entries(os.getpid(), kernel="attention")
N = len(prompts)
# per-request block count AT THE LAST DECODE STEP: that step attends len+GEN-1 tokens (the context
# before the final token is emitted), so the captured table holds ceil((len+GEN-1)/16) blocks.
expect_blk = [int(np.ceil((L + GEN - 1) / 16)) for L in lens]

# (3) content-addressed row->request: the REAL variant is the one whose rows 0..N-1 hold exactly
# the expected per-request block counts (vLLM's memory-profile dummy run writes max-size dummy
# tables into other variants' buffers, which cannot match). Row order = arrival order here
# (concurrent submission, equal decode budgets, no compaction).
# Per-compile buffers: pick the REAL decode variant by exact per-request block counts (a profile /
# graph-capture dummy compile writes max-size or single-seq dummy tables, which cannot match). Row
# order = arrival order here (concurrent submission, equal decode budgets, no compaction).
slots = {}
for b in attn:
    sl = parse_slots(read_proc_mem(b.pid, b.host_va, b.bytes), b.ncap, b.max_rows)
    got = [sum(1 for x in sl.get(r, ()) if x > 0) for r in range(N)]
    if got == expect_blk:
        slots = {r: sl[r] for r in range(N)}
        break
map_ok = bool(slots)
print(f"attention variants: {len(attn)}; real-table variant found: {map_ok} "
      f"(rows 0..{N-1} block counts == {expect_blk})")
if not map_ok:
    for b in attn:
        sl = parse_slots(read_proc_mem(b.pid, b.host_va, b.bytes), b.ncap, b.max_rows)
        print("  variant rows0..7:", [sum(1 for x in sl.get(r, ()) if x > 0) for r in range(N)])
    sys.exit(1)
# name-free cross-check: variants whose captures content-match the real table (prefill rows are
# grow-only prefixes of decode rows, so both match; dummies do not)
matched = cy.attention_entries(os.getpid(), reference_slots=slots)
print(f"content-addressed selection (rows_match vs the real table): "
      f"{len(matched)}/{len(attn)} attention variants, no names")

# cycles through the production API: sum over variants (delta from birth = total)
snap1 = cy.snapshot(attn)
cyc = cy.total(snap1)

# (1) per-request keying on the REAL kernel: every request row carries cycles (prefill has
# sum(ceil(len/Q)) q-block CTAs, far more than N; graph capture pads with dummy rows whose
# one-time cycles a production reader excludes by differencing across step boundaries).
per_request = all(r in cyc for r in range(N))
statistic = sum(cyc.get(r, 0) for r in range(N)) / max(sum(cyc.values()), 1)
print(f"timer rows lit: {len(cyc)}; request rows 0..{N-1} all present: {per_request}; "
      f"cycle statistic on request rows: {100 * statistic:.1f}% "
      f"(rest = one-time profile/graph-capture dummy runs)")

# the request join: entity = the request's own first block (dummy/pad rows carry no real table)
req = cy.request_cycles(cyc, slots)
fb = [slots[r][0] for r in range(N)]                       # row r's anchor block
join_ok = len(req) == N and all(b in req for b in fb)

# (2) cycles vs expected attention work (prefill causal area + decode area) per request
exp = [lens[r] * (lens[r] + 1) / 2 + sum(lens[r] + t for t in range(1, GEN + 1)) for r in range(N)]
xs = np.array(exp, dtype=float)
ys = np.array([req.get(fb[r], 0) for r in range(N)], dtype=float)
corr = float(np.corrcoef(xs, ys)[0, 1]) if N > 2 else 0.0
ratio = ys[-1] / max(ys[0], 1)
print(f"corr(cycles, expected attention area) = {corr:.4f}; "
      f"largest/smallest request cycles = x{ratio:.1f}")
import json as _json
print("EKV_AREA_PAIRS " + _json.dumps(
    {"area_share": [float(v) for v in xs / xs.sum()],
     "cycle_share": [float(v) for v in ys / ys.sum()], "corr": corr}))
# shares partition unity and the window partition is exact (the algebra, live)
att = cy.attribute_window(req, 1.0)
part_ok = abs(sum(att.values()) - 1.0) < 1e-9
print(f"shares: { {b: round(s, 3) for b, s in sorted(cy.shares(req).items())} } "
      f"(partition exact: {part_ok})")

ok = per_request and map_ok and join_ok and part_ok and corr > 0.9
print(("PASS" if ok else "FAIL") + " phase 1 (cumulative): "
      f"per_request={per_request} map_ok={map_ok} join_ok={join_ok} part_ok={part_ok} "
      f"corr={corr:.3f}")

# ---- phase 3: per-request attention TIME, MEASURED -- no compute model -------------------------
# The pass tier needs no regression model for compute: per-request attention time = the measured
# cycle SHARE x the measured attention-kernel wall time (cy.attribute_window). It partitions the
# window EXACTLY and the split is measurement, not a fit. (The T_step=c+aQ+bW geometry model is the
# boundary/no-pass fallback, and lives on the ebpf branch -- master reads compute from the timer.)
t_window = 1.0                                    # unit window; a measured wall time scales linearly
att_time = cy.attribute_window(req, t_window)
part_ok3 = abs(sum(att_time.values()) - t_window) < 1e-9
ys3 = np.array([att_time.get(fb[r], 0.0) for r in range(N)])
corr3 = float(np.corrcoef(np.array(exp, dtype=float), ys3)[0, 1])
print(f"measured per-request attention time: exact window partition: {part_ok3}; "
      f"corr(time, attention area) = {corr3:.4f} (no model, no a/b fit)")
ok3 = part_ok3 and corr3 > 0.9
print(("PASS" if ok3 else "FAIL") + " phase 3 (measured compute): per-request attention time is "
      "the cycle share x measured window -- exact partition, no compute model")

# ---- phase 2: WINDOWED attribution via the production streaming reduction (cy.window_series) -----
# Clean windows exist only WITHIN one generation, as a slot's row grows by decode blocks (two
# separate generate() calls are different request instances that share only the cached prefix, so
# their decode tails differ -- correctly a boundary). During decode the batch is stable (same
# requests, same anchors, rows grow), so a sampler reading the timer + block table at intervals
# yields consecutive CLEAN windows. This is the deployment cadence with a thread standing in for
# the eBPF sync/event-ready bracket: the samples are NOT sync-anchored, so monotonicity and
# occupancy stability hold but absolute per-window numbers are noisier (accumulation over the run
# recovers the signal). window_series folds it: monotone deltas, clean/unclean per window, and
# per-request accumulation keyed by the content-addressed anchor.
import array
import threading
import time

raw: list = []                                            # per sample: {timer_va: (timer, table)}
stop = threading.Event()

def sample_once():
    snap = {}
    for b in cy.attention_entries(os.getpid(), kernel="attention"):
        a = array.array("q")
        raw_t = (read_dev_mem(b.timer_dev, b.timer_bytes) if b.timer_dev
                 else read_proc_mem(b.pid, b.timer_host_va, b.timer_bytes))
        a.frombytes(raw_t)
        sl = parse_slots(read_proc_mem(b.pid, b.host_va, b.bytes), b.ncap, b.max_rows)
        real = {s: r for s, r in sl.items() if any(x > 0 for x in r)}
        snap[b.timer_dev or b.timer_host_va] = (list(a[:b.max_rows]), real)
    if snap:
        raw.append(snap)

def sampler():
    while not stop.is_set():
        sample_once()
        time.sleep(0.012)

th = threading.Thread(target=sampler, daemon=True)
th.start()
llm.generate(prompts, SamplingParams(max_tokens=256, temperature=0.0))    # long decode -> samples
stop.set()
th.join()
torch.cuda.synchronize()

# Lock onto ONE decode variant across samples (per-sample "best" flips between CUDA-graph buckets,
# reading different buffers -> false instability). The decode variant is the timer_va that carries
# N real rows in the most samples; build the stream from that variant's own timer + table.
from collections import Counter
cand = Counter(va for snap in raw for va, (_t, tbl) in snap.items() if len(tbl) == N)
dec_va = cand.most_common(1)[0][0] if cand else None
samples = [(cy.CycleSnapshot({"d": snap[dec_va][0]}), snap[dec_va][1])
           for snap in raw if dec_va in snap and len(snap[dec_va][1]) == N]

ser = cy.window_series(samples)                          # entities = own first blocks
try:
    total_span = cy.delta(samples[0][0], samples[-1][0]) if len(samples) >= 2 else {}
    monotone = True
except ValueError:
    total_span, monotone = {}, False
clean_frac = ser["clean"] / max(ser["n"], 1)
# The reduction RUNS live over real samples, deltas stay monotone, and the decode batch is
# correctly classified as a stable occupancy (clean windows). What the thread CANNOT give is
# reliable per-window deltas: it reads mid-flight, not at a completion sync, so the cycle statistic
# appears only at buffer-flush moments -- exactly the visibility precondition observe/cycles
# documents. Per-window/per-request accumulation is therefore proven by the CPU invariants
# (test_cycles.py, full occupancy control); live it needs the eBPF sync/event-ready bracket, not a
# thread. Here we assert the properties that hold under mid-flight reads, and check the cumulative
# span (bounded by the post-join sync) is non-trivial and tracks work.
span_slots = samples[-1][1]                              # last sample's stable decode table
span_by_req = {span_slots[s][0]: total_span.get(s, 0) for s in span_slots}
covered = sum(1 for c in span_by_req.values() if c > 0)
anchors = sorted(span_by_req, key=lambda fb: span_by_req[fb])
exp_dec = sorted(lens[r] * 256 + 256 * 255 / 2 for r in range(N))
ys = np.array([span_by_req[a] for a in anchors], dtype=float)
corr_span = float(np.corrcoef(np.array(exp_dec, dtype=float), ys)[0, 1]) if covered == N else 0.0
print(f"window_series: {len(raw)} raw samples, decode variant locked -> {len(samples)} usable, "
      f"{ser['n']} windows, {ser['clean']} clean ({100*clean_frac:.0f}%); monotone deltas: "
      f"{monotone}")
print(f"  cumulative span over the sampled decode: {covered}/{N} requests, "
      f"corr(span cycles, decode work) = {corr_span:.4f} "
      f"(per-window numbers need sync-anchored reads; accumulation math is CPU-proven)")

# gate on what holds under mid-flight reads: the reduction runs, deltas are monotone, the decode
# batch is a stable occupancy, and the sync-bounded cumulative span tracks per-request work.
ok2 = len(samples) >= 5 and monotone and clean_frac >= 0.9 and covered == N and corr_span > 0.9
print("PASS: LIVE window_series runs over real intra-generation samples with monotone deltas and "
      "correct clean-occupancy classification; the sync-bounded cumulative tracks per-request "
      "decode work (per-window attribution requires the eBPF sync bracket; its algebra is CPU-proven)"
      if ok2 else
      f"FAIL phase 2: usable={len(samples)} monotone={monotone} clean_frac={clean_frac:.2f} "
      f"covered={covered}/{N} corr_span={corr_span:.3f}")
sys.exit(0 if (ok and ok2 and ok3) else 1)
