"""CPU test for the eBPF lifecycle tracing runtime bridge (observe/ebpf_source): a synthetic kv_copy JSONL stream,
tailed incrementally, resolves to block-costed KVEdges via the pool geometry. No GPU/root."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from observe.ebpf_source import make_ebpf_source  # noqa: E402

BASE, NBLK, STRIDE = 0x100000, 16, 0x1000           # 16 blocks of 4 KiB


def test_stream_resolves_to_block_edges_and_tails(tmp_path):
    path = str(tmp_path / "kv_copy.jsonl")
    # block 2 offloaded (D2H: HBM->DRAM), block 5 reloaded (H2D: DRAM->HBM)
    src2 = BASE + 2 * STRIDE
    dst5 = BASE + 5 * STRIDE
    with open(path, "w") as f:
        f.write(f'{{"type":"enqueue","t_ns":100,"kind":"D2H","stream":1,"src":{src2},"dst":9,"bytes":4096}}\n')
        f.write('{"type":"completion","via":"stream-sync","t_ns":600,"n":1,"handle":1}\n')
    src = make_ebpf_source(path, BASE, NBLK, STRIDE)
    batch = src()
    edges = list(batch)
    assert batch.complete and len(edges) == 1
    e = edges[0]
    assert e.block_id == 2 and int(e.nbytes) == 4096
    # a second call with no new data yields nothing (tail advanced) -- and COMPLETE
    b2 = src()
    assert b2.complete and list(b2) == []
    # append a reload; only the new record is returned
    with open(path, "a") as f:
        f.write(f'{{"type":"enqueue","t_ns":700,"kind":"H2D","stream":2,"src":9,"dst":{dst5},"bytes":8192}}\n')
        f.write('{"type":"completion","via":"stream-sync","t_ns":1200,"n":1,"handle":2}\n')
    edges2 = list(src())
    assert len(edges2) == 1 and edges2[0].block_id == 5 and int(edges2[0].nbytes) == 8192


def test_missing_stream_is_graceful():
    src = make_ebpf_source("/nonexistent/kv_copy.jsonl", BASE, NBLK, STRIDE)
    b = src()                                           # never raises -> loop stays pass-only
    assert list(b) == [] and b.complete is False        # ...but the hole is REPORTED, not silent


def test_partial_trailing_line_held(tmp_path):
    path = str(tmp_path / "kv.jsonl")
    with open(path, "w") as f:
        f.write('{"type":"enqueue","t_ns":1,"kind":"D2H","stream":1,"src":')  # partial, no newline
    src = make_ebpf_source(path, BASE, NBLK, STRIDE)
    b = src()
    assert list(b) == [] and b.complete                 # incomplete line held back, not a gap


# ── coverage-first eBPF lifecycle tracing (the not-separate construction) ─────────────────────────────────

def test_batch_coverage_and_ledger_observed_flag():
    """unknown != known-zero: features carry lifecycle_observed only after a COMPLETE window."""
    from observe.ebpf_source import ObservationBatch
    from observe.ledger import KVLedger
    led = KVLedger()
    led.ingest_pass({7: {7, 8, 9}})
    assert led.features(7)["lifecycle_observed"] is False
    led.ingest_ebpf(ObservationBatch(complete=False, reason="stream not up"), {}, complete=False)
    assert led.features(7)["lifecycle_observed"] is False and led.lifecycle_gaps == 1
    led.ingest_ebpf(ObservationBatch(), {}, complete=True)  # plane watched, saw nothing
    assert led.features(7)["lifecycle_observed"] is True    # NOW zeros mean zero
    assert led.lifecycle_windows == 1


def test_derive_geometry_region_merge_and_lattice_ids():
    """The measured layout, in miniature: VA-adjacent allocations merge into regions; pooled
    regions hold several tensors, so ids are exact ONLY on the published kv_ptr lattice (region
    offsets mis-number). A block moved in another tensor is off-lattice: no id, never a wrong
    one."""
    from observe.blockmap import BlockAddressMap
    from observe.ebpf_source import derive_geometry, merge_regions
    allocs = [{"addr": 0x1000, "size": 0x3000},             # decoy region (different size)
              {"addr": 0x10000, "size": 0x800},             # two adjacent allocs -> ONE region
              {"addr": 0x10800, "size": 0x800},             #   [0x10000, 0x11000)
              {"addr": 0x30000, "size": 0x1000}]            # same-size region: family member
    assert merge_regions(allocs)[1] == (0x10000, 0x11000)
    # kv_ptr = a tensor base at +0x800 INSIDE the merged region (not the region base)
    copies = [{"bytes": 0x100, "dst": 0x10800, "src": 0x99999},   # lattice id 0... but see below
              {"bytes": 0x100, "dst": 0x10900, "src": 0x99999},   # lattice id 1
              {"bytes": 0x100, "dst": 0x10100, "src": 0x99999},   # other tensor: off-lattice
              {"bytes": 0x100, "dst": 0x30200, "src": 0x99999}]   # family region traffic
    geo, why = derive_geometry(allocs, kv_ptr=0x10800, copy_records=copies)
    assert geo is not None, why
    regions, stride, kvp = geo
    assert regions[0] == (0x10000, 0x11000) and stride == 0x100 and kvp == 0x10800
    assert (0x30000, 0x31000) in regions                    # same-size region joins the family
    nblk = (regions[0][1] - kvp) // stride
    amap = BlockAddressMap.from_alloc(kvp, nblk, stride)
    assert amap.resolve(0x10900).block_id == 1              # exact id on the lattice
    assert amap.resolve(0x10100) is None                    # other tensor: no id, not a wrong id
    geo2, why2 = derive_geometry(allocs, kv_ptr=0x5555, copy_records=copies)
    assert geo2 is None and "pool not identified" in why2
    geo3, why3 = derive_geometry(allocs, kv_ptr=0x10800, copy_records=[])
    assert geo3 is None and "calibrating" in why3


def test_stride_rejected_when_nothing_lands_on_the_lattice():
    """The check on the modal-size heuristic: a stride no pool transfer lands on (relative to
    the published kv_ptr) is declined loudly -- wrong identity is worse than no identity."""
    from observe.ebpf_source import derive_geometry
    allocs = [{"addr": 0x10000, "size": 0x1000}]
    off = [{"bytes": 0x180, "dst": 0x10123, "src": 0x99999},
           {"bytes": 0x180, "dst": 0x10457, "src": 0x99999}]
    geo, why = derive_geometry(allocs, kv_ptr=0x10000, copy_records=off)
    assert geo is None and "rejected" in why


def test_manual_source_reports_stream_not_up(tmp_path):
    from observe.ebpf_source import make_ebpf_source
    src = make_ebpf_source(str(tmp_path / "absent.jsonl"), 0x1000, 16, 0x100)
    b = src()
    assert b.complete is False and "stream not up" in b.reason and list(b) == []


def test_capture_integrity_reaches_the_batch(tmp_path):
    """An overflowed completion (ringbuf drops) must mark the WINDOW incomplete -- the reader's
    diagnostics were previously discarded and overflow returned complete=True."""
    path = str(tmp_path / "kv.jsonl")
    s2 = BASE + 2 * STRIDE
    with open(path, "w") as f:
        f.write(f'{{"type":"enqueue","t_ns":100,"kind":"D2H","stream":1,"src":{s2},"dst":9,"bytes":4096}}\n')
        f.write('{"type":"completion","via":"stream-sync","t_ns":600,"n":5,"handle":1}\n')  # n=5 > captured
    src = make_ebpf_source(path, BASE, NBLK, STRIDE)
    b = src()
    assert b.complete is False and "overflowed" in b.reason


def test_late_attach_fallback_bootstraps_from_copies_alone():
    """The daemon's normal situation: kv_alloc attached AFTER the pool cudaMalloc, so no captured
    region contains kv_ptr. The lattice bootstraps from lattice-landing traffic; too little
    traffic declines instead of guessing. Strides are BLOCK-sized (>= 4KiB): a per-block KV span
    is KBs at any block_size, and the floor exists because scalar copies divide every address."""
    from observe.ebpf_source import derive_geometry
    kvp = 0x40000
    S = 0x10000                                              # 64 KiB: a real per-block span
    copies = [{"bytes": S, "dst": kvp + k * S, "src": 0x9} for k in (1, 2, 5, 7, 9)]
    geo, why = derive_geometry([], kv_ptr=kvp, copy_records=copies)   # NO allocations captured
    assert geo is not None, why
    regions, stride, k = geo
    assert stride == S and k == kvp
    assert regions[0][0] == kvp and regions[0][1] >= kvp + 10 * S
    geo2, why2 = derive_geometry([], kv_ptr=kvp,
                                 copy_records=copies[:2])             # too little to trust
    assert geo2 is None and "too little" in why2


def test_scalar_copies_never_win_the_stride_election():
    """Regression for a live geometry-blindness: 8-byte flag copies outnumber (and trivially
    divide every address, so out-'hit') the true 64KiB block stride -- the old most-hits vote
    elected stride=8 and an 8B liveloop with 95K genuine reloads derived nothing. Sub-4KiB
    sizes are structurally not block moves and must be excluded from candidacy."""
    from observe.ebpf_source import derive_geometry
    kvp = 0x40000
    S = 0x10000
    copies = ([{"bytes": 8, "dst": kvp + i * 8, "src": 0x9} for i in range(500)] +   # scalars
              [{"bytes": S, "dst": kvp + k * S, "src": 0x9} for k in range(1, 40)])  # real blocks
    geo, why = derive_geometry([], kv_ptr=kvp, copy_records=copies)
    assert geo is not None, why
    _, stride, _ = geo
    assert stride == S, f"scalars won the election again (stride={stride})"


def test_meta_records_share_the_stream_without_voiding_windows(tmp_path):
    """EKV_META=1 interleaves identity-plane {\"type\":\"meta\"} records into the copy stream
    (the daemon's auto attach always sets it). A meta line must not void the window: the live
    late-attach artifact showed ~40 windows of copies dropped as 'unknown copy record type
    meta' before this was known to the reader."""
    path = str(tmp_path / "kv.jsonl")
    s2 = BASE + 2 * STRIDE
    with open(path, "w") as f:
        f.write('{"type":"meta","t_ns":50,"bytes":64,"dst":123,"src":456}\n')
        f.write(f'{{"type":"enqueue","t_ns":100,"kind":"D2H","stream":1,"src":{s2},"dst":9,"bytes":4096}}\n')
        f.write('{"type":"meta","t_ns":150,"bytes":64,"dst":123,"src":456}\n')
        f.write('{"type":"completion","via":"stream-sync","t_ns":600,"n":1,"handle":1}\n')
    src = make_ebpf_source(path, BASE, NBLK, STRIDE)
    b = src()
    assert b.complete and b.parse_errors == 0, b.reason
    edges = list(b)
    assert len(edges) == 1 and edges[0].block_id == 2, "the copy between meta lines must survive"
