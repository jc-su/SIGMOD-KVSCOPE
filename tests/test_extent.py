"""eBPF lifecycle tracing coverage: the address map's extent is a lower bound, and a short one must be
loud and self-correcting rather than silently lossy.

Regression for a measured failure. On a committed 8B live-loop run the daemon's late-attach
calibration sized the lattice from early traffic, froze at 82 KV units while the engine's pool
used 300, and dropped every copy to the other 218 for the rest of the run -- with the geometry
self-checks (id 0 absent, ids on the lattice) all still passing, and nothing in the health file
saying so. Three properties pin the fix:

  1. a copy on the pool's lattice but past the extent is COUNTED as such, not silently skipped
     (and is distinguishable from traffic that is simply not this pool's);
  2. the extent WIDENS to traffic that proves the pool is bigger, in the same window;
  3. correlation state survives across windows, so a copy completing in the next window is not
     reported as an unmatched completion.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from observe.blockmap import BlockAddressMap                              # noqa: E402
from observe.ebpf_reader import Correlator, read_copies_report            # noqa: E402
from observe.ebpf_source import ObservationBatch, _TailFuser, scan_max_unit  # noqa: E402
from observe.events import Copy                                          # noqa: E402
from observe.fuse import fuse_copies                                     # noqa: E402
from observe.ledger import KVLedger                                      # noqa: E402

BASE, STRIDE = 0x7F0000000000, 65536


def _copy(unit, nbytes=STRIDE):
    """A D2H offload of one KV unit: the device side is the pool address."""
    return Copy(addr=BASE + unit * STRIDE, nbytes=nbytes, direction="D2H",
                enqueued_at=1000 + unit, finished_at=2000 + unit, via="stream-sync")


def test_lattice_unit_sees_past_the_extent():
    m = BlockAddressMap.from_alloc(BASE, 4, STRIDE)
    assert m.extent == 4
    assert m.lattice_unit(BASE + 2 * STRIDE) == 2        # inside
    assert m.lattice_unit(BASE + 9 * STRIDE) == 9        # OUTSIDE, still nameable -> fixable
    assert m.resolve(BASE + 9 * STRIDE) is None          # ...which resolve alone cannot say
    assert m.lattice_unit(BASE + 9 * STRIDE + 17) is None   # off-lattice: not ours
    assert m.lattice_unit(BASE - STRIDE) is None            # below the pool: not ours


def test_drops_are_accounted_and_split_by_cause():
    m = BlockAddressMap.from_alloc(BASE, 4, STRIDE)
    stray = Copy(addr=0x11000000, nbytes=4096, direction="D2H", enqueued_at=1, finished_at=2,
                 via="stream-sync")
    diag: dict = {}
    edges = fuse_copies([_copy(1), _copy(7), _copy(9), stray], m, diag=diag)
    assert len(edges) == 1                               # only unit 1 is inside the extent
    assert diag["beyond_extent"] == 2                    # units 7 and 9: REAL, nameable loss
    assert diag["max_unit"] == 9                         # ...and how far to widen
    assert diag["unresolved"] == 1                       # the stray: correctly ignored
    assert fuse_copies([_copy(7)], m) == []              # diag stays opt-in; no caller breaks


def test_short_extent_marks_the_window_incomplete():
    """The ledger must not count a truncated window as a clean observation."""
    led = KVLedger()
    led.ingest_ebpf(ObservationBatch(edges=[], complete=False, beyond_extent=3), {}, complete=False)
    assert led.lifecycle_gaps == 1 and led.lifecycle_windows == 0
    assert led.features(1)["lifecycle_stale"] is True    # blind RIGHT NOW, not merely ever-blind
    led.ingest_ebpf(ObservationBatch(edges=[]), {}, complete=True)
    assert led.features(1)["lifecycle_stale"] is False


def _enq(unit, t):
    return ('{"type":"enqueue","t_ns":%d,"pid":7,"tid":7,"bytes":%d,"kind":"D2H",'
            '"api":"cudaMemcpyAsync","stream":55,"dst":%d,"src":%d}'
            % (t, STRIDE, BASE + unit * STRIDE, BASE + unit * STRIDE))


def test_scan_finds_the_widening_target_without_a_full_parse():
    assert scan_max_unit([_enq(3, 10), _enq(11, 20)], BASE, STRIDE) == 11
    assert scan_max_unit([_enq(3, 10)], BASE + 17, STRIDE) == -1     # nothing on THIS lattice
    assert scan_max_unit([], BASE, STRIDE) == -1


def test_widening_reads_only_records_that_are_kv_traffic():
    """The identity plane's `meta` records (the input_ids H2D copy) share this stream and their
    `dst` is not a KV address. One landing on the lattice by coincidence must not stretch the
    extent to a unit that does not exist."""
    meta = '{"type":"meta","t_ns":1,"dst":%d,"size":256,"vals":[1,2]}' % (BASE + 9999 * STRIDE)
    assert scan_max_unit([meta], BASE, STRIDE) == -1
    assert scan_max_unit([meta, _enq(5, 10)], BASE, STRIDE) == 5     # real traffic still counts


def test_movement_owned_by_no_live_request_is_counted_not_swallowed():
    """The last drop in the fusion path. Legitimate (a block whose request just finished), so it
    does not fail the window -- but it is counted, because unaccounted drops are exactly how the
    extent bug hid."""
    from observe.events import KVEdge
    from observe.types import Op, Tier
    led = KVLedger()
    led.ingest_pass({7: {7, 8}})
    edge = lambda b: KVEdge(op=Op.COPY, block_id=b, content_hash=None, prefix_hash=None,
                            src_tier=Tier.DRAM, dst_tier=Tier.HBM, nbytes=2048, cost=1)
    led.ingest_ebpf([edge(8), edge(4242)], {8: {7}}, complete=True)
    assert led.records[7].reload_bytes == 2048          # the owned one lands
    assert led.unowned_edges == 1                       # the orphan is VISIBLE
    assert led.lifecycle_windows == 1                   # ...without falsifying the window


def test_correlation_survives_a_window_boundary():
    """An enqueue in window N completing in window N+1 must not read as unmatched."""
    done = '{"type":"completion","t_ns":99,"handle":55,"via":"stream-sync","n":1}'
    stateless_a = read_copies_report([_enq(1, 10)])
    stateless_b = read_copies_report([done])
    assert stateless_b.unmatched_completions == 1       # the old behaviour: a false gap
    assert not stateless_b.complete
    assert stateless_a.complete

    c = Correlator()
    win_a = read_copies_report([_enq(1, 10)], c)
    win_b = read_copies_report([done], c)
    assert win_a.complete and win_b.complete           # carried state: no phantom incompleteness
    assert win_b.unmatched_completions == 0
    assert read_copies_report([done], c).unmatched_completions == 0   # idle re-sync: still fine


def test_tail_fuser_reports_the_hole_and_names_the_extent():
    tf = _TailFuser("/nonexistent")
    m = BlockAddressMap.from_alloc(BASE, 4, STRIDE)
    b = tf.fuse([_enq(1, 10), _enq(9, 20)], m)
    assert b.beyond_extent == 1 and b.max_unit == 9
    assert not b.complete and "beyond the derived extent" in b.reason
    assert "covers 4" in b.reason                       # the operator learns BOTH numbers
