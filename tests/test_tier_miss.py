"""CI (no GPU): the storage-tier miss cost = disk_read + H2D, recovered by fusing the eBPF
storage_read probe with the H2D copy stream. The cudaMemcpy data plane sees ONLY the H2D, so a
disk-tier miss (slow read THEN fast copy) looks as cheap as a CPU-tier hit (copy only). Joining
each read to the H2D it feeds makes the previously-invisible read a measured BOUNDED quantity and
exposes the tier heterogeneity the copy-only view hides.

Two backend threading models are covered: SYNCHRONOUS (read+H2D on one thread -> exact same-thread
pairing) and the ASYNC EXECUTOR (read on a worker thread, H2D on the engine thread -> nearest-
following time-proximity). LMCache's local-disk backend has both paths; the join handles both.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from observe import BOUNDED, tier_miss_costs   # noqa: E402

US = 1_000
MS = 1_000_000
KV = 9 << 20   # ~9 MiB KV chunk


def test_synchronous_same_thread_join():
    # blocking backend path: read THEN H2D on the SAME engine thread (tid 100). Two disk-tier
    # reloads (slow 5/6ms read + fast 200us copy) and one CPU-tier hit (H2D only, no read).
    reads = [{"tid": 100, "ts_ns": 1 * MS, "bytes": KV, "dur_ns": 5 * MS},
             {"tid": 100, "ts_ns": 10 * MS, "bytes": KV, "dur_ns": 6 * MS}]
    h2d = [{"tid": 100, "enqueued_at": 1 * MS + 100 * US, "cost_ns": 200 * US},   # feeds read1
           {"tid": 100, "enqueued_at": 10 * MS + 100 * US, "cost_ns": 200 * US},  # feeds read2
           {"tid": 100, "enqueued_at": 20 * MS, "cost_ns": 200 * US}]            # CPU-tier hit
    r = tier_miss_costs(reads, h2d)
    paired = [p for p in r["pairs"] if p["h2d_cost_ns"] is not None]
    assert len(paired) == 2 and all(p["join"] == "same-thread" for p in paired)
    assert all(p["provenance"] == BOUNDED for p in paired)            # read exact, pairing bounded
    assert paired[0]["total_ns"] == 5 * MS + 200 * US                 # disk_read + H2D
    a = r["aggregate"]
    assert a["n_same_thread"] == 2 and a["n_paired"] == 2
    assert a["n_h2d_cpu_tier"] == 1                                   # the 20ms H2D has no read
    # the headline: the INVISIBLE disk read dominates the H2D copy that was the only visible part
    assert a["read_dominance"] >= 20
    # the disk-tier miss (read+H2D ~5ms) is an order of magnitude over the CPU-tier hit (~200us)
    assert a["median_miss_ns"] >= 10 * a["median_h2d_ns"]


def test_async_executor_cross_thread_join():
    # async path: read on a worker thread (tid 200), H2D on the engine thread (tid 100). No
    # same-tid match -> nearest-following time-proximity (still recovers the miss cost, bounded).
    reads = [{"tid": 200, "ts_ns": 1 * MS, "bytes": KV, "dur_ns": 5 * MS}]
    h2d = [{"tid": 100, "enqueued_at": 1 * MS + 300 * US, "cost_ns": 200 * US}]
    r = tier_miss_costs(reads, h2d)
    p = r["pairs"][0]
    assert p["join"] == "time-proximity" and p["total_ns"] == 5 * MS + 200 * US
    assert r["aggregate"]["n_same_thread"] == 0 and r["aggregate"]["n_paired"] == 1


def test_h2d_before_read_is_not_paired():
    # an H2D enqueued BEFORE the read completes cannot be the copy that read feeds.
    reads = [{"tid": 100, "ts_ns": 10 * MS, "bytes": KV, "dur_ns": 5 * MS}]
    h2d = [{"tid": 100, "enqueued_at": 2 * MS, "cost_ns": 200 * US}]   # earlier than read completion
    r = tier_miss_costs(reads, h2d)
    assert r["pairs"][0]["join"] == "unpaired" and r["pairs"][0]["h2d_cost_ns"] is None
    assert r["aggregate"]["n_h2d_cpu_tier"] == 1                       # the H2D stays an upper-tier hit


def test_all_cpu_tier_no_reads():
    # no disk reads at all (everything served from the CPU tier / pool): copy-only, no miss cost.
    h2d = [{"tid": 100, "enqueued_at": i * MS, "cost_ns": 200 * US} for i in range(3)]
    r = tier_miss_costs([], h2d)
    a = r["aggregate"]
    assert a["n_reads"] == 0 and a["n_paired"] == 0 and a["n_h2d_cpu_tier"] == 3
    assert a["read_dominance"] is None


if __name__ == "__main__":
    test_synchronous_same_thread_join()
    test_async_executor_cross_thread_join()
    test_h2d_before_read_is_not_paired()
    test_all_cpu_tier_no_reads()
    print("PASS: tier miss cost = disk_read + H2D; same-thread (sync) and time-proximity (async) "
          "joins; disk read dominates the visible H2D; CPU-tier hits separate cleanly.")
