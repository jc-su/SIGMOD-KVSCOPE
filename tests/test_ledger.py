"""CPU unit tests for joining kernel observations with eBPF lifecycle records.
The sources join through physical KV-block ownership and degrade gracefully without eBPF.
No GPU/root."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from observe.ledger import KVLedger                                    # noqa: E402
from observe.events import KVEdge                                      # noqa: E402
from observe.types import Op, Tier                                     # noqa: E402


def _copy(block, src, dst, nbytes=1000, cost=5000):
    return KVEdge(op=Op.COPY, block_id=block, content_hash=None, prefix_hash=None,
                  src_tier=src, dst_tier=dst, nbytes=nbytes, cost=cost)


def test_pass_only_is_graceful():
    """Without eBPF input, every lifecycle field is 0/False."""
    led = KVLedger()
    led.ingest_pass({7: {7, 8, 9}}, cycles_by_anchor={7: 1234},
                    attention_weight_by_request={7: 0.5})
    f = led.features(7)
    assert f.pop("age_s") >= 0.0                    # pass-only records still have request age
    assert f == {"reload_bytes": 0, "recompute": 0, "reload_latency_ms": 0.0,
                 "was_offloaded": False,
                 "lifecycle_observed": False,   # no lifecycle window yet: unknown, not known-zero
                 "lifecycle_stale": False}      # ...and not stale either: nothing has gone blind
    r = led.records[7]
    assert r.n_blocks == 3 and r.cycles == 1234 and r.attention_weight == 0.5


def test_stale_is_now_not_ever():
    """`lifecycle_observed` latches on the first complete window and can never fall again, so it
    cannot tell a policy the tracer has since gone blind. `lifecycle_stale` is the current window,
    which is what a live decision needs."""
    led = KVLedger()
    led.ingest_pass({7: {7, 8}})
    led.ingest_ebpf([], {}, complete=True)
    assert led.features(7)["lifecycle_observed"] and not led.features(7)["lifecycle_stale"]
    led.ingest_ebpf([], {}, complete=False)                 # coverage lost mid-run
    f = led.features(7)
    assert f["lifecycle_observed"] and f["lifecycle_stale"]  # still "ever", but blind NOW
    led.ingest_ebpf([], {}, complete=True)
    assert not led.features(7)["lifecycle_stale"]           # and it recovers


def test_reload_folds_to_owning_anchor():
    """A DRAM->HBM copy of a block owned by anchor 7 becomes anchor 7's reload cost."""
    led = KVLedger()
    led.ingest_pass({7: {7, 8}}, cycles_by_anchor={7: 10})
    led.ingest_ebpf([_copy(8, Tier.DRAM, Tier.HBM, nbytes=2048, cost=380_000)],
                    block_to_anchors={8: {7}})
    r = led.records[7]
    assert r.reload_bytes == 2048 and r.reload_ns == 380_000
    assert abs(led.features(7)["reload_latency_ms"] - 0.38) < 1e-9
    assert led.features(7)["was_offloaded"] is False


def test_offload_and_recompute_classified():
    led = KVLedger()
    led.ingest_pass({5: {5}, 6: {6}}, cycles_by_anchor={5: 1, 6: 1})
    led.ingest_ebpf([_copy(5, Tier.HBM, Tier.DRAM, nbytes=4096),               # offload
                     _copy(6, Tier.RECOMPUTE, Tier.HBM, nbytes=0, cost=0)],    # recompute
                    block_to_anchors={5: {5}, 6: {6}})
    assert led.records[5].offload_bytes == 4096 and led.features(5)["was_offloaded"] is True
    assert led.records[6].recompute == 1 and led.records[5].recompute == 0


def test_shared_block_attributes_to_all_owners():
    """A reused prefix block resolves to every owning anchor -- shared movement is a shared cost."""
    led = KVLedger()
    led.ingest_pass({1: {1, 100}, 2: {2, 100}}, shared_anchors=[1, 2])   # block 100 shared
    led.ingest_ebpf([_copy(100, Tier.DRAM, Tier.HBM, nbytes=512, cost=1000)],
                    block_to_anchors={100: {1, 2}})
    assert led.records[1].reload_bytes == 512 and led.records[2].reload_bytes == 512


def test_edge_owning_no_live_block_dropped():
    led = KVLedger()
    led.ingest_pass({1: {1}})
    led.ingest_ebpf([_copy(999, Tier.DRAM, Tier.HBM)], block_to_anchors={})   # unknown block
    assert led.records[1].reload_bytes == 0


def test_gone_anchor_dropped_on_refresh():
    """The ledger tracks the LIVE request set: an anchor absent from a refresh is dropped."""
    led = KVLedger()
    led.ingest_pass({1: {1}, 2: {2}})
    assert set(led.records) == {1, 2}
    led.ingest_pass({1: {1}})                                            # 2 finished
    assert set(led.records) == {1}


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))


