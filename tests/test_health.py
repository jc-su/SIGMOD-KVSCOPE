"""health.py must reflect every kv_copy loss mode: ring buffer drops (copy records and metadata),
batch truncation, and batch read faults. Each one must make ``ok`` False."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from observe import collect_health, parse_collector_report, parse_ringbuf_drops
from observe.health import CollectorLoss

NO_LOSS = CollectorLoss(copy_drops=0, batch_truncated=0, batch_read_faults=0, meta_drops=0)

# The exact stderr report the kv_copy loader prints at a clean exit.
REPORT = (
    "eKV kv_copy collector running (jsonl, ring buffer 64 MB, Ctrl-C to stop)\n"
    "ringbuf drops (copy records): 12\n"
    "batch truncation (blocks beyond cap): 34\n"
    "batch read faults (chunk skipped): 5\n"
    "ringbuf drops (metadata): 6\n"
)


def test_clean_is_ok() -> None:
    h = collect_health(edges=5, requests=3)
    assert h.ok and h.lost == 0


@pytest.mark.parametrize(
    "loss",
    [
        CollectorLoss({**NO_LOSS, "copy_drops": 7}),
        CollectorLoss({**NO_LOSS, "batch_truncated": 7}),
        CollectorLoss({**NO_LOSS, "batch_read_faults": 7}),
        CollectorLoss({**NO_LOSS, "meta_drops": 7}),
    ],
)
def test_each_loss_mode_flips_ok(loss: CollectorLoss) -> None:
    h = collect_health(**loss)
    assert not h.ok, f"{loss} must flip ok=False"
    assert h.lost == 7, f"{loss} must count toward lost"


def test_lost_sums_all_modes() -> None:
    h = collect_health(copy_drops=1, batch_truncated=2, batch_read_faults=3, meta_drops=4)
    assert h.lost == 10 and not h.ok
    d = h.as_dict()
    assert d["ok"] is False and d["lost"] == 10


def test_parse_collector_report() -> None:
    kw = parse_collector_report(REPORT)
    assert kw == {"copy_drops": 12, "batch_truncated": 34, "batch_read_faults": 5, "meta_drops": 6}
    h = collect_health(**kw)
    assert not h.ok and h.lost == 57


def test_parse_keeps_copy_and_metadata_drops_apart() -> None:
    # both lines start with "ringbuf drops"; each must map to its own counter
    kw = parse_collector_report(REPORT.replace(": 12\n", ": 0\n"))
    assert kw["copy_drops"] == 0 and kw["meta_drops"] == 6


@pytest.mark.parametrize("line", REPORT.splitlines()[1:])
def test_parse_missing_line_raises(line: str) -> None:
    # a collector that crashed before its report has no counters; that is an error, not zero loss
    with pytest.raises(ValueError, match="missing the loss counters"):
        parse_collector_report(REPORT.replace(line + "\n", ""))


def test_collector_gap_flags_incomplete() -> None:
    # a collector restart emits a {"type":"gap"} marker into the stream; the reader must COUNT it
    # (not raise) and the window must read as incomplete, with health flagging it.
    from observe.ebpf_reader import read_copies_report

    lines = [
        (
            '{"type":"enqueue","t_ns":1,"pid":1,"tid":1,"bytes":4096,"kind":"H2D",'
            '"api":"cudaMemcpyAsync","stream":7,"dst":100,"src":200}'
        ),
        '{"type":"gap","reason":"collector restart 1"}',
        (
            '{"type":"completion","via":"stream-sync","t_ns":2,"pid":1,"tid":1,"n":1,'
            '"bytes":4096,"latency_ns":1,"wait_ns":0,"handle":7}'
        ),
    ]
    rep = read_copies_report(lines)
    assert rep.gaps == 1, "the gap marker must be counted, not raise"
    assert not rep.complete, "a window with a collector restart is incomplete"
    assert len(rep.copies) == 1, "the real copy is still read across the gap"
    h = collect_health(capture=rep)
    assert h.collector_gaps == 1 and not h.ok, "health must flag the collector gap"


def test_alloc_drops_flip_ok() -> None:
    h = collect_health(alloc_drops=3)
    assert not h.ok and h.lost == 3


def test_parse_ringbuf_drops() -> None:
    # kv_alloc / storage_read / ekv_readout / http_label print one line at exit
    report = (
        "eKV kv_alloc: pid=7 probe=libcuda.so.1:cuMemAlloc_v2 min_size=0 (Ctrl-C to stop)\nringbuf drops: 4\n"
    )
    assert parse_ringbuf_drops(report) == 4


def test_parse_ringbuf_drops_ignores_kv_copy_lines_and_raises_when_missing() -> None:
    # kv_copy's "ringbuf drops (copy records)" line is a different report and must not match
    with pytest.raises(ValueError, match="no 'ringbuf drops' line"):
        parse_ringbuf_drops(REPORT)
