"""Reading KV reloads from a live kv_copy capture (``observe.reloads``), on the CPU.

The capture lines have the collector's JSON format. The GPU KV cache is one address range, which the engine
publishes next to its control mailbox; host memory lies outside it.
"""

import json
from pathlib import Path

import pytest
import torch

from observe.reloads import ReloadTracker, kv_cache_path, kv_cache_ranges, publish_kv_cache, read_kv_cache

KV = (0x7E0000000000, 0x7E0040000000)  # the GPU KV cache
HOST = 0x7F0000000000  # pinned host memory of the offload
STREAM, EVENT, COMPUTE = 0xA0, 0xE0, 0xC0
BLOCK = 2 << 20


def enqueue(t_ns: int, src: int, dst: int, stream: int = STREAM, kind: str = "BATCH") -> str:
    api = "cuMemcpyBatchAsync" if kind == "BATCH" else "cudaMemcpyAsync"
    return json.dumps(
        {
            "type": "enqueue",
            "t_ns": t_ns,
            "pid": 7,
            "tid": 7,
            "bytes": BLOCK,
            "kind": kind,
            "api": api,
            "stream": stream,
            "dst": dst,
            "src": src,
        }
    )


def bind(t_ns: int, stream: int = STREAM, event: int = EVENT) -> str:
    return json.dumps({"type": "bind", "t_ns": t_ns, "pid": 7, "tid": 7, "event": event, "stream": stream})


def completion(t_ns: int, handle: int = EVENT, n: int = 1) -> str:
    return json.dumps(
        {
            "type": "completion",
            "via": "event-query",
            "t_ns": t_ns,
            "pid": 7,
            "tid": 7,
            "n": n,
            "handle": handle,
        }
    )


def reload_lines(t_ns: int, blocks: int, done_ns: int, stream: int = STREAM, event: int = EVENT) -> list[str]:
    """A reload of ``blocks`` blocks from host memory, enqueued at ``t_ns`` and polled complete at ``done_ns``."""
    lines = [enqueue(t_ns, HOST + i * BLOCK, KV[0] + i * BLOCK, stream) for i in range(blocks)]
    return [*lines, bind(t_ns + 10, stream, event), completion(done_ns, event, blocks)]


def write(path: Path, lines: list[str], end: str = "\n") -> None:
    with open(path, "a") as f:
        f.write("\n".join(lines) + end)


def test_the_engine_publishes_its_kv_cache_storages_once_each(tmp_path, monkeypatch):
    monkeypatch.setenv("EKV_CONTROL_DIR", str(tmp_path))
    buffer = torch.zeros(4, 1024, dtype=torch.uint8)
    other = torch.zeros(256, dtype=torch.uint8)
    ranges = kv_cache_ranges([buffer[0], buffer[2], other])  # two layers are views of one storage
    base = buffer.untyped_storage().data_ptr()
    assert (base, base + 4096) in ranges and len(ranges) == 2
    with pytest.raises(RuntimeError, match="not published"):
        read_kv_cache(4242)
    publish_kv_cache(4242, ranges)
    assert kv_cache_path(4242).parent == tmp_path
    assert read_kv_cache(4242) == ranges


def test_a_reload_is_in_flight_from_its_enqueue_to_its_completion(tmp_path):
    capture = tmp_path / "kv_copy.jsonl"
    write(capture, reload_lines(1_000, 3, 5_000))
    tracker = ReloadTracker(capture, [KV])
    tracker.poll()
    assert len(tracker.reloads) == 1
    reload = tracker.reloads[0]
    assert (reload.blocks, reload.nbytes, reload.start_ns, reload.done_ns) == (3, 3 * BLOCK, 1_000, 5_000)
    assert tracker.watermark_ns == 5_000
    assert tracker.in_flight_ns(0, 10_000) == 4_000
    assert tracker.in_flight_ns(2_000, 3_000) == 1_000
    assert tracker.in_flight_ns(5_000, 9_000) == 0
    assert tracker.started_bytes(0, 1_001) == 3 * BLOCK and tracker.started_bytes(1_001, 9_000) == 0


def test_a_store_and_a_runtime_copy_are_not_reloads_and_the_runtime_copy_moves_the_watermark(tmp_path):
    capture = tmp_path / "kv_copy.jsonl"
    store = [enqueue(1_000, KV[0], HOST), bind(1_010), completion(3_000)]
    runtime = [enqueue(8_000, HOST + 99 * BLOCK, 0x7D0000000000, COMPUTE, kind="H2D")]
    write(capture, store + runtime)
    tracker = ReloadTracker(capture, [KV])
    tracker.poll()
    assert tracker.reloads == []
    assert tracker.watermark_ns == 8_000


def test_a_reload_not_yet_complete_is_in_flight_to_the_end_of_the_window(tmp_path):
    capture = tmp_path / "kv_copy.jsonl"
    write(capture, reload_lines(1_000, 2, 5_000)[:-1])  # enqueued and bound, not yet polled complete
    tracker = ReloadTracker(capture, [KV])
    tracker.poll()
    assert tracker.in_flight_ns(0, 9_000) == 8_000
    write(capture, [completion(4_000)])
    tracker.poll()
    assert tracker.in_flight_ns(0, 9_000) == 3_000


def test_a_batch_split_across_two_reads_is_one_reload_and_a_partial_line_waits(tmp_path):
    capture = tmp_path / "kv_copy.jsonl"
    lines = reload_lines(1_000, 4, 5_000)
    write(capture, lines[:2])
    write(capture, [lines[2][:20]], end="")  # the collector has written part of a line
    tracker = ReloadTracker(capture, [KV])
    tracker.poll()
    assert tracker.reloads[0].blocks == 2
    write(capture, [lines[2][20:], *lines[3:]])
    tracker.poll()
    assert len(tracker.reloads) == 1 and tracker.reloads[0].blocks == 4
    assert tracker.reloads[0].done_ns == 5_000


def test_overlapping_reloads_are_counted_once(tmp_path):
    capture = tmp_path / "kv_copy.jsonl"
    write(capture, reload_lines(1_000, 1, 6_000, stream=0xA1, event=0xE1))
    write(capture, reload_lines(4_000, 1, 9_000, stream=0xA2, event=0xE2))
    tracker = ReloadTracker(capture, [KV])
    tracker.poll()
    assert len(tracker.reloads) == 2
    assert tracker.in_flight_ns(0, 20_000) == 8_000  # [1000, 9000)
    tracker.forget_before(7_000)
    assert [r.start_ns for r in tracker.reloads] == [4_000]


def test_a_copy_the_tracker_cannot_place_and_a_gap_in_the_capture_fail_loudly(tmp_path):
    capture = tmp_path / "kv_copy.jsonl"
    write(capture, [enqueue(1_000, HOST, HOST + BLOCK)])  # a batch copy that does not touch the GPU KV cache
    with pytest.raises(ValueError, match="GPU KV cache"):
        ReloadTracker(capture, [KV]).poll()
    capture.unlink()
    write(capture, [json.dumps({"type": "gap", "t_ns": 5, "reason": "restart"})])
    with pytest.raises(RuntimeError, match="lost records"):
        ReloadTracker(capture, [KV]).poll()
