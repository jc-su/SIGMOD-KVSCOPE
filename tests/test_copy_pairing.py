"""Regression for the event-completion mis-pairing bug (ebpf_reader): with two
CONCURRENT offload streams each completed via a CUDA event, the old global-oldest
guess attributed a completion to whichever copy was enqueued first — crossing
streams and stamping the wrong block's latency. The `bind` record (emitted at
cudaEventRecord) re-keys each stream's pending copies onto its event, so the event
completion pairs to EXACTLY those copies.

This test fails on the pre-fix reader (A would get stream-B's completion time)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from observe.ebpf_reader import read_copies, read_copies_report  # noqa: E402

A, B = 0xAAA, 0xBBB  # two distinct device (src) addresses, one per stream


def test_concurrent_event_streams_pair_exactly():
    # stream 100 offloads block A; stream 200 offloads block B; each is recorded
    # onto its own event. Event 22 (stream 200/B) completes FIRST, event 11 second.
    lines = [
        '{"type":"enqueue","t_ns":1000,"kind":"D2H","stream":100,"src":2730,"dst":1,"bytes":4096}',
        '{"type":"enqueue","t_ns":1001,"kind":"D2H","stream":200,"src":3003,"dst":2,"bytes":4096}',
        '{"type":"bind","t_ns":1002,"event":11,"stream":100}',
        '{"type":"bind","t_ns":1003,"event":22,"stream":200}',
        '{"type":"completion","via":"event-sync","t_ns":5000,"n":1,"handle":22}',
        '{"type":"completion","via":"event-sync","t_ns":9000,"n":1,"handle":11}',
    ]
    copies = read_copies(lines)
    by_addr = {c.addr: c for c in copies}
    a, b = by_addr[A], by_addr[B]
    # B (stream 200, event 22) finished at 5000; A (stream 100, event 11) at 9000.
    assert b.finished_at == 5000 and b.cost == 5000 - 1001, (b.finished_at, b.cost)
    assert a.finished_at == 9000 and a.cost == 9000 - 1000, (a.finished_at, a.cost)
    # the bug would give A the 5000 completion (oldest-unmatched) -> mask it:
    assert a.finished_at != 5000, "regressed: completion crossed streams"
    print(f"concurrent streams: A={a.cost}ns (event11), B={b.cost}ns (event22) — exact  OK")


def test_stream_sync_still_exact():
    # one stream, stream-sync completion (handle == stream) — must still pair FIFO.
    lines = [
        '{"type":"enqueue","t_ns":100,"kind":"D2H","stream":7,"src":4096,"dst":1,"bytes":64}',
        '{"type":"enqueue","t_ns":200,"kind":"D2H","stream":7,"src":8192,"dst":1,"bytes":64}',
        '{"type":"completion","via":"stream-sync","t_ns":900,"n":2,"handle":7}',
    ]
    copies = read_copies(lines)
    assert all(c.finished_at == 900 for c in copies) and len(copies) == 2
    print("stream-sync: 2 copies on one stream -> both completed  OK")


def test_unknown_handle_marks_incomplete_not_costed():
    # attach-mid-flight / ringbuf drop: a completion for a handle we never saw.
    # Policy: mark the window incomplete and leave the copy UNCOSTED — never guess
    # the oldest unmatched copy (a wrong cost is worse than a missing one).
    lines = [
        '{"type":"enqueue","t_ns":10,"kind":"D2H","stream":5,"src":4096,"dst":1,"bytes":64}',
        '{"type":"completion","via":"event-sync","t_ns":500,"n":1,"handle":999}',
    ]
    rep = read_copies_report(lines)
    assert len(rep.copies) == 1 and rep.copies[0].finished_at is None  # uncosted, not guessed
    assert rep.unmatched_completions == 1 and not rep.complete and rep.uncosted == 1
    print("unknown handle: window flagged incomplete, copy left uncosted (no guess)  OK")


def test_count_overflow_marks_incomplete_costs_present():
    # completion claims 2 but only 1 enqueue captured (the other was dropped).
    # Policy: cost the one we have, flag the window incomplete via overflow.
    lines = [
        '{"type":"enqueue","t_ns":10,"kind":"D2H","stream":5,"src":4096,"dst":1,"bytes":64}',
        '{"type":"completion","via":"stream-sync","t_ns":500,"n":2,"handle":5}',
    ]
    rep = read_copies_report(lines)
    assert len(rep.copies) == 1 and rep.copies[0].finished_at == 500     # the captured one is costed
    assert rep.overflowed_completions == 1 and not rep.complete
    print("count overflow: captured copy costed, window flagged incomplete  OK")


def test_h2h_is_ignored_not_fatal():
    # host<->host is never a KV tier movement: dropped, never raised.
    lines = [
        '{"type":"enqueue","t_ns":1,"kind":"H2H","stream":1,"src":10,"dst":20,"bytes":64}',
        '{"type":"enqueue","t_ns":2,"kind":"D2H","stream":1,"src":4096,"dst":1,"bytes":64}',
        '{"type":"completion","via":"stream-sync","t_ns":9,"n":1,"handle":1}',
    ]
    copies = read_copies(lines)
    assert len(copies) == 1 and copies[0].addr == 4096 and copies[0].direction == "D2H"
    print("H2H ignored: dropped without error, D2H retained  OK")


def test_default_carries_both_addresses():
    # cudaMemcpyDefault: no declared direction -> carry BOTH addresses for the fuser
    # to resolve by block membership; the reader never guesses a direction.
    lines = [
        '{"type":"enqueue","t_ns":1,"kind":"DEFAULT","stream":1,"src":4096,"dst":8192,"bytes":64}',
        '{"type":"completion","via":"stream-sync","t_ns":9,"n":1,"handle":1}',
    ]
    rep = read_copies_report(lines)
    c = rep.copies[0]
    assert c.direction == "DEFAULT" and c.addr == 4096 and c.other_addr == 8192
    assert c.finished_at == 9 and rep.ambiguous_default == 1 and rep.complete
    print("DEFAULT carried: both addresses kept, direction left to the fuser  OK")


if __name__ == "__main__":
    test_concurrent_event_streams_pair_exactly()
    test_stream_sync_still_exact()
    test_unknown_handle_marks_incomplete_not_costed()
    test_count_overflow_marks_incomplete_costs_present()
    test_h2h_is_ignored_not_fatal()
    test_default_carries_both_addresses()
    print("PASS: copy pairing exact; incomplete windows flagged not guessed; "
          "H2H ignored; DEFAULT carried")
