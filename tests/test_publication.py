"""The mailbox between the controller and the serving process, and the safe point that applies it.

The view here is a recorder: these tests check what the publisher hands to the view and when.
The real view is tested with the controller in ``test_controller.py``.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from observe import publication
from observe.publication import (
    ControlMailbox,
    MailboxWriter,
    ScheduledPublisher,
    engine_request_key,
    request_key,
)

ROWS, WIDTH = 3, 4
NO_RELEASE = np.zeros((ROWS, WIDTH), dtype=np.uint8)


class RecordingView:
    def __init__(self) -> None:
        self.releases: list[np.ndarray] = []
        self.keep: list[np.ndarray] = []
        self.audit: list[np.ndarray] = []
        self.deadlines: list[int] = []
        self.reassigned: list[tuple[dict[int, int], list[int], list[int]]] = []
        self.audits_completed = 0
        self.audits_started = 0
        self.last_audit_rows: np.ndarray = np.zeros(ROWS, dtype=bool)
        self.counts = (0, 0)

    def reads(self) -> tuple[int, int]:
        return self.counts

    def write_row_controls(self, keep: np.ndarray) -> None:
        self.keep.append(keep.copy())

    def write_release(self, mask: np.ndarray) -> list[tuple[int, int]]:
        self.releases.append(mask.copy())
        return [(int(r), int(b)) for r, b in zip(*np.nonzero(mask), strict=True)]

    def write_audit_rows(self, rows: np.ndarray) -> None:
        self.audit.append(rows.copy())

    def write_lease_deadline(self, deadline: int) -> None:
        self.deadlines.append(deadline)

    def reassign_rows(self, moved: dict[int, int], new: list[int], empty: list[int]) -> None:
        self.reassigned.append((dict(moved), list(new), list(empty)))

    def finish_audit(self) -> None:
        """The tests set ``audits_completed`` and ``last_audit_rows`` themselves."""


def rig(tmp_path) -> tuple[MailboxWriter, ScheduledPublisher, RecordingView]:
    path = tmp_path / "mailbox"
    ControlMailbox.create(path, ROWS, WIDTH).close()
    view = RecordingView()
    return MailboxWriter(path), ScheduledPublisher(path, view), view


def keys(*ids: str) -> np.ndarray:
    out = np.zeros(ROWS, dtype=np.uint64)
    for i, name in enumerate(ids):
        out[i] = request_key(name)
    return out


def test_a_generation_is_applied_once_and_acknowledged(tmp_path):
    writer, publisher, view = rig(tmp_path)
    publisher.update_requests(["q0", "d0"], [7, 11])
    keep = (np.arange(ROWS * WIDTH, dtype=np.uint8).reshape(ROWS, WIDTH)) & 1
    audit = np.array([1, 0, 0], dtype=np.uint8)
    writer.stage_rows(keep, audit, keys("q0", "d0"), NO_RELEASE)
    writer.stage_lease(123456)
    assert writer.commit() == 1
    assert publisher.drain()
    assert not publisher.drain(), "one generation must be applied once"
    status = writer.box.status()
    assert status["applied_generation"] == writer.last_generation
    assert status["applied_ns"] > 0
    expected = keep.copy()
    expected[2] = 1  # a row with no request in the batch reads every block
    np.testing.assert_array_equal(view.keep[0], expected)
    assert view.audit[0].tolist() == [True, False, False]
    assert view.deadlines == [123456]


def test_a_generation_follows_request_identity_when_the_batch_is_reordered(tmp_path):
    writer, publisher, view = rig(tmp_path)
    publisher.update_requests(["a", "b"], [2, 2])
    keep = np.array([[0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 1, 1]], dtype=np.uint8)
    writer.stage_rows(keep, np.zeros(ROWS, dtype=np.uint8), keys("a", "b"), NO_RELEASE)
    # The runtime reordered the two requests and put a new one in front of this generation.
    publisher.update_requests(["b", "c", "a"], [2, 2, 2], advance=True)
    assert writer.commit() == 1
    assert publisher.drain()
    np.testing.assert_array_equal(
        view.keep[-1], np.array([[1, 0, 1, 1], [1, 1, 1, 1], [0, 1, 1, 1]], dtype=np.uint8)
    )


def test_the_request_context_changes_generation_only_when_it_changes_or_advances(tmp_path):
    writer, publisher, _ = rig(tmp_path)
    assert publisher.update_requests(["q0", "d0"], [7, 11])
    assert not publisher.update_requests(["q0", "d0"], [7, 11])
    generation, request_keys, lengths = writer.box.request_context()
    assert generation == 1
    assert request_keys.tolist() == [request_key("q0"), request_key("d0"), 0]
    assert lengths.tolist() == [7, 11, 0]
    assert publisher.update_requests(["q0", "d0"], [8, 12], advance=True)
    generation, _, lengths = writer.box.request_context()
    assert generation == 2
    assert lengths.tolist() == [8, 12, 0]


def test_rows_are_reassigned_by_request_identity(tmp_path):
    _, publisher, view = rig(tmp_path)
    publisher.update_requests(["q0", "d0"], [7, 11])
    assert publisher.reassign_changed_requests(["d0", "new"])  # d0 moved to row 0, a new request is in row 1
    assert view.reassigned == [({0: 1}, [1], [])]
    assert not publisher.reassign_changed_requests(["q0", "d0"]), "an unchanged batch reassigns nothing"
    assert publisher.reassign_changed_requests(["q0"])  # d0 finished, so its row is empty
    assert view.reassigned[-1] == ({}, [], [1])


def test_the_read_counters_reach_the_mailbox_once_every_period(tmp_path):
    writer, publisher, view = rig(tmp_path)
    view.counts = (30, 100)
    for _ in range(publication.READ_COUNT_PERIOD - 1):
        assert not publisher.mark_reads()
    assert writer.box.status()["read_blocks"] == 0, "the counters are not published between periods"
    assert publisher.mark_reads()
    status = writer.box.status()
    assert (status["read_blocks"], status["full_blocks"]) == (30, 100)
    assert status["applied_generation"] == 0, "publishing the counters leaves the other counters alone"


def test_a_finished_audit_is_reported_with_the_keys_of_the_rows_it_read(tmp_path):
    writer, publisher, view = rig(tmp_path)
    publisher.update_requests(["q0", "d0"], [7, 11])
    assert not publisher.mark_audit(), "no audit has completed"
    view.last_audit_rows = np.array([True, False, False])
    view.audits_completed = 1
    assert publisher.mark_audit()
    assert not publisher.mark_audit(), "one completed audit is reported once"
    done, _started, audited = writer.box.audit_record()
    assert done == 1
    assert audited.tolist() == [request_key("q0"), 0, 0]


def test_a_release_follows_request_identity_and_is_queued_under_the_engines_id(tmp_path):
    from runtime import release

    release._pending.clear()
    writer, publisher, view = rig(tmp_path)
    publisher.update_requests(["a-1a2b3c4d", "b-5e6f7a8b"], [4, 4])  # the ids the engine uses
    ask = np.zeros((ROWS, WIDTH), dtype=np.uint8)
    ask[0, 1:3] = 1  # blocks 1 and 2 of the request in row 0, which is "a"
    writer.stage_rows(
        NO_RELEASE + 1, np.zeros(ROWS, dtype=np.uint8), keys("a", "b"), ask
    )  # the controller names requests by the client's ids
    publisher.update_requests(["b-5e6f7a8b", "a-1a2b3c4d"], [4, 4], advance=True)  # the rows swap
    assert writer.commit() == 1
    assert publisher.drain()
    assert view.releases[-1][1].tolist() == [0, 1, 1, 0] and not view.releases[-1][0].any()
    assert release._pending == {"a-1a2b3c4d": {1, 2}}
    release._pending.clear()


def test_the_engine_key_equals_the_front_end_key_of_the_clients_id():
    # vLLM appends a random suffix of 8 hex digits to the id a client gave
    assert engine_request_key("q12-1a2b3c4d") == request_key("q12")
    assert request_key("q12") != request_key("q13")
    assert engine_request_key("batch-7-1a2b3c4d") == request_key("batch-7")  # a short tail is not a suffix
    assert engine_request_key("job-deadbeef-1a2b3c4d") == request_key(
        "job-deadbeef"
    )  # only the last one goes


def test_row_state_must_name_its_requests(tmp_path):
    writer, _, _ = rig(tmp_path)
    with pytest.raises(ValueError):
        writer.stage_rows(np.ones((ROWS, WIDTH)), np.zeros(ROWS), np.zeros(ROWS + 1), NO_RELEASE)


def test_updated_batch_is_published_before_attention(monkeypatch):
    calls = []
    monkeypatch.setattr(
        publication,
        "reassign_changed_requests",
        lambda runner, request_ids: calls.append(("reassign", list(request_ids))),
    )
    monkeypatch.setattr(
        publication,
        "mark_final_prefill",
        lambda runner, request_ids, computed, scheduled: calls.append(("final", list(request_ids))),
    )
    monkeypatch.setattr(
        publication,
        "publish_current_requests",
        lambda runner, request_ids, lengths, advance=False: (
            calls.append(("publish", list(request_ids), list(lengths), advance)) or True
        ),
    )
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(
            req_ids=["q", "d", None], num_reqs=2, num_computed_tokens_cpu=np.array([15, 32, 0])
        ),
        _kernel_block_sizes=(16,),
    )
    scheduler_output = SimpleNamespace(num_scheduled_tokens={"q": 2, "d": 1})

    assert publication._publish_updated_batch(runner, scheduler_output, advance=True)
    assert calls == [("reassign", ["q", "d"]), ("final", ["q", "d"]), ("publish", ["q", "d"], [2, 3], True)]
