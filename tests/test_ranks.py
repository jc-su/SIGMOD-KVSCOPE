"""One controller for several workers: the decision reaches every rank, and the evidence of every rank counts.

Each rank here is a real mailbox with its own publisher and a recording view, as in ``test_publication.py``.
"""

import numpy as np
import pytest
from test_publication import ROWS, WIDTH, RecordingView

from observe.publication import ControlMailbox, MailboxWriter, ScheduledPublisher, request_key
from observe.ranks import RankedBox, RankedShareSource, RankedWriter

NO_RELEASE = np.zeros((ROWS, WIDTH), dtype=np.uint8)


def ranks(tmp_path, count: int) -> tuple[RankedWriter, list[ScheduledPublisher], list[RecordingView]]:
    writers, publishers, views = [], [], []
    for rank in range(count):
        path = tmp_path / f"mailbox{rank}"
        ControlMailbox.create(path, ROWS, WIDTH).close()
        view = RecordingView()
        writers.append(MailboxWriter(path))
        publishers.append(ScheduledPublisher(path, view))
        views.append(view)
    return RankedWriter(writers), publishers, views


def keys(*ids: str) -> np.ndarray:
    out = np.zeros(ROWS, dtype=np.uint64)
    for i, name in enumerate(ids):
        out[i] = request_key(name)
    return out


def test_one_decision_is_applied_by_every_rank(tmp_path):
    writer, publishers, views = ranks(tmp_path, 2)
    for publisher in publishers:
        publisher.update_requests(["q0", "d0"], [7, 11])
    keep = np.zeros((ROWS, WIDTH), dtype=np.uint8)
    keep[0, 1] = 1
    writer.stage_rows(keep, np.zeros(ROWS, dtype=np.uint8), keys("q0", "d0"), NO_RELEASE)
    writer.stage_lease(10**18)
    assert writer.commit() == 1
    assert writer.last_generation == 1
    for publisher, view in zip(publishers, views, strict=True):
        assert publisher.drain()
        assert view.keep[-1][0].tolist() == [0, 1, 0, 0]


def test_a_generation_counts_as_applied_when_the_last_rank_has_applied_it(tmp_path):
    writer, publishers, _ = ranks(tmp_path, 2)
    for publisher in publishers:
        publisher.update_requests(["q0", "d0"], [7, 11])
    writer.stage_rows(
        np.ones((ROWS, WIDTH), dtype=np.uint8), np.zeros(ROWS, dtype=np.uint8), keys("q0", "d0"), NO_RELEASE
    )
    writer.commit()
    publishers[0].drain()
    assert writer.box.status()["applied_generation"] == 0, "rank 1 has not applied it"
    assert writer.box.skew() == 1
    publishers[1].drain()
    assert writer.box.status()["applied_generation"] == 1
    assert writer.box.skew() == 0


def test_the_read_counts_of_the_ranks_add_up(tmp_path):
    writer, publishers, views = ranks(tmp_path, 2)
    for rank, (publisher, view) in enumerate(zip(publishers, views, strict=True)):
        view.counts = (10 * (rank + 1), 40)
        for _ in range(32):
            publisher.mark_reads()
    status = writer.box.status()
    assert (status["read_blocks"], status["full_blocks"]) == (30, 80)


def test_an_audit_is_done_when_every_rank_has_finished_it(tmp_path):
    writer, publishers, views = ranks(tmp_path, 2)
    for publisher in publishers:
        publisher.update_requests(["q0", "d0"], [7, 11])
    views[0].last_audit_rows = np.array([True, False, False])
    views[0].audits_started = 1
    publishers[0].mark_audit_start()
    views[0].audits_completed = 1
    publishers[0].mark_audit()
    assert writer.box.audit_record()[:2] == (0, 1)  # started at one rank, done at none
    views[1].last_audit_rows = np.array([True, False, False])
    views[1].audits_started = 1
    publishers[1].mark_audit_start()
    views[1].audits_completed = 1
    publishers[1].mark_audit()
    done, started, audited = writer.box.audit_record()
    assert done == started == 1
    assert audited.tolist() == [request_key("q0"), 0, 0]


def test_ranks_with_different_mailbox_shapes_are_refused(tmp_path):
    ControlMailbox.create(tmp_path / "a", ROWS, WIDTH).close()
    ControlMailbox.create(tmp_path / "b", ROWS + 1, WIDTH).close()
    with pytest.raises(ValueError, match="differ in shape"):
        RankedBox([ControlMailbox(tmp_path / "a"), ControlMailbox(tmp_path / "b")])


def test_ranks_at_different_generations_are_an_error(tmp_path):
    writer, _, _ = ranks(tmp_path, 2)
    writer.writers[0].last_generation = 3
    with pytest.raises(RuntimeError, match="different generations"):
        writer.last_generation  # noqa: B018  # reading the property is the check


class FixedSource:
    def __init__(self, shares: np.ndarray) -> None:
        self.shares = shares

    def read(self, live_rows: int, row_blocks: np.ndarray) -> np.ndarray:
        return self.shares[:live_rows]


def test_the_share_of_a_block_is_the_mean_over_the_ranks():
    a = np.array([[0.5, 0.5, np.nan], [1.0, 0.0, 0.0]])
    b = np.array([[0.3, 0.7, np.nan], [0.0, 1.0, 0.0]])
    mean = RankedShareSource([FixedSource(a), FixedSource(b)]).read(2, np.array([2, 2]))
    assert mean[0, :2].tolist() == [0.4, 0.6]
    assert mean[1].tolist() == [0.5, 0.5, 0.0]
    assert np.isnan(mean[0, 2])


def test_a_block_without_evidence_at_one_rank_has_none_overall():
    a = np.array([[0.5, 0.5]])
    b = np.array([[np.nan, np.nan]])
    assert np.isnan(RankedShareSource([FixedSource(a), FixedSource(b)]).read(1, np.array([2]))).all()
