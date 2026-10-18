"""One controller for the worker processes of a tensor-parallel engine.

Tensor parallelism splits the attention heads over the workers. Every worker holds the same block table
and the same requests, and each runs the attention kernel for its own heads. The controller therefore does
two things for the set of workers. It writes one decision to the mailbox of every worker, so that every
worker trims the same blocks of the same request. It reads the attention share from every worker and
averages the ranks, so that a block is ranked by all the heads of the model and not by the heads of one
worker.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from observe.publication import ControlMailbox, MailboxWriter


class RankedBox:
    """The read side of the mailboxes of all ranks, as one mailbox.

    The request keys and lengths come from rank 0. Every rank applies a decision by request key, so a rank
    that has not yet seen the newest batch still applies the decision to the right rows."""

    def __init__(self, boxes: Sequence[ControlMailbox]) -> None:
        if not boxes:
            raise ValueError("a ranked mailbox needs at least one rank")
        shapes = {(box.max_rows, box.row_width) for box in boxes}
        if len(shapes) != 1:
            raise ValueError(f"the ranks' mailboxes differ in shape: {sorted(shapes)}")
        self.boxes = list(boxes)
        self.max_rows, self.row_width = shapes.pop()

    def status(self) -> dict[str, int]:
        """The counters of the ranks, combined: a generation is applied when every rank has applied it.

        The application time is that of the last rank, the read counts add up, and an audit is done when
        every rank has finished it; an audit has started when any rank has started it."""
        each = [box.status() for box in self.boxes]
        combined = dict(each[0])
        combined["applied_generation"] = min(s["applied_generation"] for s in each)
        combined["applied_ns"] = max(s["applied_ns"] for s in each)
        combined["audit_done"] = min(s["audit_done"] for s in each)
        combined["audit_started"] = max(s["audit_started"] for s in each)
        combined["read_blocks"] = sum(s["read_blocks"] for s in each)
        combined["full_blocks"] = sum(s["full_blocks"] for s in each)
        return combined

    def request_context(self) -> tuple[int, np.ndarray, np.ndarray]:
        return self.boxes[0].request_context()

    def audit_record(self) -> tuple[int, int, np.ndarray]:
        """The number of audits that every rank has finished, the number that some rank has started, and the
        keys that rank 0 audited last."""
        records = [box.audit_record() for box in self.boxes]
        return min(r[0] for r in records), max(r[1] for r in records), records[0][2]

    def skew(self) -> int:
        """The most generations by which one rank is ahead of another in what it has applied."""
        applied = [s["applied_generation"] for s in (box.status() for box in self.boxes)]
        return max(applied) - min(applied)


class RankedWriter:
    """Commits each decision to the mailbox of every rank."""

    def __init__(self, writers: Sequence[MailboxWriter]) -> None:
        self.writers = list(writers)
        self.box = RankedBox([w.box for w in self.writers])

    @property
    def last_generation(self) -> int:
        generations = {w.last_generation for w in self.writers}
        if len(generations) != 1:
            raise RuntimeError(f"the ranks' mailboxes are at different generations: {sorted(generations)}")
        return generations.pop()

    def stage_rows(
        self, keep: np.ndarray, audit: np.ndarray, policy_keys: np.ndarray, release: np.ndarray
    ) -> None:
        for writer in self.writers:
            writer.stage_rows(keep, audit, policy_keys, release)

    def ack_audit(self, done: int) -> None:
        for writer in self.writers:
            writer.ack_audit(done)

    def stage_lease(self, deadline_ns: int) -> None:
        for writer in self.writers:
            writer.stage_lease(deadline_ns)

    def commit(self) -> int:
        """Commit to every rank. A rank that refuses while another accepts is an error."""
        accepted = {writer.commit() for writer in self.writers}
        if len(accepted) != 1:
            raise RuntimeError("some ranks accepted a generation and others did not")
        return accepted.pop()

    def close(self) -> None:
        for writer in self.writers:
            writer.close()


class RankedShareSource:
    """The attention share of a block, averaged over the ranks.

    A rank records the share for its own heads. The mean over ranks is the share over all the heads, as
    one worker would record with no tensor parallelism. A block with no evidence at some rank has none
    overall, and the controller then leaves the request untrimmed."""

    def __init__(self, sources: Sequence[object]) -> None:
        if not sources:
            raise ValueError("a ranked share source needs at least one rank")
        self.sources = list(sources)

    def read(self, live_rows: int, row_blocks: np.ndarray) -> np.ndarray:
        shares = [source.read(live_rows, row_blocks) for source in self.sources]  # ty: ignore[unresolved-attribute]
        return np.mean(shares, axis=0)
