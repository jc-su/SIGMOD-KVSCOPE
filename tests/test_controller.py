"""The decide stage and the control protocol, end to end on the CPU.

A real mailbox, a real controller, the real safe-point publisher, and a real ``DecodeView`` on CPU
tensors. Only the statistic source is a stand-in that returns the shares a test chooses.
"""

import time
from itertools import pairwise

import numpy as np
import pytest
import torch

from observe.controller import Controller, Settings
from observe.publication import ControlMailbox, MailboxWriter, ScheduledPublisher, request_key
from runtime.block_view import DecodeView

ROWS, WIDTH, BLOCK = 3, 16, 16


def decode_rows(n: int) -> torch.Tensor:
    """The query lengths of a batch in which all ``n`` rows decode."""
    return torch.ones(n, dtype=torch.int32)


class FakeSource:
    def __init__(self, shares: np.ndarray) -> None:
        self.shares = shares

    def read(self, live_rows: int, row_blocks: np.ndarray) -> np.ndarray:
        out = self.shares[:live_rows].copy()
        out[row_blocks[:live_rows] == 0] = np.nan  # as the real source: a row of 0 blocks has no evidence
        return out


class FakePlane:
    """Stands in for the audit plane that switches the pass's statistic on."""

    def __init__(self) -> None:
        self.on: list[bool] = []

    def set(self, on: bool) -> None:
        self.on.append(on)

    def select(self, audited: np.ndarray, token_start: np.ndarray) -> None:
        pass


def rig(
    tmp_path,
    shares: np.ndarray,
    lease_seconds: float = 0.5,
    share_decay: float = 0.5,
    refresh_blocks: int = 0,
    release_after: int = 0,
    cap_above_blocks: int = 0,
):
    path = tmp_path / "mailbox"
    ControlMailbox.create(path, ROWS, WIDTH).close()
    writer = MailboxWriter(path)
    view = DecodeView(ROWS, WIDTH, BLOCK, audit=FakePlane())
    publisher = ScheduledPublisher(path, view)
    settings = Settings(
        budget_fraction=0.5,
        recent_blocks=1,
        lease_seconds=lease_seconds,
        share_decay=share_decay,
        refresh_blocks=refresh_blocks,
        release_after=release_after,
        cap_above_blocks=cap_above_blocks,
    )
    return view, publisher, Controller(writer, FakeSource(shares), settings)


def enter(publisher: ScheduledPublisher, ids: list[str], blocks: list[int]) -> None:
    """What the safe point does before a step: reassign rows, then publish the new batch."""
    publisher.reassign_changed_requests(ids)
    publisher.update_requests(ids, blocks)


def finish_audit(publisher: ScheduledPublisher, view: DecodeView) -> None:
    """A device-complete audit step in which every present request was read in full."""
    view.last_audit_rows = np.array([bool(i) for i in publisher._request_ids] + [False] * ROWS)[:ROWS]
    view.audit_rows[:] = False  # the audit step consumed the requests
    for row, request_id in enumerate(publisher._request_ids):
        if request_id:
            view.row_audited[row] = True
            view.audited_blocks[row] = publisher._request_lengths[row]
    view.start_audit()  # published as the attention build starts the step, before the device writes
    view.audits_completed += 1
    assert publisher.mark_audit()


def concentrated(n: int) -> np.ndarray:
    scores = np.full(WIDTH, np.nan)
    scores[:n] = 0.02 / (n - 1)
    scores[3] = 0.98 - 0.0  # one block carries nearly all the attention
    scores[:n] /= np.nansum(scores[:n])
    return scores


def diffuse(n: int) -> np.ndarray:
    scores = np.full(WIDTH, np.nan)
    scores[:n] = 1.0 / n
    return scores


def test_a_request_without_evidence_keeps_every_block(tmp_path):
    view, publisher, controller = rig(tmp_path, np.full((ROWS, WIDTH), np.nan))
    enter(publisher, ["a", "b"], [10, 10])
    assert controller.tick()
    assert publisher.drain()
    assert view.keep.all()


def test_measured_requests_share_one_budget_by_attention(tmp_path):
    shares = np.full((ROWS, WIDTH), np.nan)
    shares[0], shares[1] = concentrated(12), diffuse(12)
    view, publisher, controller = rig(tmp_path, shares)
    enter(publisher, ["a", "b"], [12, 12])
    box = controller.writer.box
    publisher.view.audits_completed = 0
    # the publisher records the keys of the audit step; the controller then reads the shares
    finish_audit(publisher, view)
    assert box.audit_record()[0] == 1
    controller.tick()
    publisher.drain()
    kept = view.keep[:2, :12].sum(axis=1)
    assert kept.sum() <= int(np.ceil(0.5 * 24))
    assert kept[0] < kept[1]  # the concentrated request gives up blocks to the diffuse one
    assert view.keep[0, 3]  # and keeps the block that carries its attention
    assert view.keep[:2, 0].all() and view.keep[:2, 11].all()  # first and newest blocks


def test_a_length_cap_trims_only_the_long_requests_and_spends_the_budget_on_them(tmp_path):
    shares = np.full((ROWS, WIDTH), np.nan)
    shares[0], shares[1] = diffuse(14), diffuse(6)
    view, publisher, controller = rig(tmp_path, shares, cap_above_blocks=8)
    enter(publisher, ["long", "short"], [14, 6])
    finish_audit(publisher, view)
    controller.tick()
    publisher.drain()
    assert view.keep[1, :6].all()  # the short request reads in full although it is measured
    assert view.keep[0, :14].sum() == 7  # half of the long request's own blocks, not of both requests'
    assert [n for _key, n, _kept in controller.allocation] == [14]  # only the long request is allocated


def test_without_a_length_cap_every_measured_request_is_trimmed(tmp_path):
    shares = np.full((ROWS, WIDTH), np.nan)
    shares[0], shares[1] = diffuse(14), diffuse(6)
    view, publisher, controller = rig(tmp_path, shares)
    enter(publisher, ["long", "short"], [14, 6])
    finish_audit(publisher, view)
    controller.tick()
    publisher.drain()
    assert [n for _key, n, _kept in controller.allocation] == [14, 6]  # both requests share the budget
    assert view.keep[0, :14].sum() + view.keep[1, :6].sum() == 10


def test_a_reordered_batch_receives_each_request_its_own_mask(tmp_path):
    shares = np.full((ROWS, WIDTH), np.nan)
    shares[0], shares[1] = concentrated(12), diffuse(12)
    view, publisher, controller = rig(tmp_path, shares)
    enter(publisher, ["a", "b"], [12, 12])
    finish_audit(publisher, view)
    controller.tick()
    publisher.drain()
    before = view.keep.copy()
    enter(publisher, ["b", "c", "a"], [12, 5, 12])  # the engine reorders rows and adds a request
    controller.tick()
    publisher.drain()
    assert np.array_equal(view.keep[2], before[0])  # request a moved from row 0 to row 2
    assert np.array_equal(view.keep[0], before[1])  # request b moved from row 1 to row 0
    assert view.keep[1].all()  # the new request keeps everything


def test_an_audit_is_published_in_the_rows_it_was_taken_in_when_the_next_batch_moves_its_request(tmp_path):
    view, publisher, controller = rig(tmp_path, np.full((ROWS, WIDTH), np.nan))

    def step(ids: list[str], blocks: list[int]) -> None:
        """One engine step in the order of the vLLM hook: new batch, safe point, then the attention build."""
        enter(publisher, ids, blocks)
        publisher.drain()
        publisher.mark_audit()
        table = torch.arange(WIDTH, dtype=torch.int32)[None, :].repeat(len(ids), 1)
        view.build(table, torch.tensor([n * BLOCK for n in blocks], dtype=torch.int32), decode_rows(len(ids)))

    step(["b"], [6])  # b is audited at its first decode step
    step(["b", "a"], [6, 9])  # a is new and is audited in row 1
    assert view.last_audit_rows[:2].tolist() == [True, False]
    step(["a"], [9])  # b finished, and the engine moved a from row 1 to row 0
    done, _started, keys = controller.writer.box.audit_record()
    assert done == 2 and keys[:2].tolist() == [
        0,
        request_key("a"),
    ]  # a's statistic is in row 1, where it was taken
    assert view.row_audited[0] and view.audited_blocks[0] == 9  # and the view moved a's audit with it


class OverwrittenSource(FakeSource):
    """A share source during whose copy the engine starts the next audit step, which rewrites the statistic."""

    def __init__(self, shares: np.ndarray) -> None:
        super().__init__(shares)
        self.view: DecodeView | None = None

    def read(self, live_rows: int, row_blocks: np.ndarray) -> np.ndarray:
        assert self.view is not None
        self.view.start_audit()
        return super().read(live_rows, row_blocks)


def test_a_statistic_that_the_next_audit_step_overwrites_during_the_copy_is_discarded(tmp_path, monkeypatch):
    shares = np.full((ROWS, WIDTH), np.nan)
    shares[0] = diffuse(9)
    view, publisher, controller = rig(tmp_path, shares)
    source = OverwrittenSource(shares)
    source.view, controller.source = view, source
    enter(publisher, ["a"], [9])
    finish_audit(publisher, view)
    controller.tick()
    assert request_key("a") not in controller.shares  # the copy is not used
    assert controller.summary()["audits_discarded"] == 1
    monkeypatch.setattr(Controller, "UNMEASURED_RETRY_NS", 0)
    controller.tick()
    publisher.drain()
    assert view.audit_rows[0]  # the request is asked for a new audit


def test_a_statistic_whose_next_audit_step_has_started_is_not_read(tmp_path):
    shares = np.full((ROWS, WIDTH), np.nan)
    shares[0] = diffuse(9)
    view, publisher, controller = rig(tmp_path, shares)
    enter(publisher, ["a"], [9])
    finish_audit(publisher, view)
    view.start_audit()  # the next audit step starts before the controller reads the finished one
    controller.tick()
    assert request_key("a") not in controller.shares
    assert controller.summary()["audits_discarded"] == 1


def test_no_audit_step_starts_before_the_controller_has_read_the_last_one(tmp_path):
    shares = np.full((ROWS, WIDTH), np.nan)
    shares[0], shares[1] = diffuse(9), diffuse(9)
    view, publisher, controller = rig(tmp_path, shares)
    table = torch.arange(WIDTH, dtype=torch.int32)[None, :].repeat(2, 1)
    lens = torch.tensor([9 * BLOCK, 9 * BLOCK], dtype=torch.int32)

    def step() -> None:
        enter(publisher, ["a", "b"], [9, 9])
        publisher.drain()
        publisher.mark_audit()
        view.build(table, lens, decode_rows(2))

    enter(publisher, ["a", "b"], [9, 9])
    controller.tick()  # the controller holds a lease
    step()  # both new requests are audited
    assert view.audit_step
    view.audit_rows[0] = True  # another audit of a is asked for
    step()  # the audit has completed, and the controller has not read it yet
    assert not view.audit_step and view.audit_rows[0]  # so the request waits
    controller.tick()  # the controller reads the audit and acknowledges it
    assert request_key("a") in controller.shares
    step()
    assert view.audit_step and not view.audit_rows[0]


def test_without_a_controller_audit_steps_do_not_wait(tmp_path):
    view, publisher, _controller = rig(tmp_path, np.full((ROWS, WIDTH), np.nan))
    table = torch.arange(WIDTH, dtype=torch.int32)[None, :]
    lens = torch.tensor([9 * BLOCK], dtype=torch.int32)
    for expected in (True, True):
        enter(publisher, ["a"], [9])
        publisher.mark_audit()
        view.audit_rows[0] = True
        view.build(table, lens, decode_rows(1))
        assert view.audit_step == expected


def test_the_lease_reaches_the_view(tmp_path):
    view, publisher, controller = rig(tmp_path, np.full((ROWS, WIDTH), np.nan))
    controller.tick()
    publisher.drain()
    assert view.active()
    assert view.deadline_ns - time.monotonic_ns() <= 0.5e9


def test_a_new_request_is_audited_once_by_the_next_decode_step(tmp_path):
    shares = np.full((ROWS, WIDTH), np.nan)
    shares[0], shares[1] = concentrated(12), diffuse(12)
    view, publisher, controller = rig(tmp_path, shares)
    enter(publisher, ["a", "b"], [12, 12])
    assert view.audit_rows[:2].all()  # both are new, so the view audits both at their first decode step
    controller.tick()
    publisher.drain()
    assert view.audit_rows[:2].all()  # the controller neither cancels these nor asks again
    table = torch.arange(WIDTH, dtype=torch.int32)[None, :].repeat(2, 1)
    lens = torch.tensor([12 * BLOCK, 12 * BLOCK], dtype=torch.int32)
    _, view_lens = view.build(table, lens, decode_rows(2))
    assert view_lens.tolist() == [12 * BLOCK, 12 * BLOCK]  # audited rows are read in full
    assert view.audit_step and not view.audit_rows.any()  # and the request is consumed
    assert view.audit.on == [True]
    controller.tick()  # within the hold time the controller does not ask again
    publisher.drain()
    assert not view.audit_rows.any()
    view.build(table, lens, decode_rows(2))  # the next step finishes the audit and is a normal step
    assert view.audits_completed == 1 and view.audit.on == [True, False]
    assert view.last_audit_rows[:2].all()
    publisher.mark_audit()
    controller.tick()  # evidence has arrived, so the policy now applies and no audit is asked
    publisher.drain()
    assert not view.audit_rows.any() and not view.keep[:2].all()


def test_a_moved_request_keeps_its_state_and_a_new_request_is_audited(tmp_path):
    view, publisher, _ = rig(tmp_path, np.full((ROWS, WIDTH), np.nan))
    enter(publisher, ["a", "b"], [12, 12])
    view.audit_rows[0] = False  # a has been audited, b has not
    view.keep[0, :4] = False  # and a has a policy
    enter(publisher, ["b", "c", "a"], [12, 5, 12])  # the engine reorders the rows and admits c
    assert view.audit_rows.tolist() == [
        True,
        True,
        False,
    ]  # b keeps its pending audit, c gets one, a has none
    assert not view.keep[2, :4].any() and view.keep[0].all()  # a's policy moved with it


def test_a_request_that_has_grown_is_audited_again(tmp_path):
    shares = np.full((ROWS, WIDTH), np.nan)
    shares[0] = concentrated(12)
    view, publisher, controller = rig(tmp_path, shares, refresh_blocks=3)
    enter(publisher, ["a"], [12])
    finish_audit(publisher, view)
    controller.tick()
    publisher.drain()
    enter(publisher, ["a"], [14])  # two blocks since the audit
    controller.tick()
    publisher.drain()
    assert not view.audit_rows.any()
    enter(publisher, ["a"], [15])  # three blocks since the audit
    controller.tick()
    publisher.drain()
    assert view.audit_rows[0]


def test_under_a_length_cap_only_a_long_request_is_audited_again(tmp_path):
    shares = np.full((ROWS, WIDTH), np.nan)
    shares[0], shares[1] = diffuse(9), diffuse(4)
    view, publisher, controller = rig(tmp_path, shares, refresh_blocks=3, cap_above_blocks=8)
    enter(publisher, ["long", "short"], [9, 4])
    finish_audit(publisher, view)
    controller.tick()
    publisher.drain()
    enter(publisher, ["long", "short"], [12, 7])  # both have three blocks since the audit
    controller.tick()
    publisher.drain()
    assert view.audit_rows[0] and not view.audit_rows[1]  # the short request reads in full anyway


def test_blocks_added_since_the_audit_are_read_until_they_are_measured(tmp_path):
    shares = np.full((ROWS, WIDTH), np.nan)
    shares[0] = concentrated(12)
    view, publisher, controller = rig(tmp_path, shares)
    enter(publisher, ["a"], [12])
    finish_audit(publisher, view)
    controller.tick()
    publisher.drain()
    assert not view.keep[0, :12].all()  # the measured blocks are trimmed
    enter(publisher, ["a"], [16])
    controller.tick()
    publisher.drain()
    assert view.keep[0, 12:16].all()  # the four new blocks have no evidence and are kept


def count_allocations(monkeypatch) -> list[int]:
    """Count the calls of the allocator that the controller makes."""
    import observe.controller as module

    calls: list[int] = []
    original = module.allocate

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "allocate", counted)
    return calls


def test_a_tick_with_nothing_new_reuses_the_allocation_and_still_renews_the_lease(tmp_path, monkeypatch):
    calls = count_allocations(monkeypatch)
    shares = np.full((ROWS, WIDTH), np.nan)
    shares[0], shares[1] = concentrated(12), diffuse(12)
    view, publisher, controller = rig(tmp_path, shares)
    enter(publisher, ["a", "b"], [12, 12])
    finish_audit(publisher, view)
    controller.tick()
    publisher.drain()
    first_deadline = view.deadline_ns
    assert len(calls) == 1
    time.sleep(0.01)
    for _ in range(3):
        assert controller.tick()
    publisher.drain()
    assert len(calls) == 1  # nothing changed: the allocation was reused
    assert view.deadline_ns > first_deadline  # but the lease moved forward


def test_a_new_length_or_a_new_audit_recomputes_the_allocation(tmp_path, monkeypatch):
    calls = count_allocations(monkeypatch)
    shares = np.full((ROWS, WIDTH), np.nan)
    shares[0], shares[1] = concentrated(12), diffuse(12)
    view, publisher, controller = rig(tmp_path, shares)
    enter(publisher, ["a", "b"], [12, 12])
    finish_audit(publisher, view)
    controller.tick()
    controller.tick()
    assert len(calls) == 1
    publisher.update_requests(["a", "b"], [13, 12])  # one request grew by a block
    controller.tick()
    assert len(calls) == 2
    finish_audit(publisher, view)  # a new audit arrived
    controller.tick()
    assert len(calls) == 3
    enter(publisher, ["a"], [13])  # a request left
    controller.tick()
    assert len(calls) == 4


class StepClock:
    """Stands in for the mailbox of an engine that applies one generation per step, every ``step_ms``."""

    def __init__(self, step_ms: float) -> None:
        self.step_ms = step_ms
        self.generation = 0
        self.time_ns = 10**12

    def step(self, gap_ms: float | None = None) -> None:
        self.generation += 1
        self.time_ns += int((self.step_ms if gap_ms is None else gap_ms) * 1e6)

    def status(self) -> dict[str, int]:
        return {"applied_generation": self.generation, "applied_ns": self.time_ns}


def targeted(tmp_path, target_ms: float | None, start: float = 0.5, floor: float = 0.15) -> Controller:
    shares = np.full((ROWS, WIDTH), np.nan)
    _, _, controller = rig(tmp_path, shares)
    settings = Settings(
        budget_fraction=start, recent_blocks=1, target_step_ms=target_ms, min_budget_fraction=floor
    )
    return Controller(controller.writer, controller.source, settings)


def run_steps(controller: Controller, clock: StepClock, steps: int) -> None:
    for _ in range(steps):
        clock.step()
        controller._observe_step_time(clock.status())


def test_a_step_time_above_the_target_lowers_the_budget_to_the_floor_and_no_further(tmp_path):
    controller, clock = targeted(tmp_path, 50.0), StepClock(120.0)
    seen = [controller.fraction]
    for _ in range(40):
        run_steps(controller, clock, Controller.STEPS_PER_ADJUSTMENT + 1)
        seen.append(controller.fraction)
    assert all(b <= a for a, b in pairwise(seen))
    assert seen[1] < seen[0]
    assert seen[-1] == pytest.approx(0.15)  # the floor


def test_a_step_time_with_slack_raises_the_budget_to_one_and_no_further(tmp_path):
    controller, clock = targeted(tmp_path, 100.0, start=0.3), StepClock(40.0)
    for _ in range(40):
        run_steps(controller, clock, Controller.STEPS_PER_ADJUSTMENT + 1)
    assert controller.fraction == pytest.approx(1.0)


def test_a_step_time_close_to_the_target_leaves_the_budget_alone(tmp_path):
    controller, clock = (
        targeted(tmp_path, 100.0, start=0.4),
        StepClock(95.0),
    )  # above 90% of the target, below it
    run_steps(controller, clock, 100)
    assert controller.fraction == 0.4


def test_an_idle_gap_is_not_a_step(tmp_path):
    controller, clock = targeted(tmp_path, 50.0, start=0.5), StepClock(30.0)
    run_steps(controller, clock, 3)
    clock.step(gap_ms=5000.0)  # the engine had nothing to run
    controller._observe_step_time(clock.status())
    run_steps(controller, clock, 3)
    assert controller._steps_ms and max(controller._steps_ms) < 100.0


def test_without_a_target_the_budget_is_fixed_and_the_step_stamps_are_not_used(tmp_path):
    controller = targeted(tmp_path, None, start=0.35)
    for _ in range(5):
        controller._observe_step_time({})  # a status that has no stamps: it would fail if it were read
    assert controller.fraction == 0.35


def test_the_reaction_and_the_audit_round_trip_are_measured(tmp_path):
    shares = np.full((ROWS, WIDTH), np.nan)
    shares[0] = concentrated(12)
    view, publisher, controller = rig(tmp_path, shares, refresh_blocks=2)
    enter(publisher, ["a"], [12])
    finish_audit(publisher, view)
    controller.tick()  # absorbs the first audit, and commits generation 1
    controller.tick()  # generation 2: the controller commits many generations between two engine steps
    controller.tick()  # generation 3
    publisher.drain()  # the engine applies the newest one at the start of its next step
    controller.tick()  # sees the application of generation 3
    assert len(controller.reaction_ms) == 1 and controller.reaction_ms[0] > 0.0
    assert controller._seen_applied == 3
    publisher.update_requests(["a"], [15])  # the request grew by 3 blocks: the controller asks for an audit
    controller.tick()
    assert controller._audit_asked_ns  # asked, not yet answered
    finish_audit(publisher, view)  # the audit step completed
    controller.tick()
    assert len(controller.audit_ms) == 1 and controller.audit_ms[0] > 0.0
    summary = controller.summary()
    assert summary["reaction_ms_count"] == 1 and summary["audit_ms_count"] == 1


def test_a_new_budget_fraction_recomputes_the_allocation(tmp_path, monkeypatch):
    calls = count_allocations(monkeypatch)
    shares = np.full((ROWS, WIDTH), np.nan)
    shares[0], shares[1] = concentrated(12), diffuse(12)
    view, publisher, controller = rig(tmp_path, shares)
    enter(publisher, ["a", "b"], [12, 12])
    finish_audit(publisher, view)
    controller.tick()
    controller.tick()
    assert len(calls) == 1
    controller.fraction = 0.3
    controller.tick()
    assert len(calls) == 2


def simulate(tmp_path, loads: list[tuple[int, int]], target_ms: float, seed: int = 0) -> list[list[float]]:
    """Close the loop on a modelled engine: a step takes F + A k ms, where k is the blocks it reads.

    ``loads`` is a list of (steps, blocks held by the batch). Returns, for each phase, the step times
    and the budget fraction at the end of each step, interleaved as [step_ms, fraction, ...]."""
    controller, clock = targeted(tmp_path, target_ms, start=1.0, floor=0.15), StepClock(50.0)
    rng = np.random.default_rng(seed)
    fixed_ms, per_block_ms = 25.5, 0.00727  # the step-time model of the measurements (section 12 of the plan)
    phases = []
    for steps, blocks in loads:
        record = []
        for _ in range(steps):
            step_ms = fixed_ms + per_block_ms * blocks * controller.fraction + float(rng.normal(0.0, 1.5))
            clock.step(gap_ms=step_ms)
            controller._observe_step_time(clock.status())
            record += [step_ms, controller.fraction]
        phases.append(record)
    return phases


def test_the_target_mode_settles_near_the_target_under_load_and_returns_to_full_reads_when_load_falls(
    tmp_path,
):
    # light load, heavy load, light load. Full reads cost 62 ms, 120 ms, 62 ms, and the target is 80 ms.
    light, heavy, light_again = simulate(tmp_path, [(160, 5000), (400, 13000), (400, 5000)], 80.0)
    assert light[1::2][-1] == pytest.approx(1.0)  # no overload: every request reads in full
    settled = heavy[-80:]  # the last 40 steps of the heavy phase
    steps, fractions = settled[0::2], settled[1::2]
    assert 0.9 * 80.0 <= float(np.median(steps)) <= 1.05 * 80.0
    assert 0.45 <= fractions[-1] <= 0.7  # the fraction that the model needs is 0.58
    assert light_again[1::2][-1] == pytest.approx(1.0)  # and back to full reads


def test_the_target_mode_does_not_oscillate_once_it_has_settled(tmp_path):
    _, heavy = simulate(tmp_path, [(100, 5000), (600, 13000)], 80.0, seed=3)
    fractions = np.array(heavy[1::2][-200:])
    assert fractions.max() - fractions.min() < 0.12  # a band, not a swing between the floor and 1


def test_a_target_that_cannot_be_met_stays_at_the_floor(tmp_path):
    # even at the floor, 25.5 + 0.00727 * 13000 * 0.15 = 39.7 ms: a 20 ms target is below the fixed cost
    (heavy,) = simulate(tmp_path, [(600, 13000)], 20.0)
    assert heavy[1::2][-1] == pytest.approx(0.15)


class FakeReloads:
    """Stands in for the eBPF reload reader: reloads in flight over chosen intervals, read up to a watermark."""

    def __init__(self) -> None:
        self.intervals: list[tuple[int, int]] = []
        self.watermark_ns = 0

    def poll(self) -> None:
        pass

    def in_flight_ns(self, start_ns: int, end_ns: int) -> int:
        return sum(max(0, min(end_ns, hi) - max(start_ns, lo)) for lo, hi in self.intervals)

    def forget_before(self, ns: int) -> None:
        self.intervals = [(lo, hi) for lo, hi in self.intervals if hi >= ns]


def reload_aware(
    tmp_path, target_ms: float, start: float = 1.0, floor: float = 0.15
) -> tuple[Controller, FakeReloads]:
    shares = np.full((ROWS, WIDTH), np.nan)
    _, _, controller = rig(tmp_path, shares)
    settings = Settings(
        budget_fraction=start,
        recent_blocks=1,
        target_step_ms=target_ms,
        min_budget_fraction=floor,
        reload_aware=True,
    )
    reloads = FakeReloads()
    return Controller(controller.writer, controller.source, settings, reloads=reloads), reloads


def test_a_step_with_a_reload_in_flight_is_not_a_sample_of_the_step_time(tmp_path):
    controller, reloads = reload_aware(tmp_path, 50.0)
    clock = StepClock(40.0)
    clock.step()
    controller._observe_step_time(clock.status())  # the first application: no step yet
    for i in range(5):
        start = clock.time_ns
        clock.step(gap_ms=90.0 if i == 2 else 40.0)
        if i == 2:  # a reload from host memory was in flight during the third step, and made it long
            reloads.intervals.append((start + 1_000_000, start + 30_000_000))
        reloads.watermark_ns = clock.time_ns
        controller._observe_step_time(clock.status())
    assert controller._steps_ms == [40.0] * 4
    assert (controller.steps_classified, controller.steps_with_reload) == (5, 1)
    summary = controller.summary()
    assert summary["reload_share"] == pytest.approx(29.0 / 250.0)


def test_a_step_waits_until_the_collector_has_read_past_its_end(tmp_path):
    controller, reloads = reload_aware(tmp_path, 50.0)
    clock = StepClock(40.0)
    for _ in range(4):
        clock.step()
        controller._observe_step_time(clock.status())
    assert controller._steps_ms == [] and len(controller._unclassified) == 3
    reloads.watermark_ns = clock.time_ns - 1  # the collector has read up to just before the last step ended
    controller._observe_step_time(clock.status())
    assert controller._steps_ms == [40.0, 40.0] and len(controller._unclassified) == 1


def test_a_reload_that_completes_inside_the_running_step_is_kept_until_the_step_is_compared(tmp_path):
    controller, reloads = reload_aware(tmp_path, 50.0)
    clock = StepClock(40.0)
    clock.step()
    reloads.watermark_ns = clock.time_ns
    controller._observe_step_time(clock.status())  # the first application: a step begins now
    begin = clock.time_ns
    reloads.intervals.append((begin + 5_000_000, begin + 10_000_000))  # a reload inside the running step
    reloads.watermark_ns = begin + 20_000_000  # the collector has read past the reload, the step still runs
    controller._observe_step_time(clock.status())  # a tick in the middle of the step
    clock.step()
    reloads.watermark_ns = clock.time_ns
    controller._observe_step_time(clock.status())
    assert (controller.steps_classified, controller.steps_with_reload) == (1, 1)
    assert controller._steps_ms == []


def test_a_collector_that_stops_reporting_stops_the_reload_aware_target(tmp_path):
    controller, _ = reload_aware(tmp_path, 50.0)  # the watermark never moves
    clock = StepClock(40.0)
    with pytest.raises(RuntimeError, match="copy collector"):
        run_steps(controller, clock, Controller.MAX_UNCLASSIFIED + 2)


def drive(
    controller: Controller, reloads: FakeReloads | None, phases: list[tuple[int, int, float]], seed: int = 0
) -> list[float]:
    """Close the loop on a modelled engine whose step takes F + A k ms, plus a stall of ``stall_ms`` in a share
    ``every`` of its steps during which a reload from host memory is in flight. ``phases`` is a list of
    (steps, blocks held by the batch, share of steps with a reload). Returns the fraction after each step."""
    clock = StepClock(50.0)
    rng = np.random.default_rng(seed)
    fixed_ms, per_block_ms, stall_ms = 25.5, 0.00727, 60.0
    fractions = []
    for steps, blocks, share in phases:
        for _ in range(steps):
            start = clock.time_ns
            stalled = rng.random() < share
            step_ms = fixed_ms + per_block_ms * blocks * controller.fraction + float(rng.normal(0.0, 1.5))
            clock.step(gap_ms=step_ms + (stall_ms if stalled else 0.0))
            if reloads is not None:
                if stalled:
                    reloads.intervals.append((start, start + int(stall_ms * 1e6)))
                reloads.watermark_ns = clock.time_ns
            controller._observe_step_time(clock.status())
            fractions.append(controller.fraction)
    return fractions


def test_without_reloads_the_reload_aware_target_moves_the_fraction_as_the_plain_target_does(tmp_path):
    phases = [(160, 5000, 0.0), (400, 13000, 0.0), (400, 5000, 0.0)]
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    plain = drive(targeted(tmp_path / "a", 80.0, start=1.0), None, phases)
    aware_controller, reloads = reload_aware(tmp_path / "b", 80.0)
    aware = drive(aware_controller, reloads, phases)
    assert aware == plain
    assert aware_controller.steps_with_reload == 0 and aware_controller.steps_classified == 959


def test_reload_stalls_drive_the_plain_target_to_the_floor_and_the_reload_aware_target_holds_what_the_reads_need(
    tmp_path,
):
    # 13,000 blocks: full reads cost 120 ms and the 80 ms target needs a fraction of 0.58. In 60% of the steps a
    # reload from host memory adds 60 ms, which no fraction can remove.
    phases = [(800, 13000, 0.6)]
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    plain = drive(targeted(tmp_path / "a", 80.0, start=1.0), None, phases)
    aware_controller, reloads = reload_aware(tmp_path / "b", 80.0)
    aware = drive(aware_controller, reloads, phases)
    assert plain[-1] == pytest.approx(0.15)  # the median step carries a stall: the target trims to the floor
    assert (
        0.45 <= aware[-1] <= 0.7
    )  # the steps without a reload meet the target at the fraction the reads need
    assert aware_controller.steps_with_reload > 0.5 * aware_controller.steps_classified


def test_a_dead_controller_leaves_full_reads(tmp_path):
    shares = np.full((ROWS, WIDTH), np.nan)
    shares[0] = concentrated(12)
    view, publisher, controller = rig(tmp_path, shares, lease_seconds=0.05)
    enter(publisher, ["a"], [12])
    finish_audit(publisher, view)
    controller.tick()
    publisher.drain()
    table = torch.arange(WIDTH, dtype=torch.int32)[None, :].repeat(1, 1)
    lens = torch.tensor([12 * BLOCK], dtype=torch.int32)
    assert view.build(table, lens, decode_rows(1))[1].item() < 12 * BLOCK  # live: trimmed
    time.sleep(0.08)
    assert view.build(table, lens, decode_rows(1))[1].item() == 12 * BLOCK  # the lease expired: full reads


def test_a_recycled_row_starts_with_full_reads(tmp_path):
    shares = np.full((ROWS, WIDTH), np.nan)
    shares[0] = concentrated(12)
    view, publisher, controller = rig(tmp_path, shares)
    enter(publisher, ["a"], [12])
    finish_audit(publisher, view)
    controller.tick()
    publisher.drain()
    assert not view.keep[0].all()
    publisher.reassign_changed_requests(["z"])  # the row now belongs to another request
    assert view.keep[0].all()


def test_shares_are_smoothed_across_audits_and_pad_as_a_request_grows(tmp_path):
    shares = np.full((ROWS, WIDTH), np.nan)
    shares[0, :4] = [0.7, 0.1, 0.1, 0.1]
    view, publisher, controller = rig(tmp_path, shares, share_decay=0.5)
    enter(publisher, ["a"], [4])
    finish_audit(publisher, view)
    controller.tick()
    key = request_key("a")
    assert controller.shares[key][:4] == pytest.approx([0.7, 0.1, 0.1, 0.1])
    shares[0, :6] = [0.2, 0.1, 0.1, 0.1, 0.25, 0.25]
    enter(publisher, ["a"], [6])
    finish_audit(publisher, view)
    controller.tick()
    assert controller.shares[key][:6] == pytest.approx([0.45, 0.1, 0.1, 0.1, 0.125, 0.125])
    assert not controller.shares[key][6:].any()


def test_a_fixed_scoring_rule_needs_no_audit_and_can_keep_the_newest_blocks(tmp_path):
    path = tmp_path / "mailbox"
    ControlMailbox.create(path, ROWS, WIDTH).close()
    view = DecodeView(ROWS, WIDTH, BLOCK)
    publisher = ScheduledPublisher(path, view)
    settings = Settings(budget_fraction=0.5, recent_blocks=0)
    controller = Controller(MailboxWriter(path), None, settings, scores=lambda n: np.arange(n, dtype=float))
    enter(publisher, ["a"], [10])
    controller.tick()
    publisher.drain()
    kept = np.flatnonzero(view.keep[0, :10])
    assert len(kept) == 5 and kept[0] == 0 and kept[-1] == 9  # the first block and the newest ones
    assert kept.tolist() == [0, 6, 7, 8, 9]


def test_a_share_source_and_a_fixed_rule_are_exclusive(tmp_path):
    path = tmp_path / "mailbox"
    ControlMailbox.create(path, ROWS, WIDTH).close()
    with pytest.raises(ValueError):
        Controller(MailboxWriter(path), None, Settings(budget_fraction=0.5))
    with pytest.raises(ValueError):
        Controller(
            MailboxWriter(path),
            FakeSource(np.zeros((ROWS, WIDTH))),
            Settings(budget_fraction=0.5),
            scores=lambda n: np.zeros(n),
        )


def two_audits(publisher, view, controller) -> None:
    """Two completed audits of the request in row 0, each followed by a controller tick and a safe point."""
    for _ in range(2):
        finish_audit(publisher, view)
        publisher.mark_audit()
        controller.tick()
        publisher.drain()


def test_a_block_left_out_of_two_audits_in_a_row_is_released(tmp_path):
    shares = np.full((ROWS, WIDTH), np.nan)
    shares[0] = concentrated(12)
    view, publisher, controller = rig(tmp_path, shares, release_after=2)
    enter(publisher, ["a"], [12])
    finish_audit(publisher, view)
    publisher.mark_audit()
    controller.tick()
    publisher.drain()
    assert not view.released.any()  # one audit is not enough
    finish_audit(publisher, view)
    publisher.mark_audit()
    controller.tick()
    publisher.drain()
    gone = np.flatnonzero(view.released[0])
    assert len(gone) > 0
    assert not view.keep[0, gone].any()  # a released block is not read
    assert 0 not in gone and 11 not in gone  # the first and the newest block are never released
    assert controller.release[request_key("a")].sum() == len(gone)


def test_a_block_that_is_kept_in_an_audit_starts_again(tmp_path):
    shares = np.full((ROWS, WIDTH), np.nan)
    shares[0] = concentrated(12)
    view, publisher, controller = rig(tmp_path, shares, release_after=2)
    enter(publisher, ["a"], [12])
    finish_audit(publisher, view)
    publisher.mark_audit()
    controller.tick()
    publisher.drain()
    left_out = np.flatnonzero(~view.keep[0, :12])
    spared = int(left_out[0])
    shares[0, spared] = 5.0  # the next audit finds that block important
    shares[0] /= np.nansum(shares[0])
    finish_audit(publisher, view)
    publisher.mark_audit()
    controller.tick()
    publisher.drain()
    assert not view.released[0, spared]
    assert controller.streak[request_key("a")][spared] == 0


def test_a_released_block_takes_no_part_in_the_budget(tmp_path):
    shares = np.full((ROWS, WIDTH), np.nan)
    shares[0] = concentrated(12)
    view, publisher, controller = rig(tmp_path, shares, release_after=2)
    enter(publisher, ["a"], [12])
    two_audits(publisher, view, controller)
    gone = np.flatnonzero(view.released[0])
    controller.tick()
    publisher.drain()
    assert not view.keep[0, gone].any()
    assert view.keep[0, :12].sum() <= 6  # the budget is half of the twelve blocks, freed or not


def test_a_request_that_leaves_the_batch_is_forgotten(tmp_path):
    shares = np.full((ROWS, WIDTH), np.nan)
    shares[0] = concentrated(12)
    view, publisher, controller = rig(tmp_path, shares, release_after=2)
    enter(publisher, ["a"], [12])
    two_audits(publisher, view, controller)
    assert controller.release
    enter(publisher, ["b"], [12])  # a is gone, for example preempted, and b has taken its row
    controller.tick()
    assert not controller.release and not controller.streak and not controller.shares


def test_releasing_needs_measured_shares_and_at_least_two_audits(tmp_path):
    path = tmp_path / "mailbox"
    ControlMailbox.create(path, ROWS, WIDTH).close()
    with pytest.raises(ValueError):
        Controller(
            MailboxWriter(path), None, Settings(budget_fraction=0.5, release_after=2), scores=lambda n: n
        )
    with pytest.raises(ValueError):
        Settings(budget_fraction=0.5, release_after=1)


def test_settings_are_validated():
    with pytest.raises(ValueError):
        Settings(budget_fraction=0.0)
    with pytest.raises(ValueError):
        Settings(budget_fraction=0.5, refresh_blocks=-1)
    with pytest.raises(ValueError, match="step-time target"):
        Settings(budget_fraction=0.5, reload_aware=True)


def test_a_reload_source_is_given_exactly_when_the_target_is_reload_aware(tmp_path):
    shares = np.full((ROWS, WIDTH), np.nan)
    _, _, controller = rig(tmp_path, shares)
    plain = Settings(budget_fraction=0.5, target_step_ms=50.0)
    aware = Settings(budget_fraction=0.5, target_step_ms=50.0, reload_aware=True)
    with pytest.raises(ValueError, match="reload source"):
        Controller(controller.writer, controller.source, aware)
    with pytest.raises(ValueError, match="reload source"):
        Controller(controller.writer, controller.source, plain, reloads=FakeReloads())
