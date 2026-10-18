"""The block-table view: kept blocks, order, minimum keep, and the length of the view.

Pure CPU. The reference is a plain Python loop over each request, so the vectorized
implementation is checked against a different construction on random rows.
"""

import os
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from runtime.block_view import DecodeView, compact_block_table

BLOCK = 16
FAIL = 0


def decode_rows(n: int) -> torch.Tensor:
    """The query lengths of a batch in which all ``n`` rows decode."""
    return torch.ones(n, dtype=torch.int32)


def check(name: str, cond: bool) -> None:
    global FAIL
    print(f"  {'ok' if cond else 'FAIL'}  {name}")
    if not cond:
        FAIL = 1


def view_of(
    table: list[list[int]], lens: list[int], keep: list[list[bool]]
) -> tuple[torch.Tensor, torch.Tensor]:
    return compact_block_table(
        torch.tensor(table, dtype=torch.int32),
        torch.tensor(lens, dtype=torch.int32),
        torch.tensor(keep, dtype=torch.bool),
        BLOCK,
    )


def reference(table: list[int], length: int, keep: list[bool]) -> tuple[list[int], int]:
    """The kept blocks and the view length for one request, built one block at a time."""
    n_blocks = -(-length // BLOCK)
    if n_blocks == 0:
        return [], 0
    chosen = [p for p in range(n_blocks) if keep[p] or p in (0, n_blocks - 1)]
    last_tokens = length - (n_blocks - 1) * BLOCK
    return [table[p] for p in chosen], (len(chosen) - 1) * BLOCK + last_tokens


def main() -> None:
    ident, ident_len = view_of([[7, 8, 9, 10]], [64], [[True] * 4])
    check(
        "keep-all leaves the row and its length unchanged",
        ident.tolist() == [[7, 8, 9, 10]] and ident_len.tolist() == [64],
    )

    view, lens = view_of([[7, 8, 9, 10]], [40], [[False] * 4])
    check(
        "a partial last block is kept, so the view is first block plus last block",
        view[0, :2].tolist() == [7, 9] and lens.tolist() == [16 + 8],
    )

    view, lens = view_of([[7, 8, 9, 10]], [64], [[False, True, False, False]])
    check(
        "the first and the last block are always kept",
        view[0, :3].tolist() == [7, 8, 10] and lens.tolist() == [48],
    )

    view, lens = view_of([[1, 2, 3, 4, 5, 6]], [96], [[True, False, True, False, True, False]])
    check(
        "kept blocks keep their original order",
        view[0, :4].tolist() == [1, 3, 5, 6] and lens.tolist() == [64],
    )

    _, lens = view_of([[1, 2], [3, 4]], [0, 20], [[True, True], [False, False]])
    check("a row with no tokens has length 0", lens.tolist() == [0, 16 + 4])

    view, lens = view_of([[5]], [3], [[False]])
    check(
        "a one-block request keeps its block and its length", view[0, 0].item() == 5 and lens.tolist() == [3]
    )

    rng = random.Random(0)
    ok = True
    for _ in range(300):
        rows, width = rng.randint(1, 6), rng.randint(1, 40)
        table = [[rng.randint(1, 10_000) for _ in range(width)] for _ in range(rows)]
        lens = [rng.randint(0, width * BLOCK) for _ in range(rows)]
        keep = [[rng.random() < 0.5 for _ in range(width)] for _ in range(rows)]
        view, view_lens = view_of(table, lens, keep)
        for r in range(rows):
            blocks, length = reference(table[r], lens[r], keep[r])
            ok = ok and view[r, : len(blocks)].tolist() == blocks and view_lens[r].item() == length
    check("300 random batches match a one-block-at-a-time reference", ok)

    decode_view()
    newest_rows_check()
    print("PASS: block-table view" if not FAIL else "FAIL: block-table view")
    sys.exit(FAIL)


def decode_view() -> None:
    """The committed policy: identity before a commit, after expiry, and on audit steps."""
    table = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.int32)
    lens = torch.tensor([64, 48], dtype=torch.int32)
    mask = np.array([[1, 0, 0, 1], [1, 1, 0, 0]], dtype=np.uint32)
    hour = 3_600 * 10**9

    view = DecodeView(rows=2, width=4, block_size=BLOCK)
    got_table, got_lens = view.build(table, lens, decode_rows(2))
    check(
        "before any commit the view is the identity",
        got_table.tolist() == table.tolist() and got_lens.tolist() == [64, 48],
    )

    view.write_row_controls(mask)
    view.write_lease_deadline(time.monotonic_ns() + hour)
    got_table, got_lens = view.build(table, lens, decode_rows(2))
    check(
        "a live policy gives each row its kept blocks",
        got_table[0, :2].tolist() == [1, 4]
        and got_table[1, :3].tolist() == [5, 6, 7]  # the newest block is kept with blocks 0 and 1
        and got_lens.tolist() == [32, 48],
    )

    view.write_lease_deadline(time.monotonic_ns() - 1)
    got_table, got_lens = view.build(table, lens, decode_rows(2))
    check(
        "an expired policy restores full reads",
        got_table.tolist() == table.tolist() and got_lens.tolist() == [64, 48],
    )

    # the engine waited longer than the lease after its batch update: the generation the publisher applies in
    # the build renews the lease, so the step keeps its policy
    view.before_build = lambda: view.write_lease_deadline(time.monotonic_ns() + hour)
    _, got_lens = view.build(table, lens, decode_rows(2))
    check("a generation applied in the build renews an expired lease", got_lens.tolist() == [32, 48])
    view.before_build = None

    view.write_lease_deadline(time.monotonic_ns() + hour)
    view.reassign_rows({}, [0], [])
    _, got_lens = view.build(table, lens, decode_rows(2))
    check(
        "a row that holds a new request starts with full reads and the other row keeps its policy",
        got_lens.tolist() == [64, 48],
    )
    view.write_row_controls(mask)
    view.reassign_rows({1: 0, 0: 1}, [], [])  # the two requests swap rows
    _, got_lens = view.build(table, lens, decode_rows(2))
    check("a moved request takes its keep decisions to its new row", got_lens.tolist() == [48, 32])

    counted = DecodeView(rows=2, width=4, block_size=BLOCK)
    counted.write_row_controls(mask)
    counted.write_lease_deadline(time.monotonic_ns() + hour)
    counted.build(table, lens, decode_rows(2))
    counted.build(table, lens, decode_rows(2))
    check(
        "the view counts the blocks read against the blocks full reads would read",
        counted.reads() == (2 * (2 + 3), 2 * (4 + 3)),
    )

    class Plane:
        on: list[bool]

        def __init__(self) -> None:
            self.on = []
            self.chosen = []

        def set(self, on: bool) -> None:
            self.on.append(on)

        def select(self, audited: np.ndarray, token_start: np.ndarray) -> None:
            self.chosen.append((audited.tolist(), token_start.tolist()))

        chosen: list[tuple[list[bool], list[int]]]

    audited = DecodeView(rows=2, width=4, block_size=BLOCK, audit=Plane())
    audited.write_row_controls(mask)
    audited.write_lease_deadline(time.monotonic_ns() + hour)
    audited.write_audit_rows(np.array([True, False]))
    steps = [audited.build(table, lens, decode_rows(2))[1].tolist() for _ in range(3)]
    check(
        "an audited row is read in full for one step and the other row keeps its policy",
        steps == [[64, 48], [32, 48], [32, 48]],
    )
    check(
        "the plane is told which rows to record and where each row's query sits in the batch",
        audited.audit.chosen[0] == ([True, False], [0, 1])
        and audited.audit.chosen[1] == ([False, False], [0, 1]),
    )
    check(
        "the audit step is completed by the step after it, and is recorded",
        audited.audits_completed == 1 and audited.last_audit_rows.tolist() == [True, False],
    )
    check("the statistic is on for the audit step only", audited.audit.on == [True, False, False])  # ty: ignore[unresolved-attribute]  # the fake plane
    breakdown = audited.read_breakdown()
    check(
        "reads are split into the audit step and the rows that read their whole table without one",
        breakdown["audit"] == 4 and breakdown["untrimmed"] == 3 * 3 and breakdown["read"] == 4 + 2 + 2 + 9,
    )

    try:
        DecodeView(rows=2, width=4, block_size=BLOCK).write_audit_rows(np.array([True, False]))
        planeless = False
    except ValueError:
        planeless = True
    check("an audit without a plane to switch the statistic on is rejected", planeless)

    buffers = [view.build(table, lens, decode_rows(2))[0].data_ptr() for _ in range(3)]
    check("the view buffers keep one address, which a captured graph needs", len(set(buffers)) == 1)

    try:
        view.write_row_controls(mask, np.zeros((2, 4), dtype=np.float32))
        soft_rejected = False
    except ValueError:
        soft_rejected = True
    try:
        view.write_row_controls(np.ones((3, 4), dtype=np.uint32))
        shape_rejected = False
    except ValueError:
        shape_rejected = True
    check("soft weights and a wrong mask shape are rejected, not ignored", soft_rejected and shape_rejected)

    # A step that mixes prefill and decode trims the decode row and leaves the prefill row whole.
    mixed = DecodeView(rows=2, width=4, block_size=BLOCK, audit=Plane())
    mixed.write_row_controls(mask)
    mixed.write_lease_deadline(time.monotonic_ns() + hour)
    mixed.write_audit_rows(np.array([False, True]))
    one_decoding = torch.tensor([1, 8], dtype=torch.int32)
    _, mixed_lens = mixed.build(table, lens, one_decoding)
    check("a prefill row in a mixed step reads its whole table", mixed_lens.tolist() == [32, 48])
    check("only decode rows count as reads", mixed.reads() == (2, 4))
    check("an audit request waits while its row prefills", bool(mixed.audit_rows[1]) and not mixed.audit_step)
    mixed.build(table, lens, decode_rows(2))
    check(
        "it is served by the first step in which the row decodes",
        mixed.audit_step and not mixed.audit_rows[1],
    )

    # With prefill audits, the audit of a new request is served by the step that computes its last prompt tokens.
    checks: list[bool] = []
    early = DecodeView(rows=2, width=4, block_size=BLOCK, audit=Plane(), prefill_audit=True)
    early.on_prefill_audit = lambda: checks.append(True)
    early.write_row_controls(mask)
    early.write_lease_deadline(time.monotonic_ns() + hour)
    early.write_audit_rows(np.array([False, True]))
    early.write_final_prefill(np.array([False, False]))
    early.build(table, lens, one_decoding)
    check(
        "a prefill step that is not the request's last waits",
        bool(early.audit_rows[1]) and not early.audit_step,
    )
    early.write_final_prefill(np.array([False, True]))
    _, early_lens = early.build(table, lens, one_decoding)
    check(
        "the last prefill step serves the audit, after the plane's check",
        early.audit_step and not early.audit_rows[1] and checks == [True],
    )
    check(
        "the audited prefill row reads its whole table and adds no decode reads",
        early_lens.tolist() == [32, 48] and early.read_breakdown()["audit"] == 0,
    )
    early.build(table, lens, decode_rows(2))
    check(
        "the next step completes it",
        early.audits_completed == 1 and early.last_audit_rows.tolist() == [False, True],
    )
    early.write_audit_rows(np.array([True, False]))
    early.build(table, lens, one_decoding)
    check("the mark of a last prefill step holds for one step only", early.audit_step and checks == [True])
    try:
        DecodeView(rows=2, width=4, block_size=BLOCK, prefill_audit=True)
        planeless_prefill = False
    except ValueError:
        planeless_prefill = True
    check("a prefill audit without an audit plane is rejected", planeless_prefill)

    # Release: only after an audit, only inside what it covered, and a released block is never read again.
    rel = DecodeView(rows=2, width=4, block_size=BLOCK, audit=Plane())
    ask = np.array([[1, 1, 1, 1], [0, 0, 0, 0]], dtype=np.uint8)
    check("a release before any audit is refused", rel.write_release(ask) == [])
    rel.write_row_controls(np.ones((2, 4), dtype=np.uint32))
    rel.write_lease_deadline(time.monotonic_ns() + hour)
    rel.write_audit_rows(np.array([True, False]))
    rel.build(table, lens, decode_rows(2))  # the audit step: row 0 has four blocks
    rel.build(table, lens, decode_rows(2))  # the next step completes the audit
    check(
        "after the audit, blocks 1 and 2 are released; block 0 and the last audited block are not",
        rel.write_release(ask) == [(0, 1), (0, 2)],
    )
    check("a block is released once", rel.write_release(ask) == [])
    got_table, got_lens = rel.build(table, lens, decode_rows(2))
    check(
        "a released block is not read",
        got_table[0, :2].tolist() == [1, 4] and got_lens.tolist() == [32, 48],
    )
    rel.write_lease_deadline(time.monotonic_ns() - 1)
    _, got_lens = rel.build(table, lens, decode_rows(2))
    check("it stays unread when the lease has expired", got_lens.tolist() == [32, 48])
    rel.write_lease_deadline(time.monotonic_ns() + hour)
    rel.write_audit_rows(np.array([True, False]))
    _, got_lens = rel.build(table, lens, decode_rows(2))
    check("it stays unread in an audit", got_lens.tolist() == [32, 48])
    rel.reassign_rows({1: 0, 0: 1}, [], [])  # the two requests swap rows
    check("a released block moves with its request", rel.released[1].tolist() == [False, True, True, False])
    rel.reassign_rows({}, [1], [])  # a new request takes the row
    check("a new request starts with nothing released", not rel.released[1].any() and not rel.row_audited[1])


def newest_rows_check() -> None:
    """A prefill step may be an audit step only if every recording kernel records from the program that holds a
    request's newest query tokens, as the pass reports in its coverage file."""
    import json
    import tempfile

    from runtime.audit import AuditPlane

    def outcome(entries: list[dict[str, object]]) -> str:
        with tempfile.TemporaryDirectory() as directory:
            os.environ["EKV_RUNTIME_DIR"] = directory
            try:
                with open(f"{directory}/ekv_coverage_{os.getpid()}.json", "w") as f:
                    f.writelines(json.dumps(entry) + "\n" for entry in entries)
                AuditPlane().require_newest_rows()
                return "ok"
            except RuntimeError:
                return "refused"
            finally:
                del os.environ["EKV_RUNTIME_DIR"]

    safe = {"kernel": "attention", "attn_share_sites": 1, "attn_share_newest_rows": 1}
    unsafe = {"kernel": "extend", "attn_share_sites": 1, "attn_share_newest_rows": 0}
    other = {"kernel": "gather", "attn_share_sites": 0}
    check("every recording kernel records the newest rows: allowed", outcome([safe, other]) == "ok")
    check("one recording kernel records every program: refused", outcome([safe, unsafe]) == "refused")
    check("no recording kernel compiled yet: refused", outcome([other]) == "refused")


if __name__ == "__main__":
    main()
