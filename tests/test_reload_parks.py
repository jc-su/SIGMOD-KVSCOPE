"""CI (no GPU): the ZERO-TOUCH reload-park attribution (reload -> next-admitted-user proxy) +
two-class provenance. No scheduler/connector hook: ownership is inferred from admission ORDER
(OTel) + reload completions (eBPF) + the index_lookup. The shared-reload over-attribution (req 36)
is fixed because one reload has ONE owner — the request admitted right after it — so a request
reusing an already-resident shared block owns nothing and gets KV-park 0 (EXACT).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from observe import BOUNDED, EXACT, SHARED, attribute_reload_parks   # noqa: E402

# block 10 = a SHARED prefix block; block 20 = a block two requests are admitted on together.
# Times in ns (admissions ~100ms apart >> the 5ms tie window; D/E only 1ms apart -> a real tie).
MS = 1_000_000
# (block, enqueue_ns, completion_ns) — each reload is a 10/5ms copy.
RELOADS = [(10, 90 * MS, 100 * MS), (10, 390 * MS, 400 * MS), (20, 195 * MS, 200 * MS)]
ADMISSION = {"A": 110 * MS, "B": 500 * MS, "C": 410 * MS, "D": 205 * MS, "E": 206 * MS}
ARRIVAL = {"A": 50 * MS, "B": 50 * MS, "C": 350 * MS, "D": 150 * MS, "E": 150 * MS}
BLOCKS = {"A": {10}, "B": {10}, "C": {10}, "D": {20}, "E": {20}}


def test_reload_next_admitted_user():
    out = attribute_reload_parks(RELOADS, ADMISSION, ARRIVAL, BLOCKS)
    # A owns reload@90-100ms -> KV-park = the reload SPAN (10ms), NOT 60ms of arrival->completion
    assert out["A"]["kv_park_ns"] == 10 * MS and out["A"]["provenance"] == BOUNDED
    # B reuses block 10 long after (admitted 500ms) -> owns NO reload -> KV-park 0, EXACT (req-36 fix)
    assert out["B"]["kv_park_ns"] == 0 and out["B"]["provenance"] == EXACT
    # C owns reload@390-400ms -> KV-park = 10ms span
    assert out["C"]["kv_park_ns"] == 10 * MS and out["C"]["provenance"] == BOUNDED
    # D and E are admitted together (1ms apart) on block 20's reload -> they CO-WAITED (the step
    # was blocked until the reload) -> SHARED, and BOTH are charged the full 5ms span (not 0/coin-flip)
    assert out["D"]["provenance"] == SHARED and out["D"]["kv_park_ns"] == 5 * MS
    assert out["E"]["provenance"] == SHARED and out["E"]["kv_park_ns"] == 5 * MS


def test_exclusive_vs_shared():
    # exclusive reloads -> one owner; a shared reload (Z) -> its known co-waiter SET (D and E),
    # not a spurious broadcast: only the requests that co-waited share it.
    out = attribute_reload_parks(RELOADS, ADMISSION, ARRIVAL, BLOCKS)
    assert out["A"]["n_owned"] == 1 and out["A"]["provenance"] == BOUNDED   # exclusive
    assert out["C"]["n_owned"] == 1 and out["C"]["provenance"] == BOUNDED   # exclusive
    assert out["B"]["n_owned"] == 0                                         # cache-hit, owns nothing
    assert out["D"]["n_owned"] == 1 and out["E"]["n_owned"] == 1            # both own the shared Z
    # 3 reloads: two exclusive (A,C) + one shared by exactly the 2 co-waiters (D,E)
    shared_reqs = [r for r, v in out.items() if v["provenance"] == SHARED]
    assert sorted(shared_reqs) == ["D", "E"]


if __name__ == "__main__":
    test_reload_next_admitted_user()
    test_exclusive_vs_shared()
    print("PASS: reload -> next-admitted-user (zero-touch); req-36-style reuse -> 0 EXACT; "
          "ties -> attribution-ambiguous; one reload -> one owner (no double-charge).")
