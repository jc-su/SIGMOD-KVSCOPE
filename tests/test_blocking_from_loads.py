"""CI (no GPU): blocking_from_loads turns eBPF KV-load transfers into the SAME
async_load BlockingEvents the scheduler hook produced — so the park is recovered from
the data plane, no scheduler hook. A request parked until ALL its loads finished, so its
park = [first enqueue, last completion] across its loads."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from observe.latency_attribution import attribute_latency, blocking_from_loads


def main():
    # eBPF-captured KV loads: (blocks, enqueue_ns, completion_ns). Request A loads in two
    # transfers; B in one; some blocks belong to neither (other requests / system).
    loads = [
        (("10", "11"), 100, 250),     # A's first load
        ((20,), 120, 300),            # B's load
        ((12,), 260, 400),            # A's second load (later)
        ((99,), 130, 140),            # unattributed block -> ignored
    ]
    b2r = {10: "A", 11: "A", 12: "A", 20: "B"}
    evs = {e.request_id: e for e in blocking_from_loads(loads, b2r)}

    assert set(evs) == {"A", "B"}, evs
    # A parked across BOTH its loads: [min enqueue 100, max completion 400] = 300
    assert evs["A"].entered_at == 100 and evs["A"].exited_at == 400, evs["A"]
    assert evs["A"].duration == 300 and evs["A"].n_steps == 2, evs["A"]
    assert set(evs["A"].blocks) == {10, 11, 12}, evs["A"]
    # B: [120, 300] = 180, one load
    assert evs["B"].duration == 180 and evs["B"].n_steps == 1, evs["B"]
    print(f"A park={evs['A'].duration}ns over {evs['A'].n_steps} loads, blocks={evs['A'].blocks}")
    print(f"B park={evs['B'].duration}ns over {evs['B'].n_steps} loads")

    # drops straight into attribute_latency as the async_load source (replacing sched_hook).
    latency = attribute_latency(blocking=blocking_from_loads(loads, b2r))
    assert latency["A"].async_load == 300 and latency["B"].async_load == 180
    assert latency["A"].total == 300, latency["A"]
    print(f"attribute_latency: A async_load={latency['A'].async_load}ns B async_load={latency['B'].async_load}ns")
    print("PASS: async_load park reconstructed from eBPF KV-load transfers (no scheduler hook), "
          "aggregated per request, drop-in for attribute_latency.")


if __name__ == "__main__":
    main()
