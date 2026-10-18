"""CI (no GPU): the KV causal flame carries a per-confidence-class width breakdown, so the
renderer can color each path by how much of its width is exact vs bounded vs ambiguous —
turning "we estimate here" into a visible, structured part of the picture (not a hidden caveat).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from observe import (ATTRIBUTION_AMBIGUOUS, BOUNDED, EXACT,  # noqa: E402
                     build_flame, provenance_by_request)
from observe.events import KVEdge                            # noqa: E402
from observe.types import ContentHash, Op, Tier              # noqa: E402


def _edge(req, cost, ch):
    # a COPY (reload) edge DRAM->HBM, attributed to request `req`, costing `cost` ns
    return KVEdge(op=Op.COPY, block_id=1, content_hash=ContentHash(ch), prefix_hash=None,
                  src_tier=Tier.DRAM, dst_tier=Tier.HBM, nbytes=4096, cost=cost, request=req)


def test_flame_provenance_breakdown():
    # three reloads on the SAME causal path (COPY / dram->hbm / resolved), different confidence
    edges = [_edge("A", 100, 1), _edge("B", 40, 2), _edge("C", 10, 3)]
    prov = {"A": EXACT, "B": BOUNDED, "C": ATTRIBUTION_AMBIGUOUS}
    fl = build_flame(edges, provenance=provenance_by_request(prov))

    # walk to the leaf (copy -> CPU->GPU -> resolved)
    node = fl
    for name in ("copy", "CPU->GPU", "resolved"):
        node = next(c for c in node["children"] if c["name"] == name)
    assert node["value"] == 150                                 # total width
    assert node["provenance"] == {EXACT: 100, BOUNDED: 40, ATTRIBUTION_AMBIGUOUS: 10}
    assert node["confidence"] == EXACT                          # dominant class (widest)

    # the root aggregates the same breakdown
    assert fl["provenance"] == {EXACT: 100, BOUNDED: 40, ATTRIBUTION_AMBIGUOUS: 10}


def test_flame_without_provenance_unchanged():
    edges = [_edge("A", 100, 1)]
    fl = build_flame(edges)                                     # no provenance -> no breakdown key
    assert "provenance" not in fl and "confidence" not in fl
    assert fl["value"] == 100


if __name__ == "__main__":
    test_flame_provenance_breakdown()
    test_flame_without_provenance_unchanged()
    print("PASS: flame nodes carry per-confidence-class width + dominant class for coloring; "
          "no-provenance path is unchanged.")
