"""Fold ``KVEdge``s into a KV Causal Flame — the hierarchical, additive-width tree
the HTML prototype renders.

Width is an additive quantity (summed ``cost`` ns, or ``bytes``); the path encodes
the causal structure (op → tier-transition → resolution). That is the flame-graph
methodology applied to KV edges: width sums, the hierarchy is the causal path, and
two flames subtract for differential mode (red = worse, blue = better).
"""
from __future__ import annotations

import json
from collections.abc import Iterable

from .events import KVEdge
from .types import Weight


class _Node:
    __slots__ = ("name", "value", "children", "prov")

    def __init__(self, name: str):
        self.name = name
        self.value = 0.0
        self.children: dict[str, _Node] = {}
        self.prov: dict[str, float] = {}     # confidence class -> width contributed

    def add(self, path: list[str], w: float, pc: str | None = None) -> None:
        self.value += w
        if pc is not None:
            self.prov[pc] = self.prov.get(pc, 0.0) + w
        node = self
        for part in path:
            child = node.children.get(part)
            if child is None:
                child = node.children[part] = _Node(part)
            child.value += w
            if pc is not None:
                child.prov[pc] = child.prov.get(pc, 0.0) + w
            node = child

    def to_dict(self) -> dict:
        d = {
            "name": self.name,
            "value": round(self.value, 3),
            "children": [c.to_dict() for c in sorted(
                self.children.values(), key=lambda n: -n.value)],
        }
        if self.prov:                        # per-class width + the dominant class (for coloring)
            d["provenance"] = {k: round(v, 3) for k, v in self.prov.items()}
            d["confidence"] = max(self.prov, key=self.prov.get)
        return d


def _weight(e: KVEdge, w: Weight) -> float:
    v = e.cost if w == "cost" else e.nbytes
    return float(v) if v else 0.0


def _path(e: KVEdge) -> list[str]:
    src = e.src_tier.value if e.src_tier is not None else "?"
    dst = e.dst_tier.value if e.dst_tier is not None else "?"
    consequence = "resolved" if e.content_hash is not None else "unresolved"
    return [e.op.value, f"{src}->{dst}", consequence]


def _accumulate(edges: Iterable[KVEdge], weight: Weight, provenance=None) -> _Node:
    root = _Node("kv")
    for e in edges:
        w = _weight(e, weight)
        if w:
            root.add(_path(e), w, provenance(e) if provenance else None)
    return root


def build_flame(edges: Iterable[KVEdge], *, weight: Weight = "cost", provenance=None) -> dict:
    """A flame tree whose width is ``cost`` (latency) or ``bytes`` (volume). When ``provenance``
    is given — a callable ``KVEdge -> confidence class`` (EXACT / BOUNDED / IDENTITY_AMBIGUOUS /
    ATTRIBUTION_AMBIGUOUS) — every node also carries a per-class width breakdown and the dominant
    ``confidence``, so the renderer can color/texture each path by how much of its width is hard-
    measured vs estimated vs ambiguous. This makes "we estimate here" a visible, structured part
    of the picture instead of a hidden caveat."""
    return _accumulate(edges, weight, provenance).to_dict()


def provenance_by_request(prov_map: dict, default: str = "exact"):
    """A ``provenance`` callable from a {request_id -> confidence class} map (e.g. from the
    reload-park / identity attribution). Edges with no request, or an unmapped request, get
    ``default``."""
    return lambda e: prov_map.get(getattr(e, "request", None), default)


def differential_flame(before: Iterable[KVEdge], after: Iterable[KVEdge], *,
                       weight: Weight = "cost") -> dict:
    """Signed-width tree (after − before) per causal path, for differential mode."""
    a, b = _accumulate(after, weight), _accumulate(before, weight)

    def diff(na: _Node, nb: _Node) -> dict:
        keys = set(na.children) | set(nb.children)
        return {
            "name": na.name,
            "delta": round(na.value - nb.value, 3),
            "children": [diff(na.children.get(k, _Node(k)), nb.children.get(k, _Node(k)))
                         for k in keys],
        }
    return diff(a, b)


def dump_flame(edges: Iterable[KVEdge], path: str, *, weight: Weight = "cost") -> dict:
    fl = build_flame(edges, weight=weight)
    with open(path, "w") as f:
        json.dump(fl, f, indent=2)
    return fl
