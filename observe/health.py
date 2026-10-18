"""Capture-health metrics aggregated across compiler and eBPF observations.

Each source already counts its own integrity
signals; this folds them into a single ``CaptureHealth`` with an ``ok`` gate so a
serving deployment can alarm/scrape on capture loss instead of trusting a silent 0.

  * eBPF kv_copy collector (one BPF ring buffer carries every record):
      - copy_drops       : copy records lost on a failed ring buffer reservation (stats[0])
      - batch_truncated  : blocks beyond KV_BATCH_MAX in one cuMemcpyBatchAsync (stats[1])
      - batch_read_faults: blocks in chunks skipped on a host-read fault (stats[2])
      - meta_drops       : metadata blobs lost on a failed ring buffer reservation (stats[3])
  * eBPF kv_alloc collector:
      - alloc_drops      : allocation records lost on a failed ring buffer reservation
  * reader integrity     : copy_unmatched / copy_overflow / copy_uncosted (reader CaptureReport)
  * identity / fusion    : requests seen, edges produced

`ok` is False if any face lost data.
"""

from __future__ import annotations

import re
from collections.abc import Collection
from dataclasses import asdict, dataclass
from typing import TypedDict

from .ebpf_reader import CaptureReport


class CollectorLoss(TypedDict):
    """The kv_copy collector's four loss counters, as ``collect_health`` keyword arguments."""

    copy_drops: int
    batch_truncated: int
    batch_read_faults: int
    meta_drops: int


@dataclass(frozen=True, slots=True)
class CaptureHealth:
    # eBPF kv_copy collector (ring buffer transport and batch capture)
    copy_drops: int = 0  # copy records lost on a failed reservation (stats[0])
    batch_truncated: int = 0  # blocks beyond KV_BATCH_MAX in a batch (stats[1])
    batch_read_faults: int = 0  # blocks in chunks skipped on a host-read fault (stats[2])
    meta_drops: int = 0  # metadata blobs lost on a failed reservation (stats[3])
    alloc_drops: int = 0  # kv_alloc records lost on a failed reservation
    # reader integrity
    copy_unmatched: int = 0  # completions for an unknown handle (mid-flight/drop)
    copy_overflow: int = 0  # completions claiming more copies than captured
    copy_uncosted: int = 0  # copies still pending at window end (not an error)
    collector_gaps: int = 0  # collector restarts in the window (records lost across each)
    # identity / fusion
    requests_seen: int = 0
    edges: int = 0

    @property
    def ok(self) -> bool:
        """No data lost: every face captured completely (pending-uncosted is allowed —
        those copies just haven't synced yet)."""
        return self.lost == 0

    @property
    def lost(self) -> int:
        """Total records known to be lost across faces (for a single SLO metric)."""
        return (
            self.copy_drops
            + self.batch_truncated
            + self.batch_read_faults
            + self.meta_drops
            + self.alloc_drops
            + self.copy_unmatched
            + self.copy_overflow
            + self.collector_gaps
        )

    def as_dict(self) -> dict[str, int | bool]:
        d: dict[str, int | bool] = dict(asdict(self))
        d["ok"] = self.ok
        d["lost"] = self.lost
        return d


def collect_health(
    *,
    capture: CaptureReport | None = None,
    copy_drops: int = 0,
    batch_truncated: int = 0,
    batch_read_faults: int = 0,
    meta_drops: int = 0,
    alloc_drops: int = 0,
    requests: Collection[object] | int | None = None,
    edges: Collection[object] | int | None = None,
) -> CaptureHealth:
    """Fold the per-face counters into one health view.

    capture           : ebpf_reader.CaptureReport (copy integrity)
    copy_drops        : collector stats[0] (copy records lost on a failed reservation)
    batch_truncated   : collector stats[1] (blocks beyond KV_BATCH_MAX)
    batch_read_faults : collector stats[2] (chunks skipped on a read fault)
    meta_drops        : collector stats[3] (metadata blobs lost on a failed reservation)
    alloc_drops       : kv_alloc's "ringbuf drops" count (parse_ringbuf_drops)
    requests          : request ids (or a count) seen by the identity face
    edges             : the fused KVEdges (or a count)."""
    n_req = requests if isinstance(requests, int) else len(set(requests or ()))
    n_edges = edges if isinstance(edges, int) else len(edges or ())
    return CaptureHealth(
        copy_drops=copy_drops,
        batch_truncated=batch_truncated,
        batch_read_faults=batch_read_faults,
        meta_drops=meta_drops,
        alloc_drops=alloc_drops,
        copy_unmatched=capture.unmatched_completions if capture else 0,
        copy_overflow=capture.overflowed_completions if capture else 0,
        copy_uncosted=capture.uncosted if capture else 0,
        collector_gaps=capture.gaps if capture else 0,
        requests_seen=n_req,
        edges=n_edges,
    )


# The kv_copy collector prints its loss counters to stderr at shutdown; this maps each
# reported line to a collect_health kwarg so a deployment can source health from the
# collector's own audit output (keeping the loss-mode names in ONE place).
_REPORT_LINES: dict[str, re.Pattern[str]] = {
    "copy_drops": re.compile(r"^ringbuf drops \(copy records\):\s*(\d+)$", re.MULTILINE),
    "batch_truncated": re.compile(r"^batch truncation[^:]*:\s*(\d+)$", re.MULTILINE),
    "batch_read_faults": re.compile(r"^batch read faults[^:]*:\s*(\d+)$", re.MULTILINE),
    "meta_drops": re.compile(r"^ringbuf drops \(metadata\):\s*(\d+)$", re.MULTILINE),
}


def parse_collector_report(text: str) -> CollectorLoss:
    """Extract the four loss counters from the kv_copy collector's stderr report.

    Raises ``ValueError`` if a counter line is missing: the collector prints all four at a clean
    exit, so a missing line means it crashed or was killed before its report."""
    found: dict[str, int] = {}
    for key, pat in _REPORT_LINES.items():
        m = pat.search(text)
        if m:
            found[key] = int(m.group(1))
    missing = [key for key in _REPORT_LINES if key not in found]
    if missing:
        raise ValueError(f"kv_copy collector report is missing the loss counters {missing}")
    return CollectorLoss(
        copy_drops=found["copy_drops"],
        batch_truncated=found["batch_truncated"],
        batch_read_faults=found["batch_read_faults"],
        meta_drops=found["meta_drops"],
    )


_RINGBUF_DROPS = re.compile(r"^ringbuf drops:\s*(\d+)$", re.MULTILINE)


def parse_ringbuf_drops(text: str) -> int:
    """The ring buffer drop count that kv_alloc, storage_read, ekv_readout, and http_label print to
    stderr at exit ("ringbuf drops: N").

    Raises ``ValueError`` if the line is missing: each of them prints it at every exit after a
    successful load, so a missing line means it crashed or was killed before its report."""
    m = _RINGBUF_DROPS.search(text)
    if m is None:
        raise ValueError("collector report has no 'ringbuf drops' line")
    return int(m.group(1))
