"""Per-request KV observation and attribution.

The eBPF collectors and GPU compiler pass require no engine source-code changes and produce:

  request<->block + per-request compute : the eKV MLIR pass (GPU-side capture, read
      externally) ⋈ the eBPF network label    -> net_label / zero_touch
  KV transfer cost (physical, by addr)  : read_copies (eBPF kv_copy)  -> Copy
                         │
       fuse_copies (via BlockAddressMap) ; async-load park via latency_attribution.blocking_from_loads
                         ▼
               KVEdge / RequestLatency  ──build_flame──>  KV Causal Flame

The vocabulary lives in ``types`` (NewTypes + Literals + Tier/Op); the records in
``events``; the address bridge in ``blockmap``.
"""
# domain vocabulary
from .types import (Address, BlockId, CompletionVia, ContentHash, Cycles,  # noqa: F401
                    Direction, EventHandle, KVRole, Nanoseconds, Op, Pid,
                    RequestId, SizeBytes, StreamHandle, Tier, Weight, dir_tiers)
# records
from .events import (Bind, BlockingEvent, Completion, Copy, CopyRecord,  # noqa: F401
                     Enqueue, KVEdge, RequestKVUse)
# the address bridge + the faces + the fuser + the flame
from .blockmap import (Block, BlockAddressMap, content_hashes_from_tokens,  # noqa: F401
                       identify_kv_pool)
from .ebpf_reader import (CaptureReport, correlate, parse_records,  # noqa: F401
                          parse_records_binary, read_copies, read_copies_binary,
                          read_copies_binary_file, read_copies_file,
                          read_copies_file_report, read_copies_report)
from .fuse import dedup_by_hash, fuse_copies  # noqa: F401
from .flame import (build_flame, differential_flame, dump_flame,  # noqa: F401
                    provenance_by_request)
from .health import (CaptureHealth, collect_health,  # noqa: F401
                     parse_collector_report, parse_ringbuf_drops)
from .latency_attribution import (ATTRIBUTION_AMBIGUOUS, BOUNDED, EXACT,  # noqa: F401
                   IDENTITY_AMBIGUOUS, SHARED, PathLatency, RequestLatency, attribute_latency,
                   attribute_reload_parks, blocking_from_loads, fold_paths,
                   parks_from_schedule, split_admission_delay,
                   split_admission_delay_pass, summarize, tier_miss_costs)
from .recompute import classify_rematerializations  # noqa: F401

__all__ = [
    # vocabulary
    "Address", "SizeBytes", "Nanoseconds", "Cycles", "ContentHash", "StreamHandle",
    "EventHandle", "Pid", "RequestId", "BlockId", "Direction", "CompletionVia",
    "Weight", "Tier", "Op", "dir_tiers",
    # records
    "Enqueue", "Bind", "Completion", "CopyRecord", "Copy",
    "KVEdge", "RequestKVUse", "KVRole", "BlockingEvent",
    # latency-attribution model (per-request KV-induced latency + causal-path fold). async_load park is
    # reconstructed zero-touch from the eBPF KV-load transfers via blocking_from_loads.
    "RequestLatency", "PathLatency", "attribute_latency", "blocking_from_loads",
    "parks_from_schedule", "split_admission_delay", "split_admission_delay_pass",
    "attribute_reload_parks", "tier_miss_costs", "EXACT", "BOUNDED", "SHARED",
    "IDENTITY_AMBIGUOUS", "ATTRIBUTION_AMBIGUOUS", "fold_paths", "summarize",
    "classify_rematerializations",
    # health / metrics
    "CaptureHealth", "collect_health", "parse_collector_report", "parse_ringbuf_drops",
    # address bridge + faces + fusion + flame
    "Block", "BlockAddressMap", "content_hashes_from_tokens", "identify_kv_pool",
    "parse_records", "correlate", "CaptureReport",
    "read_copies", "read_copies_file", "read_copies_report", "read_copies_file_report",
    "parse_records_binary", "read_copies_binary", "read_copies_binary_file",
    "fuse_copies", "dedup_by_hash", "build_flame", "differential_flame",
    "dump_flame", "provenance_by_request",
]
