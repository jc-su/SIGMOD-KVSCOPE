"""Domain vocabulary for eKV — the type-level distinctions that keep the three
observation faces honest.

Everything downstream speaks in these names instead of bare ``int``/``str``, so the
type checker rejects, say, passing a device ``Address`` where a ``ContentHash`` is
expected (both are ints at runtime, but they are NOT the same thing). ``NewType`` is
zero-cost: ``Address(x)`` is just ``x`` at runtime — the nominal distinction lives in
the checker, the clarity lives in the reader.
"""
from __future__ import annotations

from enum import StrEnum
from typing import Literal, NewType, TypeAlias

# ── identities & quantities (nominal ints) ──────────────────────────────────
Address = NewType("Address", int)            # a device- or host-side virtual address
SizeBytes = NewType("SizeBytes", int)        # a transfer / region size, in bytes
Nanoseconds = NewType("Nanoseconds", int)    # a duration or wall-clock instant, in ns
Cycles = NewType("Cycles", int)              # GPU clock64 cycles
ContentHash = NewType("ContentHash", int)    # vLLM block content hash — the cross-rank key
StreamHandle = NewType("StreamHandle", int)  # a CUDA stream pointer
EventHandle = NewType("EventHandle", int)    # a CUDA event pointer
Pid = NewType("Pid", int)
RequestId = NewType("RequestId", str)        # a serving-request id

# a physical KV block: a plain id, or (layer, block) once resolved per layer
BlockId: TypeAlias = int | tuple[int, int]

# ── closed string sets the wire formats actually use ────────────────────────
# Directions the fuser acts on. H2H is intentionally absent — it's never a KV tier
# movement, so the reader drops it. DEFAULT (cudaMemcpyDefault) has no declared
# direction: the reader carries BOTH addresses and the fuser infers offload/load
# from which side is a KV block (never guesses). BATCH = a driver-batch element.
Direction = Literal["H2D", "D2H", "D2D", "BATCH", "DEFAULT"]
CompletionVia = Literal["stream-sync", "event-sync", "event-query"]
Weight = Literal["cost", "bytes"]                        # the quantity a flame's width measures
# how a request touched a KV block:
KVRole = Literal["hit", "allocate", "store", "evict", "cache_evict", "load", "offload"]
# "evict"        = a request released its blocks (free; the content may STILL be cached)
# "cache_evict"  = the block pool actually removed content from the cache; THIS is the
#                  true recompute trigger — a freed block stays cached under prefix
#                  caching, so free() is not eviction.


class Tier(StrEnum):
    """A location in the placement model σ. The *value* is vLLM's ``medium`` string,
    so a ``Tier`` compares and formats as the raw medium it came from."""
    HBM = "GPU"          # vLLM medium == "GPU"
    DRAM = "CPU"
    SSD = "SSD"
    REMOTE = "REMOTE"
    RECOMPUTE = "BOT"    # the bottom location: regenerate from tokens


class Op(StrEnum):
    """The kind of causal edge a record represents."""
    STORE = "store"      # a block became cached
    EVICT = "evict"      # a cached block was removed
    COPY = "copy"        # a physical transfer between tiers (eBPF)
    COMPUTE = "compute"  # GPU compute / KV read (Triton timer)


# cudaMemcpyKind direction -> the (src, dst) tiers it moves between
_DIR_TIERS: dict[str, tuple[Tier, Tier]] = {
    "H2D": (Tier.DRAM, Tier.HBM),
    "D2H": (Tier.HBM, Tier.DRAM),
    "D2D": (Tier.HBM, Tier.HBM),
}


def dir_tiers(direction: Direction) -> tuple[Tier | None, Tier | None]:
    """The (src, dst) tiers a copy direction moves between; (None, None) if unknown
    (e.g. a driver BATCH element, whose per-element direction the fuser infers)."""
    return _DIR_TIERS.get(direction, (None, None))
