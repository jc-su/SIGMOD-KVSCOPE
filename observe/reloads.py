"""Reloads of KV blocks from host memory, read live from the eBPF ``kv_copy`` capture.

vLLM's native CPU KV offload moves blocks between the GPU KV cache and pinned host memory with
``cuMemcpyBatchAsync`` on a transfer stream, records a CUDA event after the batch, and polls the event. The
collector records each block of a batch (direction ``BATCH``, both addresses), the binding of the stream's
copies to the event, and the first successful poll of the event.

The addresses alone do not say which side is GPU memory, so the engine publishes the address ranges of its
GPU KV cache (``publish_kv_cache``, called where the control mailbox is created). A batch copy whose
destination lies in the GPU KV cache is a reload (host memory to GPU); one whose source lies there is a
store. A reload is in flight from its enqueue until the completion that the collector records, which is the
first poll that found the event complete, so the interval is an upper bound of the copy's time on the GPU.

``ReloadTracker`` tails a growing capture file. Its watermark is the time of the newest record it has read:
the collector also records the engine's own runtime copies, which every engine step makes, so the watermark
passes the end of a step soon after the step ends. A window that ends before the watermark is complete:
every reload that started in it has been read.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, assert_never

from .ebpf_reader import parse_records
from .events import Bind, Completion, Enqueue, Gap

if TYPE_CHECKING:
    import torch


def kv_cache_path(pid: int) -> Path:
    """Where engine process ``pid`` publishes the address ranges of its GPU KV cache, next to its mailbox."""
    return Path(os.environ.get("EKV_CONTROL_DIR", "/dev/shm")) / f"ekv_kv_cache_{int(pid)}.json"


def kv_cache_ranges(caches: Iterable[torch.Tensor]) -> list[tuple[int, int]]:
    """The address ranges of the storages that hold the GPU KV cache tensors ``caches``, sorted.

    Several layers can be views of one storage (vLLM lays the layers out in one buffer for its CPU offload), so
    a range is a storage, counted once."""
    storages = {c.untyped_storage().data_ptr(): c.untyped_storage().nbytes() for c in caches}
    if not storages:
        raise ValueError("the engine has no KV cache tensor")
    return sorted((base, base + size) for base, size in storages.items())


def publish_kv_cache(pid: int, ranges: list[tuple[int, int]]) -> None:
    path = kv_cache_path(pid)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"kv_cache": ranges}))
    os.replace(tmp, path)  # a reader never sees a partial file


def read_kv_cache(pid: int) -> list[tuple[int, int]]:
    """The GPU KV cache ranges that engine process ``pid`` published. Raises ``RuntimeError`` if it published
    none: it runs without KVScope's safe point, or has not run a step yet."""
    path = kv_cache_path(pid)
    if not path.exists():
        raise RuntimeError(f"engine pid {pid} has not published its GPU KV cache ({path})")
    return [(int(lo), int(hi)) for lo, hi in json.loads(path.read_text())["kv_cache"]]


@dataclass
class Reload:
    """One reload job: the blocks of one batch copy into GPU memory."""

    start_ns: int  # when the engine enqueued the batch
    nbytes: int
    blocks: int
    done_ns: int | None = None  # the first poll that found it complete; None while in flight


class ReloadTracker:
    """The reloads in a live ``kv_copy`` capture of one engine process."""

    def __init__(self, capture: Path, kv_cache: list[tuple[int, int]]) -> None:
        """``kv_cache`` is the address ranges of the engine's GPU KV cache (``read_kv_cache``)."""
        if not kv_cache:
            raise ValueError("the GPU KV cache has no address range")
        self.capture = capture
        self.kv_cache = kv_cache
        self.offset = 0  # bytes of the capture consumed: complete lines only
        self.watermark_ns = 0  # the time of the newest record read
        self.reloads: list[Reload] = []  # in the order of their start, completed or not
        self.read_count = 0  # reloads read since the start, also those forgotten
        self.read_bytes = 0
        self._pending: dict[int, list[Reload]] = {}  # stream or event handle -> reloads not yet complete
        # the (stream, enqueue time) of the newest reload and the reload: the blocks of one batch are
        # consecutive records, which a read of the capture may split
        self._last: tuple[tuple[int, int], Reload] | None = None

    def _on_gpu(self, address: int) -> bool:
        return any(lo <= address < hi for lo, hi in self.kv_cache)

    def poll(self) -> None:
        """Read the records that the collector has written since the last poll."""
        with open(self.capture, "rb") as f:
            f.seek(self.offset)
            data = f.read()
        end = data.rfind(b"\n")
        if end < 0:
            return
        self.offset += end + 1
        for record in parse_records(data[:end].decode().splitlines()):
            self.watermark_ns = max(self.watermark_ns, record.at)
            match record:
                case Enqueue():
                    self._enqueue(record)
                case Bind(event, stream, _):
                    moved = self._pending.pop(stream, [])
                    if moved:
                        self._pending.setdefault(event, []).extend(moved)
                case Completion(handle, _, _, at):
                    for reload in self._pending.pop(handle, []):
                        reload.done_ns = at
                case Gap(_, reason):
                    raise RuntimeError(f"the copy collector lost records ({reason}): reloads are not known")
                case _ as unreachable:
                    assert_never(unreachable)

    def _enqueue(self, record: Enqueue) -> None:
        if record.direction != "BATCH":
            return  # a runtime copy of the engine: it only moves the watermark
        if record.other_addr is None:
            raise ValueError(f"a batch copy without its destination: {record}")
        from_gpu, to_gpu = self._on_gpu(record.addr), self._on_gpu(record.other_addr)
        if from_gpu == to_gpu:
            raise ValueError(f"a batch copy that is not between the GPU KV cache and host memory: {record}")
        if from_gpu:
            return  # a store to host memory
        key = (record.stream, record.at)
        if self._last is None or self._last[0] != key:
            self._last = (key, Reload(start_ns=record.at, nbytes=0, blocks=0))
            self.reloads.append(self._last[1])
            self._pending.setdefault(record.stream, []).append(self._last[1])
            self.read_count += 1
        reload = self._last[1]
        reload.nbytes += record.nbytes
        reload.blocks += 1
        self.read_bytes += record.nbytes

    def in_flight_ns(self, start_ns: int, end_ns: int) -> int:
        """How long, within [start_ns, end_ns), at least one reload was in flight.

        A reload whose completion has not been read is in flight until ``end_ns``."""
        covered, reach = 0, start_ns
        for reload in self.reloads:  # in the order of their start
            lo = max(reload.start_ns, reach)
            hi = min(end_ns if reload.done_ns is None else reload.done_ns, end_ns)
            if hi > lo:
                covered += hi - lo
                reach = hi
        return covered

    def started_bytes(self, start_ns: int, end_ns: int) -> int:
        """The bytes of the reloads that started within [start_ns, end_ns)."""
        return sum(r.nbytes for r in self.reloads if start_ns <= r.start_ns < end_ns)

    def forget_before(self, ns: int) -> None:
        """Drop the reloads that completed before ``ns``: no window asked about later can contain them."""
        self.reloads = [r for r in self.reloads if r.done_ns is None or r.done_ns >= ns]
