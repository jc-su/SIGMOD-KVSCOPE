"""What the compiler pass captured for each request: its block-table row and its attention cycles.

The captured row says which physical blocks the request's attention read, so it is the identity
that joins kernel evidence to lifecycle events (``observe.ledger``) and the check that no request
reads another request's private block. The cycles say how much GPU time each request's attention
took. Both are keyed by the scheduler row (the slot) of the launch that wrote them.

A reader in the serving process reads the buffers directly. A reader in another process maps the
capture buffer through its named shared-memory segment (or ``/proc/<pid>/mem``) and opens the cycle
counter by CUDA IPC handle. Reading never writes serving memory.
"""

from __future__ import annotations

import ctypes
import os

import numpy as np

from observe.cuda import Cuda
from observe.zero_touch import parse_slots, pub_path, read_json_lines, read_proc_mem


class Capture:
    def __init__(self, pid: int) -> None:
        self.pid = pid
        self.local = pid == os.getpid()
        self.cuda = Cuda()
        self._timers: dict[int, int] = {}  # descriptor timer address -> readable device address
        self._previous: dict[int, np.ndarray] = {}

    def _entries(self, key: str) -> list[dict]:
        """The published descriptors of kernels that carry ``key``; each launch variant has its own."""
        return [e for e in read_json_lines(pub_path(self.pid)) if int(e.get(key, 0) or 0)]

    def _capture_bytes(self, entry: dict) -> bytes | None:
        size = int(entry["bytes"])
        if self.local:
            return ctypes.string_at(int(entry["host_va"]), size)
        segment = str(entry.get("cap_shm", ""))
        if segment:
            try:
                with open(f"/dev/shm{segment}", "rb") as f:
                    return f.read(size)
            except FileNotFoundError:
                # A descriptor file outlives its process. The segment of a dead earlier process
                # with this pid is gone, and reading the old address would read unrelated memory.
                return None
        return read_proc_mem(self.pid, int(entry["host_va"]), size)

    def buffers(self) -> list[dict[int, list[int]]]:
        """Return ``{slot: block ids}`` of each capture buffer: the rows the last launch of that kernel variant read."""
        out: list[dict[int, list[int]]] = []
        for entry in self._entries("host_va"):
            if not int(entry.get("bytes", 0) or 0):
                continue
            raw = self._capture_bytes(entry)
            if raw is None:
                continue
            slots = parse_slots(raw, int(entry["ncap"]), int(entry["max_rows"]))
            out.append({slot: [int(x) for x in row if x > 0] for slot, row in slots.items()})
        return out

    def rows(self) -> dict[int, list[int]]:
        """Return ``{slot: block ids}`` of the longest completed row of each slot.

        Prefill and decode launches keep separate buffers, and several can hold the same slot, so
        the longest row wins over whichever buffer is read last.
        """
        out: dict[int, list[int]] = {}
        for buffer in self.buffers():
            for slot, blocks in buffer.items():
                if len(blocks) > 2 and len(blocks) > len(out.get(slot, ())):
                    out[slot] = blocks
        return out

    def cycle_deltas(self) -> dict[int, int]:
        """Return ``{slot: cycles}`` accumulated since the previous call.

        The first call records a baseline and returns nothing, so cycles accumulated before the
        reader started (warm-up, graph capture) never appear.
        """
        out: dict[int, int] = {}
        for entry in self._entries("timer_dev"):
            size = int(entry.get("timer_bytes", 0) or 0)
            if not size:
                continue
            key = int(entry["timer_dev"])
            if key not in self._timers:
                self._timers[key] = key if self.local else self.cuda.ipc_open(str(entry["timer_ipc"]))
            current = np.frombuffer(self.cuda.d2h(self._timers[key], size), dtype=np.int64).copy()
            previous = self._previous.get(key)
            self._previous[key] = current
            if previous is None:
                continue
            delta = current - previous
            for slot in np.nonzero(delta > 0)[0]:
                out[int(slot)] = out.get(int(slot), 0) + int(delta[slot])
        return out
