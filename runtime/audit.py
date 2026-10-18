"""The serving-process side of the attention-share statistic.

The compiler pass records tile log-sum-exp values only when a device flag is set, and it finds the
layer of a launch from the position of the kernel's KV pointer in a table of per-layer cache
addresses. Both live in device memory that the pass allocates and describes in its published
descriptor. ``AuditPlane`` fills the table once the engine has allocated its KV caches, and sets
the flag for audit steps.
"""

import ctypes
import json
import os

import numpy as np


def _cudart() -> ctypes.CDLL:
    for name in ("libcudart.so.12", "libcudart.so"):
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    raise RuntimeError("libcudart not found")


class AuditPlane:
    """Fills the layer table and sets the audit flag of one serving process."""

    def __init__(self, pid: int | None = None) -> None:
        self.pid = os.getpid() if pid is None else pid
        self._cuda = _cudart()
        self._flag_dev = 0
        self._table_dev = 0
        self._layers = 0
        self._table_filled = False
        self._flag_on: bool | None = None
        self._newest_rows_checked = False

    def _discover(self) -> bool:
        """Read the pass's descriptor for the shared statistic block. False if no kernel has one yet."""
        directory = os.environ.get("EKV_RUNTIME_DIR") or "/tmp"
        try:
            with open(f"{directory}/ekv_pass_{self.pid}.json") as f:
                for line in f:
                    entry = json.loads(line)
                    if int(entry.get("stat_flag_dev", 0)):
                        self._flag_dev = int(entry["stat_flag_dev"])
                        self._table_dev = int(entry["layer_table_dev"])
                        self._layers = int(entry["tile_lse_layers"])
                        return True
        except FileNotFoundError:
            return False
        return False

    def _copy_to_device(self, device: int, data: np.ndarray) -> None:
        rc = self._cuda.cudaMemcpy(
            ctypes.c_void_p(device), data.ctypes.data_as(ctypes.c_void_p), ctypes.c_size_t(data.nbytes), 1
        )
        if rc != 0:
            raise RuntimeError(f"cudaMemcpy host to device failed with code {rc}")

    def require_newest_rows(self) -> None:
        """Fail unless every kernel that records the statistic records it only from the program that holds a
        request's newest query tokens, which a prefill step needs: there several programs serve one request,
        and each would store its own query rows into the request's cells. Checked once."""
        if self._newest_rows_checked:
            return
        directory = os.environ.get("EKV_RUNTIME_DIR") or "/tmp"
        with open(f"{directory}/ekv_coverage_{self.pid}.json") as f:
            recording = [entry for entry in map(json.loads, f) if int(entry.get("attn_share_sites", 0))]
        if not recording:
            raise RuntimeError(
                "a prefill audit was requested before any kernel that records the statistic compiled"
            )
        unsafe = [entry["kernel"] for entry in recording if not int(entry.get("attn_share_newest_rows", 0))]
        if unsafe:
            raise RuntimeError(
                f"kernels {unsafe} record the statistic from every program of a request: a prefill step cannot "
                "be an audit step with them"
            )
        self._newest_rows_checked = True

    def select(self, audited: np.ndarray, token_start: np.ndarray) -> None:
        """Choose the rows to record in the next step. The pass records every row it is given and needs none."""

    def attach(self, cache_addresses: list[int]) -> bool:
        """Fill the layer table with the base address of each layer's KV cache. True once it is filled."""
        if self._table_filled:
            return True
        if not self._flag_dev and not self._discover():
            return False
        if len(cache_addresses) > self._layers:
            raise ValueError(
                f"the pass sized its layer table for {self._layers} layers, the engine has {len(cache_addresses)}"
            )
        table = np.zeros(self._layers, dtype=np.uint64)
        table[: len(cache_addresses)] = cache_addresses
        self._copy_to_device(self._table_dev, table)
        self._table_filled = True
        return True

    def set(self, on: bool) -> None:
        """Turn collection on or off for the launches that follow."""
        if on == self._flag_on:
            return
        if not self._flag_dev and not self._discover():
            if on:
                raise RuntimeError("an audit step was requested but the pass published no statistic buffers")
            return
        if on and not self._table_filled:
            raise RuntimeError("an audit step was requested before the layer table was filled")
        self._copy_to_device(self._flag_dev, np.array([1 if on else 0], dtype=np.uint32))
        self._flag_on = on
