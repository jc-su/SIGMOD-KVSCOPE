"""The part of the CUDA runtime that a controller process needs: open another process's device
buffer by IPC handle, and copy device memory to the host.

The controller never writes serving memory, so there is no host-to-device copy here.
"""

from __future__ import annotations

import ctypes


class Cuda:
    def __init__(self) -> None:
        for name in ("libcudart.so.12", "libcudart.so"):
            try:
                self.lib = ctypes.CDLL(name)
                break
            except OSError:
                pass
        else:
            raise RuntimeError("libcudart not found")
        rc = self.lib.cudaFree(ctypes.c_void_p(0))  # creates the context
        if rc != 0:
            raise RuntimeError(f"cuda init rc={rc}")

        # cudaIpcMemHandle_t is a 64-byte struct passed by value. An array argtype decays to a
        # pointer, and a c_char array field converts to an immutable bytes copy, so the handle is
        # a struct of c_ubyte.
        class Handle(ctypes.Structure):
            _fields_ = [("b", ctypes.c_ubyte * 64)]

        self._handle_type = Handle
        self._open = self.lib.cudaIpcOpenMemHandle
        self._open.argtypes = [ctypes.POINTER(ctypes.c_void_p), Handle, ctypes.c_uint]

    def ipc_open(self, hex_handle: str) -> int:
        raw = bytes.fromhex(hex_handle)
        if len(raw) != 64:
            raise ValueError(f"an IPC handle has 64 bytes, got {len(raw)}")
        handle = self._handle_type()
        ctypes.memmove(ctypes.byref(handle), raw, 64)
        pointer = ctypes.c_void_p()
        rc = self._open(ctypes.byref(pointer), handle, 1)  # cudaIpcMemLazyEnablePeerAccess
        if rc != 0:
            raise RuntimeError(f"cudaIpcOpenMemHandle rc={rc}")
        if pointer.value is None:
            raise RuntimeError("cudaIpcOpenMemHandle returned a null pointer")
        return pointer.value

    def d2h(self, dev: int, n_bytes: int) -> bytes:
        out = (ctypes.c_char * n_bytes)()
        rc = self.lib.cudaMemcpy(out, ctypes.c_void_p(dev), ctypes.c_size_t(n_bytes), 2)
        if rc != 0:
            raise RuntimeError(f"cudaMemcpy D2H rc={rc}")
        return bytes(out)

    def d2h_many(self, requests: list[tuple[int, int]]) -> list[bytes]:
        return [self.d2h(dev, size) for dev, size in requests]
