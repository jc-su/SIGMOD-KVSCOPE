"""The attention-share statistic recorded by a kernel of its own, for any attention backend.

The compiler pass records the statistic inside the engine's Triton attention kernel. A backend whose
kernels are not Triton (FlashInfer, FlashAttention) has no such kernel to instrument. This module
records the same statistic with a separate Triton kernel that runs after each attention layer:

  m[layer][row][block][query row] = the largest score over the keys of the block's tile
  l[layer][row][block][query row] = the sum of exp(score - m) over those keys

for the query rows of one KV group per layer (layer L records group L mod groups), only for the rows
of an audit step. The layout, the units, and the descriptor files equal the pass's, so the controller
reads either one with the same code (``observe.tile_shares``).

The kernel reads the keys of one KV group of the audited rows. On an audit step the engine reads
those rows in full already, so the statistic costs one more read of one eighth of their keys per
layer. On every other step the kernel starts, sees the flag off, and returns.
"""

import ctypes
import json
import os

import numpy as np
import torch
import triton
import triton.language as tl

from observe.zero_touch import pub_path
from runtime.audit import AuditPlane

TILE_BLOCKS = 2  # blocks per tile: the kernel records one value per tile, as the pass does


@triton.jit
def tile_lse_kernel(
    q_ptr,
    k_ptr,
    table_ptr,
    lens_ptr,
    qrow_ptr,
    select_ptr,
    flag_ptr,
    lse_ptr,
    scale,
    layer,
    group,
    q_stride_token,
    q_stride_head,
    k_stride_block,
    k_stride_pos,
    k_stride_head,
    table_stride,
    lse_layer_stride,
    lse_row_stride,
    lse_plane_stride,
    BLOCK_SIZE: tl.constexpr,
    TILE_BLOCKS: tl.constexpr,
    GROUP_ROWS: tl.constexpr,
    ROWS_PAD: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    if tl.load(flag_ptr) != 0:  # noqa: SIM102  # Triton branches on scalars, without short-circuit
        if tl.load(select_ptr + row) != 0:
            length = tl.load(lens_ptr + row)
            tile_tokens: tl.constexpr = TILE_BLOCKS * BLOCK_SIZE
            first = tile * tile_tokens
            if first < length:
                pos = first + tl.arange(0, tile_tokens)
                valid = pos < length
                block = tl.load(table_ptr + row * table_stride + pos // BLOCK_SIZE, mask=valid, other=0)
                dims = tl.arange(0, HEAD_DIM)
                k = tl.load(
                    k_ptr
                    + block[:, None].to(tl.int64) * k_stride_block
                    + (pos % BLOCK_SIZE)[:, None] * k_stride_pos
                    + group * k_stride_head
                    + dims[None, :],
                    mask=valid[:, None],
                    other=0.0,
                )
                rows = tl.arange(0, ROWS_PAD)
                token = tl.load(qrow_ptr + row)
                q = tl.load(
                    q_ptr
                    + token.to(tl.int64) * q_stride_token
                    + (group * GROUP_ROWS + rows)[:, None] * q_stride_head
                    + dims[None, :],
                    mask=(rows < GROUP_ROWS)[:, None],
                    other=0.0,
                )
                score = tl.dot(q, tl.trans(k)) * scale
                score = tl.where(valid[None, :], score, float("-inf"))
                peak = tl.max(score, 1)
                total = tl.sum(tl.exp(score - peak[:, None]), 1)
                for i in tl.static_range(TILE_BLOCKS):
                    block_index = tile * TILE_BLOCKS + i
                    if block_index * BLOCK_SIZE < length:
                        cell = (
                            lse_ptr
                            + layer * lse_layer_stride
                            + row * lse_row_stride
                            + block_index * GROUP_ROWS
                            + rows
                        )
                        # plane 0 holds the tile max, plane 1 the tile sum
                        tl.store(cell, peak, mask=rows < GROUP_ROWS)
                        tl.store(cell + lse_plane_stride, total, mask=rows < GROUP_ROWS)


class _Device:
    """A device allocation that Triton accepts as a pointer argument: it needs an address and an element type."""

    def __init__(self, address: int, dtype: torch.dtype) -> None:
        self.address = address
        self.dtype = dtype

    def data_ptr(self) -> int:
        return self.address


class RecordingPlane(AuditPlane):
    """An audit plane whose statistic block is allocated here and recorded by ``tile_lse_kernel``."""

    def __init__(self, layers: int, rows: int, width: int, group_rows: int, block_size: int) -> None:
        super().__init__()
        if group_rows > 16:
            raise ValueError(f"{group_rows} query heads per KV head: the kernel handles at most 16")
        if (block_size * TILE_BLOCKS) & (block_size * TILE_BLOCKS - 1):
            raise ValueError("a tile of the audit kernel must hold a power of two of tokens")
        self._layers = layers
        self.rows, self.width, self.group_rows, self.block_size = rows, width, group_rows, block_size
        self._layer_of: dict[str, int] = {}  # layer name -> layer number, in the order the layers first run
        lse_bytes = 2 * layers * rows * width * group_rows * 4  # two planes: the tile max and the tile sum
        self._lse = self._allocate(lse_bytes)
        self._cuda.cudaMemset(ctypes.c_void_p(self._lse), 0xFF, ctypes.c_size_t(lse_bytes))  # NaN: unwritten
        self._flag_dev = self._allocate(4)
        self._table_dev = self._allocate(layers * 8)
        self._cuda.cudaMemset(ctypes.c_void_p(self._flag_dev), 0, ctypes.c_size_t(4))
        self._cuda.cudaMemset(ctypes.c_void_p(self._table_dev), 0, ctypes.c_size_t(layers * 8))
        self._select = torch.zeros(rows, dtype=torch.uint8, device="cuda")
        self._token = torch.zeros(rows, dtype=torch.int32, device="cuda")
        self._publish()

    def _allocate(self, size: int) -> int:
        pointer = ctypes.c_void_p()
        rc = self._cuda.cudaMalloc(ctypes.byref(pointer), ctypes.c_size_t(size))
        if rc != 0 or pointer.value is None:
            raise RuntimeError(f"cudaMalloc of {size} bytes failed with code {rc}")
        return pointer.value

    def _ipc(self, address: int) -> str:
        handle = (ctypes.c_ubyte * 64)()
        rc = self._cuda.cudaIpcGetMemHandle(ctypes.byref(handle), ctypes.c_void_p(address))
        if rc != 0:
            raise RuntimeError(f"cudaIpcGetMemHandle failed with code {rc}")
        return bytes(handle).hex()

    def _publish(self) -> None:
        """Write the descriptor files in the format the pass writes, so a controller reads this block unchanged."""
        entry = {
            "host_va": self._lse,
            "ncap": self.width,
            "max_rows": self.rows,
            "tile_lse_dev": self._lse,
            "tile_lse_ipc": self._ipc(self._lse),
            "tile_lse_layers": self._layers,
            "tile_lse_group_rows": self.group_rows,
            "stat_flag_dev": self._flag_dev,
            "layer_table_dev": self._table_dev,
            "layer_table_ipc": self._ipc(self._table_dev),
        }
        path = pub_path(self.pid)
        with open(path, "a") as f:
            f.write(json.dumps(entry) + "\n")
        os.chmod(path, 0o600)
        coverage = {"host_va": self._lse, "attn_share_tile_blocks": TILE_BLOCKS}
        with open(path.replace("ekv_pass_", "ekv_coverage_"), "a") as f:
            f.write(json.dumps(coverage) + "\n")

    def _discover(self) -> bool:
        return True  # the block exists from construction

    @property
    def layer_names(self) -> list[str]:
        """The attention layers in the order they first ran, which is their layer number."""
        return sorted(self._layer_of, key=lambda name: self._layer_of[name])

    def select(self, audited: np.ndarray, token_start: np.ndarray) -> None:
        """Choose the rows to record in the next step, and the token index of each row's query."""
        rows = len(audited)
        self._select[:rows].copy_(torch.from_numpy(audited.astype(np.uint8)))
        self._token[:rows].copy_(torch.from_numpy(token_start.astype(np.int32)))

    def record(
        self,
        layer_name: str,
        table: torch.Tensor,
        lens: torch.Tensor,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        scale: float,
    ) -> None:
        """Launch the kernel for one attention layer, after the layer has attended.

        A layer is numbered by its name, in the order the layers first run. The engine runs every
        layer once with small placeholder caches while it profiles memory, so the cache address of a
        layer is not stable. The name is.

        ``table`` and ``lens`` are the view's buffers, which hold the audited rows' whole tables.
        The kernel checks the flag on the device, so this launch is the same on every step and a
        CUDA graph can capture it.
        """
        if kv_cache.ndim != 5 or kv_cache.shape[1] != 2:
            raise ValueError(
                f"expected a KV cache [blocks, 2, block, heads, dim], got {tuple(kv_cache.shape)}"
            )
        if kv_cache.shape[2] != self.block_size:
            raise ValueError("the KV cache's block size differs from the view's")
        layer = self._layer_of.setdefault(layer_name, len(self._layer_of))
        self.launch(layer, table, lens, query, kv_cache[:, 0], scale)

    def launch(
        self,
        layer: int,
        table: torch.Tensor,
        lens: torch.Tensor,
        query: torch.Tensor,
        keys: torch.Tensor,
        scale: float,
    ) -> None:
        """Launch the kernel for ``layer``; ``keys`` is the layer's key cache [blocks, block, heads, dim]."""
        if layer >= self._layers:
            raise RuntimeError(f"more attention layers ran than the {self._layers} the plane was sized for")
        kv_heads, dim = keys.shape[2], keys.shape[3]
        if kv_heads * self.group_rows != query.shape[1] or dim != query.shape[2]:
            raise ValueError("the query and KV cache shapes do not match the plane's group size")
        tile_lse_kernel[(self.rows, triton.cdiv(self.width, TILE_BLOCKS))](
            query,
            keys,
            table,
            lens,
            self._token,
            self._select,
            _Device(self._flag_dev, torch.uint32),
            _Device(self._lse, torch.float32),
            scale,
            layer,
            layer % kv_heads,
            query.stride(0),
            query.stride(1),
            keys.stride(0),
            keys.stride(1),
            keys.stride(2),
            table.stride(0),
            self.rows * self.width * self.group_rows,
            self.width * self.group_rows,
            self._layers * self.rows * self.width * self.group_rows,
            BLOCK_SIZE=self.block_size,
            TILE_BLOCKS=TILE_BLOCKS,
            GROUP_ROWS=self.group_rows,
            ROWS_PAD=16,
            HEAD_DIM=dim,
        )
