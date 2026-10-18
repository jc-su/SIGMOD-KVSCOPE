"""Return the blocks a request no longer reads to the engine's block pool.

The block-table view makes a request read fewer blocks, and the blocks it leaves out still occupy
GPU memory. When the controller has kept a block out of a request's reads across several audits,
the block can be released: the engine's block pool takes it back, and the memory serves another
request. The engine already does this for sliding-window attention, where the blocks that fall out
of the window are swapped for the pool's null block and freed (``remove_skipped_blocks``). This
module does the same for the blocks the controller names.

Release is permanent. A released block cannot be read again, so the view refuses to read it under
any policy (``DecodeView.released``), including after the lease has expired. It needs prefix caching
off, because a cached block can be shared, and one KV-cache group.

The view marks a block as released first, and the engine's scheduler frees it at the start of its
next scheduling step. Until then the block is unread but still allocated, so there is no step in which
a request reads memory that another request owns.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from collections.abc import Iterable

_lock = threading.Lock()
_pending: dict[str, set[int]] = defaultdict(set)
blocks_released = 0  # blocks returned to the pool in this process


def queue_release(request_id: str, block_indices: Iterable[int]) -> None:
    """Ask for blocks ``block_indices`` of the request the engine calls ``request_id`` to be freed."""
    with _lock:
        _pending[request_id].update(int(i) for i in block_indices)


def apply_pending(scheduler) -> int:  # a vLLM Scheduler
    """Free the queued blocks. Returns how many were freed. Call it from the scheduler's thread."""
    global blocks_released
    with _lock:
        pending = dict(_pending)
        _pending.clear()
    if not pending:
        return 0
    managers = scheduler.kv_cache_manager.coordinator.single_type_managers
    if len(managers) != 1:
        raise RuntimeError("releasing blocks supports one KV-cache group; this model has several")
    manager = managers[0]
    if manager.block_pool.enable_caching:
        raise RuntimeError("releasing blocks needs prefix caching off, because a cached block can be shared")
    freed = []
    for request_id, indices in pending.items():
        blocks = manager.req_to_blocks.get(request_id)
        if blocks is None:  # the request has finished, and its blocks are already free
            continue
        for index in sorted(indices):
            if index >= len(blocks) or blocks[index].is_null:
                continue
            if blocks[index].ref_cnt != 1:
                raise RuntimeError(f"block {index} of request {request_id} is shared and cannot be released")
            freed.append(blocks[index])
            blocks[index] = manager._null_block
    manager.block_pool.free_blocks(freed)
    blocks_released += len(freed)
    return len(freed)


def install_scheduler_release() -> int:
    """Free queued blocks at the start of every scheduling step. Returns the schedulers patched."""
    from vllm.v1.core.sched.scheduler import Scheduler

    original = Scheduler.schedule
    if getattr(original, "_ekv_release", False):
        return 0

    def schedule(self, *args, **kwargs):
        apply_pending(self)
        return original(self, *args, **kwargs)

    schedule._ekv_release = True  # ty: ignore[unresolved-attribute]  # marks the patched method
    Scheduler.schedule = schedule
    return 1
