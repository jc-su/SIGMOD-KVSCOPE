"""Returning blocks to the engine's pool: only the named blocks, only once, and refusing the unsafe cases.

The scheduler is a stand-in with the structure the module touches: a coordinator with one manager
that holds each request's blocks and a pool that takes freed blocks back.
"""

from types import SimpleNamespace

import pytest

from runtime import release


class Block:
    def __init__(self, block_id: int, null: bool = False, refs: int = 1) -> None:
        self.block_id, self.is_null, self.ref_cnt = block_id, null, refs


class Pool:
    def __init__(self, caching: bool = False) -> None:
        self.enable_caching = caching
        self.freed: list[Block] = []

    def free_blocks(self, blocks) -> None:
        self.freed += list(blocks)


def scheduler(requests: dict[str, list[Block]], caching: bool = False, groups: int = 1):
    pool = Pool(caching)
    managers = tuple(
        SimpleNamespace(req_to_blocks=dict(requests), block_pool=pool, _null_block=Block(0, null=True))
        for _ in range(groups)
    )
    coordinator = SimpleNamespace(single_type_managers=managers)
    return SimpleNamespace(kv_cache_manager=SimpleNamespace(coordinator=coordinator)), managers[0], pool


@pytest.fixture(autouse=True)
def reset():
    release._pending.clear()
    release.blocks_released = 0


def test_only_the_named_blocks_are_freed_and_replaced_by_the_null_block():
    blocks = [Block(10 + i) for i in range(5)]
    sched, manager, pool = scheduler({"a": blocks})
    release.queue_release("a", [1, 3])
    assert release.apply_pending(sched) == 2
    assert [b.block_id for b in pool.freed] == [11, 13]
    assert [b.is_null for b in manager.req_to_blocks["a"]] == [False, True, False, True, False]
    assert release.blocks_released == 2


def test_a_block_is_freed_once():
    sched, _, pool = scheduler({"a": [Block(1), Block(2), Block(3)]})
    release.queue_release("a", [1])
    release.apply_pending(sched)
    release.queue_release("a", [1])
    assert release.apply_pending(sched) == 0
    assert len(pool.freed) == 1


def test_a_finished_request_is_ignored():
    sched, _, pool = scheduler({"a": [Block(1), Block(2)]})
    release.queue_release("gone", [1])
    assert release.apply_pending(sched) == 0
    assert pool.freed == []


def test_nothing_queued_does_nothing():
    sched, _, _ = scheduler({})
    assert release.apply_pending(sched) == 0


def test_a_shared_block_is_refused():
    sched, _, _ = scheduler({"a": [Block(1), Block(2, refs=2)]})
    release.queue_release("a", [1])
    with pytest.raises(RuntimeError, match="shared"):
        release.apply_pending(sched)


def test_prefix_caching_is_refused():
    sched, _, _ = scheduler({"a": [Block(1), Block(2)]}, caching=True)
    release.queue_release("a", [1])
    with pytest.raises(RuntimeError, match="prefix caching"):
        release.apply_pending(sched)


def test_release_against_vllms_own_block_manager():
    """vLLM's real manager and pool: a freed block is reusable, and finishing returns every block once."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("vllm")
    from vllm import SamplingParams
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
    from vllm.v1.request import Request

    block = 16
    spec = FullAttentionSpec(block_size=block, num_kv_heads=8, head_size=128, dtype=torch.bfloat16)
    config = KVCacheConfig(
        num_blocks=200,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(layer_names=["layer0"], kv_cache_spec=spec)],
    )
    manager = KVCacheManager(config, max_model_len=4096, hash_block_size=block, enable_caching=False)
    pool = manager.block_pool

    def request(name: str, blocks: int) -> Request:
        return Request(
            request_id=name,
            prompt_token_ids=list(range(blocks * block)),
            sampling_params=SamplingParams(max_tokens=16),
            pooling_params=None,
        )

    a, b = request("a", 20), request("b", 10)
    for r in (a, b):
        assert manager.allocate_slots(r, r.num_tokens) is not None
    assert pool.get_num_free_blocks() == 199 - 30

    release.queue_release("a", [1, 2, 3, 4, 5, 6])
    assert release.apply_pending(SimpleNamespace(kv_cache_manager=manager)) == 6
    assert pool.get_num_free_blocks() == 199 - 30 + 6

    c = request("c", 6)  # a new request takes the memory the released blocks held
    assert manager.allocate_slots(c, c.num_tokens) is not None
    assert pool.get_num_free_blocks() == 199 - 30

    for r in (a, b, c):  # finishing frees what each still holds, and skips the released blocks
        manager.free(r)
    assert pool.get_num_free_blocks() == 199


def test_several_cache_groups_are_refused():
    sched, _, _ = scheduler({"a": [Block(1), Block(2)]}, groups=2)
    release.queue_release("a", [1])
    with pytest.raises(RuntimeError, match="one KV-cache group"):
        release.apply_pending(sched)
