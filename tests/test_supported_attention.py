"""The view refuses a model whose attention it cannot trim exactly, and accepts plain full attention.

The builder, the layers, and their cache specs are stand-ins with the attributes the check reads. The cache specs
are vLLM's own classes.
"""

from types import SimpleNamespace

import pytest
import torch
from vllm.v1.attention.backend import AttentionType
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec, SlidingWindowSpec

from runtime import block_view
from runtime.block_view import require_supported_attention

FULL = FullAttentionSpec(block_size=16, num_kv_heads=8, head_size=128, dtype=torch.float16)


def layer(sliding_window=None, alibi=None, attn_type=AttentionType.DECODER):
    return SimpleNamespace(
        sliding_window=sliding_window, impl=SimpleNamespace(alibi_slopes=alibi), attn_type=attn_type
    )


def builder(spec=FULL, layers=None, names=("l0", "l1")):
    layers = layers or {name: layer() for name in names}
    context = SimpleNamespace(static_forward_context=layers)
    return SimpleNamespace(
        kv_cache_spec=spec,
        layer_names=list(layers),
        vllm_config=SimpleNamespace(compilation_config=context),
    )


def test_plain_full_attention_is_accepted():
    require_supported_attention(builder())


def test_a_sliding_window_cache_group_is_refused():
    spec = SlidingWindowSpec(
        block_size=16, num_kv_heads=8, head_size=128, dtype=torch.float16, sliding_window=1024
    )
    with pytest.raises(RuntimeError, match="SlidingWindowSpec"):
        require_supported_attention(builder(spec))


def test_a_state_space_cache_group_is_refused():
    spec = MambaSpec(block_size=16, shapes=((1,),), dtypes=(torch.float16,))
    with pytest.raises(RuntimeError, match="MambaSpec"):
        require_supported_attention(builder(spec))


def test_full_attention_with_a_window_or_chunk_in_its_spec_is_refused():
    windowed = FullAttentionSpec(
        block_size=16, num_kv_heads=8, head_size=128, dtype=torch.float16, sliding_window=512
    )
    chunked = FullAttentionSpec(
        block_size=16, num_kv_heads=8, head_size=128, dtype=torch.float16, attention_chunk_size=256
    )
    for spec in (windowed, chunked):
        with pytest.raises(RuntimeError, match="sliding window or chunked"):
            require_supported_attention(builder(spec))


def test_a_layer_with_a_sliding_window_is_refused_and_named():
    with pytest.raises(RuntimeError, match="layer bad has a sliding window"):
        require_supported_attention(builder(layers={"ok": layer(), "bad": layer(sliding_window=4096)}))


def test_a_layer_with_a_position_bias_is_refused():
    with pytest.raises(RuntimeError, match="position bias"):
        require_supported_attention(builder(layers={"bad": layer(alibi=torch.ones(8))}))


def test_attention_that_is_not_decoder_self_attention_is_refused():
    with pytest.raises(RuntimeError, match="encoder"):
        require_supported_attention(builder(layers={"bad": layer(attn_type=AttentionType.ENCODER)}))


def test_a_second_cache_group_is_refused(monkeypatch):
    class Batch:
        block_table_tensor = torch.zeros(2, 4, dtype=torch.int32)

    monkeypatch.setattr(block_view, "_view", None)
    monkeypatch.setattr(block_view, "_plane", None)
    monkeypatch.setattr(block_view, "_layers", ())
    monkeypatch.setenv("EKV_ATTN_SHARE", "")
    config = SimpleNamespace(scheduler_config=SimpleNamespace(max_num_seqs=2))
    first = builder(names=("a", "b"))
    first.vllm_config.scheduler_config = config.scheduler_config
    second = builder(names=("c", "d"))
    second.vllm_config.scheduler_config = config.scheduler_config
    view = block_view._view_for(first, Batch())
    assert block_view._view_for(first, Batch()) is view  # the same group again is fine
    with pytest.raises(RuntimeError, match="several"):
        block_view._view_for(second, Batch())
