import torch
import pytest
import copy

from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Attention, Qwen3_5DynamicCache
from basisserve.core.qwen35_gated_v_runtime import GatedVAttention, fork_gated_v_cache, reset_gated_v_cache


def setup():
    torch.manual_seed(417)
    config = Qwen3_5TextConfig(hidden_size=18, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, num_hidden_layers=2, layer_types=['linear_attention', 'full_attention'], attention_bias=True)
    config._attn_implementation = 'sdpa'
    native = Qwen3_5Attention(config, 1).double().eval().requires_grad_(False)
    x = torch.randn(2, 9, 18, dtype=torch.float64)
    # Nontrivial partial RoPE, first four channels only.
    angles = torch.randn(2, 9, 2, dtype=x.dtype).repeat(1, 1, 2)
    return config, native, x, (angles.cos(), angles.sin())


def test_identity_bias_partial_rope_and_padding():
    _, native, x, pos = setup()
    eye = torch.eye(8, dtype=x.dtype).repeat(2, 1, 1)
    adapter = GatedVAttention(native, eye, eye, query_chunk=3)
    mask = torch.ones(2, 1, 9, 9, dtype=torch.bool).tril()
    mask[1, :, :, :2] = False
    expected = native(x, pos, mask)[0]
    actual = adapter(x, pos, mask)[0]
    torch.testing.assert_close(actual, expected, atol=1e-9, rtol=1e-7)


def test_compact_cached_chunked_decode_and_reorder():
    config, native, x, pos = setup()
    e = torch.linalg.qr(torch.randn(2, 8, 3, dtype=x.dtype)).Q
    adapter = GatedVAttention(native, e, e.mT, query_chunk=2)
    expected = adapter(x, pos)[0]
    cache = Qwen3_5DynamicCache(config)
    pieces = []
    for start, stop in [(0, 3), (3, 8), (8, 9)]:
        pieces.append(adapter(x[:, start:stop], tuple(p[:, start:stop] for p in pos),
            past_key_values=cache, cache_position=torch.arange(start, stop))[0])
    torch.testing.assert_close(torch.cat(pieces, 1), expected, atol=1e-9, rtol=1e-7)
    assert cache.value_cache[1].shape == (2, 2, 9, 3)
    assert cache.key_cache[1].shape == (2, 2, 9, 8)
    old = cache.value_cache[1].clone()
    cache.reorder_cache(torch.tensor([1, 0]))
    torch.testing.assert_close(cache.value_cache[1], old.flip(0))
    fork = fork_gated_v_cache(cache)
    reset_gated_v_cache(fork)
    assert fork.value_cache[1] is None and cache.value_cache[1] is not None
    with pytest.raises(AssertionError):
        native(x[:, :1], tuple(p[:, :1] for p in pos), None, past_key_values=cache)


def test_grouped_palu_folding():
    _, native, x, pos = setup()
    eye = torch.eye(16, dtype=x.dtype).unsqueeze(0)
    adapter = GatedVAttention(native, eye, eye, query_chunk=3)
    torch.testing.assert_close(adapter(x, pos)[0], native(x, pos, None)[0], atol=1e-9, rtol=1e-7)


def test_grouped_compressed_palu_matches_dense_projected_v():
    _, native, x, pos = setup()
    encoder = torch.randn(1, 16, 5, dtype=x.dtype)
    decoder = torch.randn(1, 5, 16, dtype=x.dtype)
    adapter = GatedVAttention(native, encoder, decoder, query_chunk=3)
    reference = copy.deepcopy(native)
    product = (encoder @ decoder)[0].T
    with torch.no_grad():
        reference.v_proj.weight.copy_(product @ native.v_proj.weight)
        reference.v_proj.bias.copy_(product @ native.v_proj.bias)
    torch.testing.assert_close(adapter(x, pos)[0], reference(x, pos, None)[0], atol=1e-8, rtol=1e-7)
