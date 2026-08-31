from __future__ import annotations

import torch
from transformers import DynamicCache, Qwen3Config

from evaluation.benchmark_qwen3_8b_c1_v96_triton_decode import (
    _static_restore,
    _to_static_cache,
)


def test_dynamic_prompt_cache_converts_to_fixed_capacity() -> None:
    config = Qwen3Config(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
    )
    dynamic = DynamicCache()
    sources = []
    for layer_idx in range(config.num_hidden_layers):
        key = torch.arange(2 * 5 * 4, dtype=torch.float32).reshape(1, 2, 5, 4)
        key = key + 100 * layer_idx
        value = torch.arange(2 * 5 * 3, dtype=torch.float32).reshape(1, 2, 5, 3)
        value = value + 100 * layer_idx
        dynamic.update(key, value, layer_idx)
        sources.append((key, value))

    static = _to_static_cache(dynamic, config=config, capacity=12)

    assert int(static.get_seq_length()) == 5
    for layer, (key, value) in zip(static.layers, sources, strict=True):
        assert tuple(layer.keys.shape) == (1, 2, 12, 4)
        assert tuple(layer.values.shape) == (1, 2, 12, 3)
        torch.testing.assert_close(layer.keys[:, :, :5], key)
        torch.testing.assert_close(layer.values[:, :, :5], value)
        assert torch.count_nonzero(layer.keys[:, :, 5:]) == 0
        assert torch.count_nonzero(layer.values[:, :, 5:]) == 0

    for layer in static.layers:
        layer.cumulative_length.add_(3)
    _static_restore(static, 5)
    assert int(static.get_seq_length()) == 5
