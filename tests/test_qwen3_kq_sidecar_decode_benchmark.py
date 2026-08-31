from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from evaluation.benchmark_qwen3_8b_c1_kq_sidecar_decode import (
    _cache_bytes,
    _pipeline_device_map,
)


def test_pipeline_device_map_splits_qwen3_layers_evenly() -> None:
    mapping = _pipeline_device_map(layers=36, devices=2)

    assert mapping["model.embed_tokens"] == 0
    assert mapping["model.rotary_emb"] == 0
    assert mapping["model.layers.0"] == 0
    assert mapping["model.layers.17"] == 0
    assert mapping["model.layers.18"] == 1
    assert mapping["model.layers.35"] == 1
    assert mapping["model.norm"] == 1
    assert mapping["lm_head"] == 1


@pytest.mark.parametrize("geometry", ((0, 2), (36, 0), (2, 3)))
def test_pipeline_device_map_rejects_invalid_geometry(
    geometry: tuple[int, int],
) -> None:
    with pytest.raises(ValueError, match="pipeline"):
        _pipeline_device_map(*geometry)


def test_cache_bytes_counts_keys_and_values_separately() -> None:
    cache = SimpleNamespace(
        layers=[
            SimpleNamespace(
                keys=torch.empty(2, 3, dtype=torch.bfloat16),
                values=torch.empty(2, 5, dtype=torch.bfloat16),
            ),
            SimpleNamespace(
                keys=torch.empty(7, dtype=torch.float32),
                values=None,
            ),
        ]
    )

    assert _cache_bytes(cache) == {
        "key_bytes": 6 * 2 + 7 * 4,
        "value_bytes": 10 * 2,
        "total_bytes": 6 * 2 + 7 * 4 + 10 * 2,
    }
