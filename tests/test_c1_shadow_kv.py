from __future__ import annotations

import pytest
import torch

from basisserve.core.c1_shadow_kv import (
    C1ShadowKeyValueCache,
    ShadowKeyConfig,
    dequantize,
    quantize,
)


def _states(
    *,
    tokens: int,
    key_offset: float = 0.0,
    value_offset: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    keys = torch.arange(2 * tokens * 8, dtype=torch.float32).reshape(1, 2, tokens, 8)
    values = torch.arange(2 * tokens * 3, dtype=torch.float32).reshape(1, 2, tokens, 3)
    return keys / 13 + key_offset, values / 7 + value_offset


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"bits": 3, "group_size": 4, "recent_exact_window": 0}, "bits"),
        ({"bits": 4, "group_size": 0, "recent_exact_window": 0}, "group size"),
        ({"bits": 4, "group_size": 4, "recent_exact_window": -1}, "window"),
    ),
)
def test_shadow_key_config_validation(kwargs: dict[str, int], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ShadowKeyConfig(**kwargs)


def test_identity_quantization_aliases_exact_keys() -> None:
    keys, _ = _states(tokens=5)
    packed = quantize(
        keys,
        ShadowKeyConfig(bits=16, group_size=4, recent_exact_window=0),
    )

    assert packed.values.data_ptr() == keys.data_ptr()
    assert packed.scales is None
    assert packed.logical_storage_bytes() == keys.numel() * keys.element_size()
    assert packed.physical_storage_bytes() == packed.logical_storage_bytes()
    torch.testing.assert_close(
        dequantize(packed, dtype=keys.dtype, device=keys.device),
        keys,
        rtol=0.0,
        atol=0.0,
    )


def test_four_bit_reference_is_finite_and_distinguishes_byte_accounting() -> None:
    keys, _ = _states(tokens=5)
    packed = quantize(
        keys,
        ShadowKeyConfig(bits=4, group_size=4, recent_exact_window=0),
    )
    observed = dequantize(packed, dtype=torch.float32, device="cpu")

    assert packed.values.dtype == torch.int8
    assert packed.scales is not None
    assert torch.isfinite(observed).all()
    scale_bytes = packed.scales.numel() * packed.scales.element_size()
    assert packed.logical_storage_bytes() == (keys.numel() + 1) // 2 + scale_bytes
    assert packed.physical_storage_bytes() == keys.numel() + scale_bytes
    assert packed.physical_storage_bytes() > packed.logical_storage_bytes()


def test_quantization_rejects_incompatible_group_size() -> None:
    with pytest.raises(ValueError, match="divisible"):
        quantize(
            torch.randn(1, 2, 3, 10),
            ShadowKeyConfig(bits=4, group_size=4, recent_exact_window=0),
        )


def test_recent_exact_window_is_bitwise_exact_in_draft_attention() -> None:
    config = ShadowKeyConfig(bits=4, group_size=4, recent_exact_window=2)
    cache = C1ShadowKeyValueCache(num_layers=1, config=config)
    committed_keys, committed_values = _states(tokens=5)
    cache.begin_target_prefill()
    with cache.forward_pass(query_length=5):
        cache.update(committed_keys, committed_values, 0)
    cache.finish_target_prefill()

    draft_keys, draft_values = _states(
        tokens=1,
        key_offset=100,
        value_offset=200,
    )
    cache.begin_draft()
    with cache.forward_pass(query_length=1):
        observed_keys, observed_values = cache.update(draft_keys, draft_values, 0)

    torch.testing.assert_close(
        observed_keys[..., -3:-1, :],
        committed_keys[..., -2:, :],
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(observed_keys[..., -1:, :], draft_keys)
    torch.testing.assert_close(
        observed_values,
        torch.cat((committed_values, draft_values), dim=-2),
    )
    cache.rollback_draft()
    assert cache.runtime_mode == "idle"
    assert cache.shadow_layers[0].draft_key_provisional is None
    assert cache.shadow_layers[0].draft_c1_value_provisional is None


def test_fully_accepted_transaction_commits_target_not_draft() -> None:
    cache = C1ShadowKeyValueCache(
        num_layers=2,
        config=ShadowKeyConfig(bits=8, group_size=4, recent_exact_window=0),
    )
    committed_keys, committed_values = _states(tokens=2)
    cache.begin_target_prefill()
    with cache.forward_pass(query_length=2):
        for layer in range(2):
            cache.update(committed_keys + layer, committed_values + layer, layer)
    cache.finish_target_prefill()

    draft_keys, draft_values = _states(tokens=3, key_offset=100, value_offset=100)
    cache.begin_draft()
    with cache.forward_pass(query_length=3):
        for layer in range(2):
            cache.update(draft_keys + layer, draft_values + layer, layer)
    target_keys, target_values = _states(tokens=3, key_offset=200, value_offset=200)
    cache.begin_verify()
    with cache.forward_pass(query_length=3):
        for layer in range(2):
            cache.update(target_keys + layer, target_values + layer, layer)
    cache.commit_pending_target_prefix(3)

    assert cache.committed_length == 5
    for layer_index, layer in enumerate(cache.shadow_layers):
        torch.testing.assert_close(
            layer.exact_key_committed[..., -3:, :],
            target_keys + layer_index,
        )
        torch.testing.assert_close(
            layer.c1_value_committed[..., -3:, :],
            target_values + layer_index,
        )
        assert layer.draft_key_provisional is None
        assert layer.draft_c1_value_provisional is None
        assert layer.target_key_pending is None
        assert layer.target_c1_value_pending is None


def test_partial_commit_and_exact_correction_keep_layers_synchronized() -> None:
    cache = C1ShadowKeyValueCache(
        num_layers=2,
        config=ShadowKeyConfig(bits=4, group_size=4, recent_exact_window=1),
    )
    initial_keys, initial_values = _states(tokens=2)
    cache.begin_target_prefill()
    with cache.forward_pass(query_length=2):
        for layer in range(2):
            cache.update(initial_keys + layer, initial_values + layer, layer)
    cache.finish_target_prefill()

    draft_keys, draft_values = _states(tokens=3, key_offset=10, value_offset=10)
    cache.begin_draft()
    with cache.forward_pass(query_length=3):
        for layer in range(2):
            cache.update(draft_keys + layer, draft_values + layer, layer)
    target_keys, target_values = _states(tokens=3, key_offset=20, value_offset=20)
    cache.begin_verify()
    with cache.forward_pass(query_length=3):
        for layer in range(2):
            cache.update(target_keys + layer, target_values + layer, layer)
    cache.commit_pending_target_prefix(1)
    assert cache.committed_length == 3

    correction_keys, correction_values = _states(
        tokens=1,
        key_offset=30,
        value_offset=30,
    )
    cache.begin_verify()
    with cache.forward_pass(query_length=1):
        for layer in range(2):
            cache.update(
                correction_keys + layer,
                correction_values + layer,
                layer,
            )
    cache.commit_pending_target_prefix(1)

    assert cache.committed_length == 4
    for layer_index, layer in enumerate(cache.shadow_layers):
        torch.testing.assert_close(
            layer.exact_key_committed[..., -2:-1, :],
            target_keys[..., :1, :] + layer_index,
        )
        torch.testing.assert_close(
            layer.exact_key_committed[..., -1:, :],
            correction_keys + layer_index,
        )
        assert layer.c1_value_committed.shape[-1] == 3

    cache.truncate_committed(2)
    assert cache.committed_length == 2
    assert all(layer.shadow_key_committed is not None for layer in cache.shadow_layers)
    assert all(
        layer.shadow_key_committed.original_shape[-2] == 2
        for layer in cache.shadow_layers
        if layer.shadow_key_committed is not None
    )
    assert all(layer.c1_value_committed.shape[-2] == 2 for layer in cache.shadow_layers)


def test_zero_accepted_prefix_commits_only_explicit_correction() -> None:
    cache = C1ShadowKeyValueCache(
        num_layers=2,
        config=ShadowKeyConfig(bits=4, group_size=4, recent_exact_window=0),
    )
    initial_keys, initial_values = _states(tokens=2)
    cache.begin_target_prefill()
    with cache.forward_pass(query_length=2):
        for layer in range(2):
            cache.update(initial_keys + layer, initial_values + layer, layer)
    cache.finish_target_prefill()

    draft_keys, draft_values = _states(tokens=2, key_offset=10, value_offset=10)
    cache.begin_draft()
    with cache.forward_pass(query_length=2):
        for layer in range(2):
            cache.update(draft_keys + layer, draft_values + layer, layer)
    rejected_keys, rejected_values = _states(
        tokens=2,
        key_offset=20,
        value_offset=20,
    )
    cache.begin_verify()
    with cache.forward_pass(query_length=2):
        for layer in range(2):
            cache.update(rejected_keys + layer, rejected_values + layer, layer)
    cache.commit_pending_target_prefix(0)

    assert cache.committed_length == 2
    for layer_index, layer in enumerate(cache.shadow_layers):
        torch.testing.assert_close(
            layer.exact_key_committed, initial_keys + layer_index
        )
        torch.testing.assert_close(
            layer.c1_value_committed,
            initial_values + layer_index,
        )
        assert layer.draft_key_provisional is None
        assert layer.draft_c1_value_provisional is None
        assert layer.target_key_pending is None
        assert layer.target_c1_value_pending is None

    correction_keys, correction_values = _states(
        tokens=1,
        key_offset=30,
        value_offset=30,
    )
    cache.begin_verify()
    with cache.forward_pass(query_length=1):
        for layer in range(2):
            cache.update(correction_keys + layer, correction_values + layer, layer)
    cache.commit_pending_target_prefix(1)

    assert cache.committed_length == 3
    for layer_index, layer in enumerate(cache.shadow_layers):
        torch.testing.assert_close(
            layer.exact_key_committed[..., -1:, :],
            correction_keys + layer_index,
        )
        torch.testing.assert_close(
            layer.c1_value_committed[..., -1:, :],
            correction_values + layer_index,
        )


def test_failed_forward_rolls_back_every_partially_updated_layer() -> None:
    cache = C1ShadowKeyValueCache(
        num_layers=2,
        config=ShadowKeyConfig(bits=16, group_size=4, recent_exact_window=0),
    )
    keys, values = _states(tokens=2)
    cache.begin_target_prefill()
    with pytest.raises(RuntimeError, match="every layer"):
        with cache.forward_pass(query_length=2):
            cache.update(keys, values, 0)

    assert all(layer.committed_length == 0 for layer in cache.shadow_layers)
    cache.abort_transaction()
    assert cache.runtime_mode == "idle"


def test_cache_has_no_persistent_approximate_value_state() -> None:
    cache = C1ShadowKeyValueCache(
        num_layers=1,
        config=ShadowKeyConfig(bits=4, group_size=4, recent_exact_window=0),
    )
    layer = cache.shadow_layers[0]
    value_state_names = {
        name for name in vars(layer) if "value" in name and not name.startswith("_")
    }
    assert value_state_names == {
        "values",
        "c1_value_committed",
        "draft_c1_value_provisional",
        "target_c1_value_pending",
    }
