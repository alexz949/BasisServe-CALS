from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from basisserve.core.c1_tp_decode import (
    C1TPHeadOwnership,
    PackedC1TPLayer,
    headwise_c1_encode,
)
from basisserve.core.c1_variable_v_attention import (
    C1RankLocalAttentionReference,
    C1StaticKVCache,
    fold_local_c1_value_projection,
    reference_grouped_query_attention,
)
from basisserve.kernels.ragged_allgather import StaticRaggedPlan


def _packed_factors(
    encoder: torch.Tensor,
    *,
    query_heads: int,
    hidden_size: int,
) -> PackedC1TPLayer:
    source_rank = int(encoder.shape[1])
    plan = StaticRaggedPlan.from_head_ranks(
        [source_rank],
        heads_per_source=query_heads,
    )
    decoder = torch.zeros(
        query_heads * source_rank,
        hidden_size,
        dtype=encoder.dtype,
        device=encoder.device,
    )
    return PackedC1TPLayer(
        layer_index=0,
        process_rank=0,
        ownership=C1TPHeadOwnership(
            process_rank=0,
            kv_head=0,
            query_head_start=0,
            query_head_stop=query_heads,
        ),
        source_ranks=(source_rank,),
        plan=plan,
        local_encoder=encoder,
        local_decoder=decoder,
        global_decoder=decoder,
        manifest_sha256="0" * 64,
        artifact_sha256="1" * 64,
    )


@pytest.mark.parametrize("with_bias", [False, True])
def test_folded_value_projection_matches_dense_value_then_encoder(
    with_bias: bool,
) -> None:
    generator = torch.Generator().manual_seed(20260826)
    batch, tokens, hidden_size, head_dim, source_rank = 2, 5, 7, 4, 3
    hidden_states = torch.randn(
        batch,
        tokens,
        hidden_size,
        dtype=torch.float64,
        generator=generator,
    )
    dense_weight = torch.randn(
        head_dim,
        hidden_size,
        dtype=torch.float64,
        generator=generator,
    )
    dense_bias = (
        torch.randn(head_dim, dtype=torch.float64, generator=generator)
        if with_bias
        else None
    )
    encoder = torch.randn(
        head_dim,
        source_rank,
        dtype=torch.float64,
        generator=generator,
    )

    compact_weight, compact_bias = fold_local_c1_value_projection(
        dense_weight,
        encoder,
        dense_bias,
    )
    dense_values = F.linear(hidden_states, dense_weight, dense_bias)
    expected = dense_values @ encoder
    observed = F.linear(hidden_states, compact_weight, compact_bias)

    assert compact_weight.shape == (source_rank, hidden_size)
    assert compact_bias is None if dense_bias is None else compact_bias.shape == (source_rank,)
    torch.testing.assert_close(observed, expected, rtol=1e-12, atol=1e-12)


def test_static_cache_tracks_dense_keys_and_variable_width_values() -> None:
    cache = C1StaticKVCache(
        batch_size=2,
        capacity=4,
        head_dim=5,
        value_head_dim=3,
        dtype=torch.float32,
        device="cpu",
    )
    key_pointer = cache.key_storage.data_ptr()
    value_pointer = cache.value_storage.data_ptr()
    assert cache.key_bytes == 2 * 4 * 5 * 4
    assert cache.value_bytes == 2 * 4 * 3 * 4
    assert cache.value_bytes / cache.key_bytes == 3 / 5

    generator = torch.Generator().manual_seed(20260827)
    first_keys = torch.randn(2, 2, 5, generator=generator)
    first_values = torch.randn(2, 2, 3, generator=generator)
    second_keys = torch.randn(2, 1, 5, generator=generator)
    second_values = torch.randn(2, 1, 3, generator=generator)
    keys, values = cache.append(first_keys, first_values)
    assert cache.sequence_length == 2
    torch.testing.assert_close(keys, first_keys)
    torch.testing.assert_close(values, first_values)

    keys, values = cache.append(second_keys, second_values)
    assert cache.sequence_length == 3
    assert cache.key_storage.data_ptr() == key_pointer
    assert cache.value_storage.data_ptr() == value_pointer
    torch.testing.assert_close(keys, torch.cat((first_keys, second_keys), dim=1))
    torch.testing.assert_close(values, torch.cat((first_values, second_values), dim=1))

    with pytest.raises(RuntimeError, match="capacity exceeded"):
        cache.append(torch.randn(2, 2, 5), torch.randn(2, 2, 3))
    assert cache.sequence_length == 3
    cache.truncate(2)
    assert cache.sequence_length == 2
    torch.testing.assert_close(cache.current()[0], first_keys)
    with pytest.raises(ValueError, match="valid prefix"):
        cache.truncate(3)
    cache.reset()
    assert cache.sequence_length == 0
    assert tuple(cache.current()[0].shape) == (2, 0, 5)


def test_compact_prefill_matches_dense_attention_then_headwise_encoder() -> None:
    generator = torch.Generator().manual_seed(20260828)
    batch, heads, tokens, head_dim, source_rank = 2, 3, 5, 4, 2
    query = torch.randn(
        batch,
        heads,
        tokens,
        head_dim,
        dtype=torch.float64,
        generator=generator,
    )
    key = torch.randn(
        batch,
        tokens,
        head_dim,
        dtype=torch.float64,
        generator=generator,
    )
    dense_value = torch.randn(
        batch,
        tokens,
        head_dim,
        dtype=torch.float64,
        generator=generator,
    )
    encoder = torch.randn(
        head_dim,
        source_rank,
        dtype=torch.float64,
        generator=generator,
    )

    dense_output, dense_weights = reference_grouped_query_attention(
        query,
        key,
        dense_value,
        is_causal=True,
    )
    compact_output, compact_weights = reference_grouped_query_attention(
        query,
        key,
        dense_value @ encoder,
        is_causal=True,
    )
    expected = headwise_c1_encode(dense_output, encoder)

    torch.testing.assert_close(compact_weights, dense_weights, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        compact_output.reshape(batch, tokens, heads * source_rank),
        expected,
        rtol=1e-12,
        atol=1e-12,
    )
    future = torch.triu(torch.ones(tokens, tokens, dtype=torch.bool), diagonal=1)
    assert torch.count_nonzero(dense_weights[..., future]) == 0


def test_rank_local_reference_streams_prefill_and_decode() -> None:
    generator = torch.Generator().manual_seed(20260829)
    batch = 2
    sequence = 5
    prefill = 3
    query_heads = 3
    hidden_size = 7
    head_dim = 4
    source_rank = 2
    dtype = torch.float64

    dense_projection = nn.Linear(
        hidden_size,
        head_dim,
        bias=True,
        dtype=dtype,
    )
    with torch.no_grad():
        dense_projection.weight.copy_(
            torch.randn(
                head_dim,
                hidden_size,
                dtype=dtype,
                generator=generator,
            )
        )
        dense_projection.bias.copy_(
            torch.randn(head_dim, dtype=dtype, generator=generator)
        )
    encoder = torch.randn(
        head_dim,
        source_rank,
        dtype=dtype,
        generator=generator,
    )
    factors = _packed_factors(
        encoder,
        query_heads=query_heads,
        hidden_size=hidden_size,
    )
    attention = C1RankLocalAttentionReference.from_dense_projection(
        dense_projection,
        factors,
    )
    cache = C1StaticKVCache(
        batch_size=batch,
        capacity=sequence,
        head_dim=head_dim,
        value_head_dim=source_rank,
        dtype=dtype,
        device="cpu",
    )
    hidden = torch.randn(
        batch,
        sequence,
        hidden_size,
        dtype=dtype,
        generator=generator,
    )
    query = torch.randn(
        batch,
        query_heads,
        sequence,
        head_dim,
        dtype=dtype,
        generator=generator,
    )
    key = torch.randn(
        batch,
        sequence,
        head_dim,
        dtype=dtype,
        generator=generator,
    )

    dense_keys: list[torch.Tensor] = []
    dense_values: list[torch.Tensor] = []
    compact_values: list[torch.Tensor] = []
    chunk_boundaries = ((0, prefill), (prefill, prefill + 1), (prefill + 1, sequence))
    for start, stop in chunk_boundaries:
        hidden_chunk = hidden[:, start:stop]
        query_chunk = query[:, :, start:stop]
        key_chunk = key[:, start:stop]
        dense_value_chunk = dense_projection(hidden_chunk)
        dense_keys.append(key_chunk)
        dense_values.append(dense_value_chunk)
        compact_values.append(dense_value_chunk @ encoder)

        observed = attention(
            hidden_chunk,
            query_chunk,
            key_chunk,
            cache,
            is_causal=True,
        )
        dense_output, dense_weights = reference_grouped_query_attention(
            query_chunk,
            torch.cat(dense_keys, dim=1),
            torch.cat(dense_values, dim=1),
            is_causal=True,
        )
        expected = headwise_c1_encode(dense_output, encoder)

        assert observed.head_coordinates.shape == (
            batch,
            stop - start,
            query_heads,
            source_rank,
        )
        torch.testing.assert_close(
            observed.projected_values,
            compact_values[-1],
            rtol=1e-12,
            atol=1e-12,
        )
        torch.testing.assert_close(
            observed.attention_weights,
            dense_weights,
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            observed.local_coordinates,
            expected,
            rtol=1e-12,
            atol=1e-12,
        )

    cached_keys, cached_values = cache.current()
    torch.testing.assert_close(cached_keys, key)
    torch.testing.assert_close(
        cached_values,
        torch.cat(compact_values, dim=1),
        rtol=1e-12,
        atol=1e-12,
    )
    assert cache.sequence_length == sequence


def test_virtual_tp_compact_attention_matches_ragged_global_decode() -> None:
    """Exercise the exact compact-attention -> rank-order wire -> decoder path."""

    generator = torch.Generator().manual_seed(20260830)
    batch, tokens = 2, 4
    query_heads = 2
    head_dim = 4
    input_hidden = 7
    output_hidden = 5
    source_ranks = (2, 3, 1)
    dtype = torch.float64
    plan = StaticRaggedPlan.from_head_ranks(
        source_ranks,
        heads_per_source=query_heads,
    )
    global_decoder = torch.randn(
        plan.total_width,
        output_hidden,
        dtype=dtype,
        generator=generator,
    )
    local_coordinates: list[torch.Tensor] = []
    dense_contributions: list[torch.Tensor] = []

    for process_rank, source_rank in enumerate(source_ranks):
        encoder = torch.randn(
            head_dim,
            source_rank,
            dtype=dtype,
            generator=generator,
        )
        wire_start = plan.offsets[process_rank]
        wire_stop = wire_start + plan.source_widths[process_rank]
        factors = PackedC1TPLayer(
            layer_index=0,
            process_rank=process_rank,
            ownership=C1TPHeadOwnership(
                process_rank=process_rank,
                kv_head=process_rank,
                query_head_start=process_rank * query_heads,
                query_head_stop=(process_rank + 1) * query_heads,
            ),
            source_ranks=source_ranks,
            plan=plan,
            local_encoder=encoder,
            local_decoder=global_decoder[wire_start:wire_stop],
            global_decoder=global_decoder,
            manifest_sha256="0" * 64,
            artifact_sha256="1" * 64,
        )
        dense_projection = nn.Linear(
            input_hidden,
            head_dim,
            bias=False,
            dtype=dtype,
        )
        with torch.no_grad():
            dense_projection.weight.copy_(
                torch.randn(
                    head_dim,
                    input_hidden,
                    dtype=dtype,
                    generator=generator,
                )
            )
        attention = C1RankLocalAttentionReference.from_dense_projection(
            dense_projection,
            factors,
        )
        cache = C1StaticKVCache(
            batch_size=batch,
            capacity=tokens,
            head_dim=head_dim,
            value_head_dim=source_rank,
            dtype=dtype,
            device="cpu",
        )
        hidden = torch.randn(
            batch,
            tokens,
            input_hidden,
            dtype=dtype,
            generator=generator,
        )
        query = torch.randn(
            batch,
            query_heads,
            tokens,
            head_dim,
            dtype=dtype,
            generator=generator,
        )
        key = torch.randn(
            batch,
            tokens,
            head_dim,
            dtype=dtype,
            generator=generator,
        )

        compact = attention(hidden, query, key, cache, is_causal=True)
        dense_attention, _ = reference_grouped_query_attention(
            query,
            key,
            dense_projection(hidden),
            is_causal=True,
        )
        dense_coordinates = headwise_c1_encode(dense_attention, encoder)
        assert compact.local_coordinates.shape[-1] == plan.source_widths[process_rank]
        torch.testing.assert_close(
            compact.local_coordinates,
            dense_coordinates,
            rtol=1e-12,
            atol=1e-12,
        )
        local_coordinates.append(compact.local_coordinates)
        dense_contributions.append(dense_coordinates @ factors.local_decoder)

    gathered_coordinates = torch.cat(local_coordinates, dim=-1)
    observed = gathered_coordinates @ global_decoder
    expected = torch.stack(dense_contributions).sum(dim=0)
    assert gathered_coordinates.shape == (batch, tokens, plan.total_width)
    torch.testing.assert_close(observed, expected, rtol=1e-12, atol=1e-12)
