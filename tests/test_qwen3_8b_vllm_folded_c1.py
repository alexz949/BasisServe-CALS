from __future__ import annotations

import torch

from basisserve.core.qwen3_8b_vllm_folded_c1 import (
    folded_c1_projection_weights,
)


def test_folded_c1_projection_weights_match_per_source_factorization() -> None:
    generator = torch.Generator().manual_seed(19)
    num_sources = 2
    heads_per_source = 2
    num_query_heads = num_sources * heads_per_source
    head_dim = 4
    hidden_size = 7
    ranks = (2, 3)
    maximum_rank = max(ranks)
    dense_value = torch.randn(
        num_sources * head_dim,
        hidden_size,
        generator=generator,
    )
    encoders = torch.randn(
        num_sources,
        head_dim,
        maximum_rank,
        generator=generator,
    )
    decoders = torch.randn(
        num_query_heads,
        maximum_rank,
        hidden_size,
        generator=generator,
    )

    folded_value, folded_output = folded_c1_projection_weights(
        dense_value,
        encoders,
        decoders,
        ranks,
    )
    assert tuple(folded_value.shape) == tuple(dense_value.shape)
    assert tuple(folded_output.shape) == (
        hidden_size,
        num_query_heads * head_dim,
    )
    for source, rank in enumerate(ranks):
        dense_rows = slice(source * head_dim, (source + 1) * head_dim)
        latent_rows = slice(source * head_dim, source * head_dim + rank)
        assert torch.allclose(
            folded_value[latent_rows],
            encoders[source, :, :rank].T @ dense_value[dense_rows],
        )
        assert torch.count_nonzero(
            folded_value[source * head_dim + rank : (source + 1) * head_dim]
        ) == 0
        for head in range(
            source * heads_per_source,
            (source + 1) * heads_per_source,
        ):
            latent_columns = slice(head * head_dim, head * head_dim + rank)
            assert torch.allclose(
                folded_output[:, latent_columns],
                decoders[head, :rank].T,
            )
            assert torch.count_nonzero(
                folded_output[
                    :,
                    head * head_dim + rank : (head + 1) * head_dim,
                ]
            ) == 0
