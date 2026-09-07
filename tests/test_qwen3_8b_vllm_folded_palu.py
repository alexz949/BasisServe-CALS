from __future__ import annotations

import torch

from basisserve.core.qwen3_8b_vllm_folded_palu import (
    reconstructed_palu_value_projection,
)


def test_reconstructed_palu_value_matches_group_factorization() -> None:
    generator = torch.Generator().manual_seed(43)
    num_groups = 2
    group_width = 8
    hidden_size = 7
    ranks = (3, 5)
    writer = torch.randn(sum(ranks), hidden_size, generator=generator)
    decoder = torch.randn(
        num_groups,
        group_width,
        max(ranks),
        generator=generator,
    )
    hidden = torch.randn(5, hidden_size, generator=generator)

    dense_value = reconstructed_palu_value_projection(
        writer,
        decoder,
        ranks,
    )
    writer_offset = 0
    for group, rank in enumerate(ranks):
        latent = hidden @ writer[writer_offset : writer_offset + rank].T
        expected = latent @ decoder[group, :, :rank].T
        writer_offset += rank
        observed = (
            hidden @ dense_value[group * group_width : (group + 1) * group_width].T
        )
        torch.testing.assert_close(observed, expected)
