from __future__ import annotations

import torch

from basisserve.core.tp_source_wo_c1 import (
    TPSourceWOLayout,
    covariance_to_source_blocks,
    fold_factors_to_dense_weight,
    identity_factors,
)


def test_deepseek_v2_lite_tp8_rank192_communication_accounting() -> None:
    layout = TPSourceWOLayout(
        input_width=2048,
        output_width=2048,
        tp_size=8,
        source_rank=192,
        dtype_bytes=2,
    )

    assert layout.source_width == 256
    assert layout.gathered_width == 1536
    assert layout.dense_allgather_ring_bytes_per_rank == 3584
    assert layout.compressed_allgather_ring_bytes_per_rank == 2688
    assert layout.dense_allreduce_ring_bytes_per_rank == 7168
    assert layout.reduction_vs_dense_allgather == 0.25
    assert layout.reduction_vs_dense_allreduce == 0.625
    assert layout.accounting()["kv_cache_compression"] == "none"


def test_folded_weight_matches_explicit_private_allgather_path() -> None:
    generator = torch.Generator().manual_seed(7)
    layout = TPSourceWOLayout(
        input_width=12,
        output_width=9,
        tp_size=3,
        source_rank=2,
    )
    encoders = torch.randn(3, 4, 2, generator=generator, dtype=torch.float64)
    decoders = torch.randn(3, 2, 9, generator=generator, dtype=torch.float64)
    activation = torch.randn(5, 12, generator=generator, dtype=torch.float64)

    source_activations = activation.reshape(5, 3, 4)
    gathered = torch.einsum("bsd,sdr->bsr", source_activations, encoders)
    explicit = torch.einsum("bsr,sro->bo", gathered, decoders)
    folded = fold_factors_to_dense_weight(encoders, decoders, layout)

    torch.testing.assert_close(activation @ folded.transpose(0, 1), explicit)


def test_full_source_rank_identity_is_exact() -> None:
    generator = torch.Generator().manual_seed(11)
    layout = TPSourceWOLayout(
        input_width=12,
        output_width=7,
        tp_size=3,
        source_rank=4,
    )
    weight = torch.randn(7, 12, generator=generator, dtype=torch.float64)

    encoders, decoders = identity_factors(weight, layout)
    folded = fold_factors_to_dense_weight(encoders, decoders, layout)

    torch.testing.assert_close(folded, weight, rtol=0.0, atol=0.0)


def test_covariance_block_view_preserves_quadratic_form() -> None:
    generator = torch.Generator().manual_seed(19)
    layout = TPSourceWOLayout(
        input_width=12,
        output_width=7,
        tp_size=3,
        source_rank=2,
    )
    samples = torch.randn(31, 12, generator=generator, dtype=torch.float64)
    covariance = samples.transpose(0, 1) @ samples
    blocks = covariance_to_source_blocks(covariance, layout)
    reconstructed = blocks.permute(0, 2, 1, 3).reshape(12, 12)

    torch.testing.assert_close(reconstructed, covariance)
