from __future__ import annotations

import torch

from basisserve.diagnostics.similarity import centered_linear_cka
from evaluation.analyze_llama2_mha_c1_cka_head_allocation import (
    HEAD_DIM,
    compression_cka_from_covariance,
    minimum_moved,
    optimize_balanced_cka_partition,
    partition_score,
)
from basisserve.analysis.neuron_repartition import contiguous_balanced_partition


def test_covariance_compression_cka_matches_explicit_rows() -> None:
    generator = torch.Generator().manual_seed(19)
    activation = torch.randn(384, HEAD_DIM, generator=generator, dtype=torch.float64)
    encoder = torch.randn(HEAD_DIM, 17, generator=generator, dtype=torch.float64)
    centered = activation - activation.mean(0, keepdim=True)
    covariance = centered.T @ centered / len(centered)

    actual = compression_cka_from_covariance(covariance, encoder)
    expected = centered_linear_cka(activation, activation @ encoder)

    assert abs(actual - expected) < 1.0e-12


def test_balanced_cka_search_finds_noncontiguous_improvement() -> None:
    similarity = torch.full((32, 32), 0.05, dtype=torch.float64)
    similarity.fill_diagonal_(1.0)
    planted = [
        torch.tensor([0, 1, 2, 4]),
        torch.tensor([3, 5, 6, 7]),
        *[torch.arange(start, start + 4) for start in range(8, 32, 4)],
    ]
    for group in planted:
        for first in group:
            for second in group:
                if first != second:
                    similarity[first, second] = 0.9

    contiguous = contiguous_balanced_partition(32, 8)
    selected, _ = optimize_balanced_cka_partition(
        similarity,
        tp_size=8,
        random_partitions=4,
        local_search_rounds=32,
        seed=7,
    )

    assert partition_score(similarity, selected) > partition_score(similarity, contiguous)
    assert {tuple(group.tolist()) for group in selected} == {
        tuple(group.tolist()) for group in planted
    }
    assert minimum_moved(contiguous, selected) == 2
