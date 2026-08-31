from __future__ import annotations

import math

import torch

from basisserve.core.v_qk_score_probe import (
    apply_qk_score_probe,
    fit_centered_score_probe,
)


def test_centered_score_cg_recovers_synthetic_map() -> None:
    generator = torch.Generator().manual_seed(97)
    kv_heads, group, rows, head_dim, rank, tokens = 2, 2, 24, 4, 3, 11
    query = torch.randn(
        kv_heads, group, rows, head_dim, generator=generator, dtype=torch.float64
    )
    true_weight = torch.randn(
        kv_heads, group, head_dim, rank, generator=generator, dtype=torch.float64
    )
    grams = []
    crosses = []
    for kv_head in range(kv_heads):
        head_grams = []
        head_crosses = [[] for _ in range(group)]
        for row in range(rows):
            source = torch.randn(tokens, rank, generator=generator, dtype=torch.float64)
            source = source - source.mean(dim=0, keepdim=True)
            gram = source.mT @ source
            head_grams.append(gram)
            for local_head in range(group):
                projected = (
                    query[kv_head, local_head, row]
                    @ true_weight[kv_head, local_head]
                    / math.sqrt(head_dim)
                )
                target = source @ projected
                head_crosses[local_head].append(source.mT @ target)
        grams.append(torch.stack(head_grams))
        crosses.append(torch.stack([torch.stack(value) for value in head_crosses]))
    gram = torch.stack(grams)
    cross = torch.stack(crosses)

    fit = fit_centered_score_probe(
        query,
        gram,
        cross,
        relative_damping=0.0,
        cg_iterations=128,
        cg_relative_tolerance=1.0e-10,
    )

    torch.testing.assert_close(
        fit.weight.reshape_as(true_weight),
        true_weight,
        rtol=1.0e-7,
        atol=1.0e-7,
    )
    assert fit.maximum_relative_residual < 1.0e-9


def test_apply_qk_score_probe_matches_explicit_proxy_key() -> None:
    generator = torch.Generator().manual_seed(101)
    query = torch.randn(4, 5, generator=generator)
    source = torch.randn(2, 13, 3, generator=generator)
    weight = torch.randn(4, 5, 3, generator=generator)

    observed = apply_qk_score_probe(
        query, source, weight, heads_per_group=2
    )
    kv_index = torch.tensor([0, 0, 1, 1])
    proxy_key = torch.einsum(
        "htr,hdr->htd", source.index_select(0, kv_index), weight
    )
    expected = torch.einsum("hd,htd->ht", query, proxy_key) / math.sqrt(5)

    torch.testing.assert_close(observed, expected)
