from __future__ import annotations

import torch

from evaluation.build_qwen3_8b_pairwise_k_identity_control import (
    identity_pair_factors,
)


def test_identity_pair_factors_reproduce_both_layer_scores() -> None:
    generator = torch.Generator().manual_seed(20260828)
    pair_key, pair_query = identity_pair_factors(
        num_layers=4,
        num_kv_heads=2,
        num_query_heads=4,
        head_dim=7,
    )
    assert pair_key.shape == (2, 2, 14, 14)
    assert pair_query.shape == (4, 4, 7, 14)
    for layer_slot in (0, 1):
        key0 = torch.randn(11, 7, generator=generator)
        key1 = torch.randn(11, 7, generator=generator)
        query = torch.randn(5, 7, generator=generator)
        pair_code = (
            key0 @ pair_key[0, 0, :7]
            + key1 @ pair_key[0, 0, 7:]
        )
        projected_query = query @ pair_query[layer_slot, 0]
        exact_key = key0 if layer_slot == 0 else key1
        torch.testing.assert_close(
            projected_query @ pair_code.mT,
            query @ exact_key.mT,
        )
