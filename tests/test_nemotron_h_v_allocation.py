import itertools

import pytest
import torch

from evaluation.allocate_nemotron_h_v96 import (
    RANKS, allocate, close_candidate, _fold_ragged_to_padded_weights,
)

LAYERS=(7,18,29,40)


def test_closed_candidate_preserves_subspaces_and_improves_deployed_reconstruction():
    torch.manual_seed(37)
    groups, heads, dim, hidden, rank = 2, 4, 4, 6, 2
    A = torch.randn(groups, dim, rank).bfloat16()
    D = torch.zeros(heads, rank, hidden).bfloat16()
    basis = torch.randn(heads * dim, heads * dim)
    covariance = basis @ basis.T + torch.eye(heads * dim)
    weight = torch.randn(hidden, heads * dim)
    closed_A, closed_D, report = close_candidate(A, D, covariance, weight, anchor_rank=3)
    # The established closure canonicalizes the encoder gauge before fixing
    # it for the decoder solve. BF16 export preserves the subspace to rounding.
    for before, after in zip(A, closed_A):
        q = torch.linalg.qr(before.float()).Q
        remainder = after.float() - q @ q.T @ after.float()
        assert remainder.norm() / after.float().norm() < 0.005
    assert report['closure'] == 'full_layer_closed_form_decoder_refit'
    assert report['relative_residual'] < 1e-4
    products = torch.stack([closed_A[h // 2].float() @ closed_D[h].float() for h in range(heads)])
    residual = weight.T - products.reshape(heads * dim, hidden)
    initial_loss = (weight @ covariance * weight).sum()
    deployed_loss = (residual.T @ covariance * residual.T).sum()
    assert deployed_loss < initial_loss
    kept_A, kept_D, report = close_candidate(A, D, covariance, weight, anchor_rank=rank)
    assert torch.equal(kept_A, A) and torch.equal(kept_D, D)
    assert report['closure'] == 'uniform_anchor_factor_bank'


def test_full_rank_endpoint_reproduces_dense_v_and_o():
    torch.manual_seed(41)
    A = torch.zeros(2, 4, 4)
    D = torch.zeros(4, 4, 6)
    v = torch.randn(8, 6).bfloat16()
    o = torch.randn(6, 16).bfloat16()
    A, D, report = close_candidate(A, D, torch.eye(16), o)
    folded_v, folded_o = _fold_ragged_to_padded_weights(
        dense_v_weight=v, A=A, D=D, source_ranks=(4, 4))
    assert torch.equal(folded_v, v.float())
    assert torch.equal(folded_o, o.float())
    assert report['closure'] == 'exact_identity_dense_o_endpoint'


def test_allocation_is_attention_only_exact_mean96_and_global_minimum():
    results = {rank: dict(layers=list(LAYERS), records=[dict(layer=layer,
        fit_config=dict(cache_rank_per_head=rank),
        heldout=dict(factor_dtype_relative_mse=(128 - rank) / 128)) for layer in LAYERS])
        for rank in RANKS}
    result = allocate(results, [0.1, 0.2, 0.3, 0.4], [-0.5, -0.4, -0.3, -0.2],
        layers=LAYERS,target_mean_rank=96)
    selected = result['layer_ranks']
    assert len(selected) == 4 and sum(selected) == 4 * 96
    curves = result['predicted_costs']
    minimum = min(sum(curve[rank] for curve, rank in zip(curves, ranks))
        for ranks in itertools.product(RANKS, repeat=4) if sum(ranks) == 384)
    assert result['predicted_cost'] == pytest.approx(minimum)
    results[32]['records'][0]['layer'] = 0
    with pytest.raises(AssertionError):
        allocate(results, [0.1] * 4, [-0.1] * 4,layers=LAYERS,target_mean_rank=96)
