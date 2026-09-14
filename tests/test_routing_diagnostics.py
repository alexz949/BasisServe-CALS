import pytest
import torch

from evaluation.routing_diagnostics import window_metrics


@pytest.mark.parametrize('dim,rotary_dim', [(16, 16), (256, 64)])
def test_full_router_metrics_have_known_reconstruction_and_mass(dim, rotary_dim):
    torch.manual_seed(83)
    key = torch.randn(1, 2, 97, dim)
    value = torch.randn(1, 2, 97, 8)
    queries = torch.zeros(1, 1, 4, dim)
    cos = torch.ones(1, 97, rotary_dim)
    factors = {
        f'base_left_b{dim}': torch.zeros(2, 8, dim),
        f'base_right_b{dim}': torch.zeros(2, dim, dim),
        f'base_bias_b{dim}': torch.zeros(2, dim),
        f'residual_encoder_b{dim}_r{dim}': torch.eye(dim).repeat(2, 1, 1),
        f'residual_query_b{dim}_r{dim}': torch.eye(dim).repeat(4, 1, 1),
    }
    result = window_metrics(value, key, queries, [96], cos, torch.zeros_like(cos),
        rank=dim, factors=factors, budget=64)
    assert result['relative_mse'] == 0
    assert result['attention_mass'] == pytest.approx(64/97)
    assert result['non_sink_attention_mass'] == pytest.approx(64/65)
    assert result['exact_attention_mass'] == result['attention_mass']
    assert result['exact_non_sink_attention_mass'] == result['non_sink_attention_mass']
    factors[f'residual_encoder_b{dim}_r{dim}'].zero_()
    result = window_metrics(value, key, queries, [32], cos, torch.zeros_like(cos),
        rank=dim, factors=factors, budget=64)
    assert result['relative_mse'] == 1
    assert result['attention_mass'] == pytest.approx(1)
    assert result['non_sink_attention_mass'] == pytest.approx(1)
