import pytest
import torch

from evaluation.fit_qwen35_k_router import fit_base
from evaluation.eval_qwen3_8b_v80_conditional_residual_router import _value_codes, _stack_base_map, _apply_base
from evaluation.eval_qwen3_8b_v80_conditional_residual_router import _fit_residual_grid, S80CompactSoftmaxFisherRouting


@pytest.mark.parametrize('rank', [16, 32])
def test_both_capacities_recover_known_affine_map(rank):
    torch.manual_seed(37)
    encoder = torch.linalg.qr(torch.randn(2, 40, 36)).Q
    values = torch.randn(3, 64, 2, 40)
    left, right, bias = torch.randn(2, 36, rank), torch.randn(2, rank, 40), torch.randn(2, 40)
    targets = torch.stack([_apply_base(_value_codes(value, encoder), (left, right, bias)) for value in values])
    rows = torch.cat((values, torch.zeros_like(values)), -1)
    fitted = fit_base(rows, targets, encoder, rank)[rank]
    factors = _stack_base_map(fitted, device='cpu')
    fresh = torch.randn(64, 2, 40)
    codes = _value_codes(fresh, encoder)
    actual = _apply_base(codes, factors)
    expected = _apply_base(codes, (left, right, bias))
    relative = (actual-expected).norm()/expected.norm()
    assert relative < 1e-5
    assert factors[0].shape == (2, 36, rank) and factors[1].shape == (2, rank, 40)


@pytest.mark.parametrize('rank', [16, 32])
def test_residual_fit_and_export_for_both_capacities(rank):
    torch.manual_seed(53)
    queries = torch.randn(4, 32, 64)
    grams = torch.eye(64).repeat(4, 32, 1, 1)
    statistics = S80CompactSoftmaxFisherRouting(queries_by_head=queries,
        fisher_grams_by_head=grams, head_to_kv_group=torch.zeros(4, dtype=torch.long),
        value_dim=0, key_dim=64, scaling=64**-0.5,
        teacher_fisher_energy=float(queries.square().sum())/64)
    factors, losses = _fit_residual_grid({rank: statistics}, {rank: statistics},
        residual_ranks=(rank,), sweeps=1, relative_damping=1e-5,
        iterative_tolerance=1e-5, iterative_max_iterations=2, device=torch.device('cpu'))
    encoder, query = factors[rank, rank]
    assert encoder.shape == (1, 64, rank) and query.shape == (4, 64, rank)
    assert torch.isfinite(encoder).all() and torch.isfinite(query).all()
    assert losses[f'b{rank}_r{rank}']['fit_page_fisher_nmse'] <= 1.00001
