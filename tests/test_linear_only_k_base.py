"""Independent least-squares and runtime checks for intercept-free K Base."""

import torch

from basisserve.core.c1_v_conditional_k_router import (
    build_conditional_routing_sidecar,
    fit_affine_reduced_rank_map,
)


def fit(x, y, rank, fit_bias):
    return fit_affine_reduced_rank_map(row_count=len(x), input_sum=x.sum(0),
        target_sum=y.sum(0), input_gram=x.T @ x, input_target_gram=x.T @ y,
        rank=rank, fit_bias=fit_bias)


def test_linear_only_matches_uncentered_lstsq_with_nonzero_means():
    rng = torch.Generator().manual_seed(17)
    x = torch.randn(200, 5, generator=rng, dtype=torch.float64) + 2
    y = x @ torch.randn(5, 4, generator=rng, dtype=torch.float64) + 7
    model = fit(x, y, 4, False)
    expected = torch.linalg.lstsq(x, y).solution
    torch.testing.assert_close(model.weight, expected)
    assert torch.count_nonzero(model.bias) == 0
    affine = fit(x, y, 4, True)
    assert (model.apply(x)-y).square().sum() > (affine.apply(x)-y).square().sum()
    # Clearing an affine intercept does not solve the constrained problem.
    assert (model.apply(x)-y).square().sum() < (x @ affine.weight-y).square().sum()


def test_linear_rank_constraint_matches_prediction_space_svd():
    rng = torch.Generator().manual_seed(31)
    x = torch.randn(150, 6, generator=rng, dtype=torch.float64) + 1.5
    y = torch.randn(150, 4, generator=rng, dtype=torch.float64) + 3
    least_squares = x @ torch.linalg.lstsq(x, y).solution
    u, s, vh = torch.linalg.svd(least_squares, full_matrices=False)
    for rank in (0, 1, 2, 4):
        model = fit(x, y, rank, False)
        torch.testing.assert_close(model.apply(x), (u[:, :rank] * s[:rank]) @ vh[:rank])
        assert torch.linalg.matrix_rank(model.weight) <= rank
        affine = fit(x, y, rank, True)
        assert (affine.apply(x)-y).square().sum() <= (model.apply(x)-y).square().sum() + 1e-8


def test_linear_singular_input_and_zero_input_have_no_intercept():
    rng = torch.Generator().manual_seed(43)
    source = torch.randn(90, 2, generator=rng, dtype=torch.float64)
    x = torch.cat([source, source, torch.zeros(90, 1, dtype=torch.float64)], dim=1)
    y = torch.randn(90, 3, generator=rng, dtype=torch.float64) + 4
    model = fit(x, y, 3, False)
    torch.testing.assert_close(model.apply(x), x @ torch.linalg.pinv(x) @ y)
    empty = fit(torch.zeros_like(x), y, 3, False)
    assert torch.count_nonzero(empty.apply(x)) == 0


def test_linear_base_runtime_rotates_then_recomputes_residual():
    rng = torch.Generator().manual_seed(59)
    x = torch.randn(100, 3, generator=rng, dtype=torch.float64) + 2
    y = torch.randn(100, 4, generator=rng, dtype=torch.float64) + 5
    model = fit(x, y, 2, False)
    codes = x[:7].reshape(1, 1, 7, 3)
    exact = y[:7].reshape(1, 1, 7, 4)
    angles = torch.randn(1, 7, 2, generator=rng, dtype=torch.float64).repeat(1, 1, 2)
    cos, sin = angles.cos(), angles.sin()
    encoder = torch.randn(1, 4, 2, generator=rng, dtype=torch.float64)
    sidecar = build_conditional_routing_sidecar(codes, exact, base_left=model.left[None],
        base_right=model.right[None], base_bias=model.bias[None], residual_encoder=encoder,
        cos=cos, sin=sin)
    pre = model.apply(codes)
    rotated = pre * cos[:, None] + torch.cat([-pre[..., 2:], pre[..., :2]], -1) * sin[:, None]
    torch.testing.assert_close(sidecar[..., :4], rotated)
    torch.testing.assert_close(sidecar[..., 4:], (exact-rotated) @ encoder[0])
