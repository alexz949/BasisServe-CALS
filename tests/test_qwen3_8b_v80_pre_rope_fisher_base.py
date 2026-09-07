import torch

from basisserve.core.c1_v_conditional_k_router import AffineReducedRankMap
from evaluation.eval_qwen3_8b_v80_pre_rope_fisher_base import (
    _balanced_factors,
    _page_fisher_document_loss,
)


def test_balanced_factors_preserve_affine_map() -> None:
    generator = torch.Generator().manual_seed(307)
    maps = tuple(
        AffineReducedRankMap(
            left=torch.randn(5, 2, generator=generator),
            right=torch.randn(2, 4, generator=generator),
            bias=torch.randn(4, generator=generator),
        )
        for _ in range(2)
    )
    left, right, bias = _balanced_factors(maps)
    for group, item in enumerate(maps):
        torch.testing.assert_close(left[group] @ right[group], item.left @ item.right)
        torch.testing.assert_close(bias[group], item.bias)


def test_pre_rope_page_fisher_loss_is_zero_for_exact_predictor() -> None:
    generator = torch.Generator().manual_seed(311)
    tokens, groups, head_dim = 16, 2, 4
    heads_per_group, queries_per_document = 2, 2
    dense_value = torch.randn(tokens, groups, head_dim, generator=generator)
    exact_key = dense_value.clone()
    rows = torch.cat((dense_value, exact_key), dim=-1)
    queries = torch.randn(
        queries_per_document,
        groups * heads_per_group,
        head_dim,
        generator=generator,
    )
    identity = torch.eye(head_dim).expand(groups, -1, -1).clone()
    bias = torch.zeros(groups, head_dim)
    cos = torch.ones(1, tokens, head_dim)
    sin = torch.zeros_like(cos)
    loss, energy = _page_fisher_document_loss(
        queries,
        rows,
        query_positions=torch.tensor([7, 15]),
        value_encoder=identity,
        left=identity,
        right=identity,
        bias=bias,
        cos=cos,
        sin=sin,
        page_size=2,
        pinned_prefix_pages=1,
    )
    assert float(loss) == 0.0
    assert float(energy) > 0.0


def test_pre_rope_page_fisher_loss_has_finite_factor_gradients() -> None:
    generator = torch.Generator().manual_seed(313)
    tokens, groups, head_dim, value_rank, rank = 16, 2, 4, 3, 2
    rows = torch.randn(tokens, groups, 2 * head_dim, generator=generator)
    queries = torch.randn(2, 4, head_dim, generator=generator)
    value_encoder = torch.randn(
        groups,
        head_dim,
        value_rank,
        generator=generator,
    )
    left = torch.randn(
        groups,
        value_rank,
        rank,
        generator=generator,
        requires_grad=True,
    )
    right = torch.randn(
        groups,
        rank,
        head_dim,
        generator=generator,
        requires_grad=True,
    )
    bias = torch.randn(
        groups,
        head_dim,
        generator=generator,
        requires_grad=True,
    )
    cos = torch.ones(1, tokens, head_dim)
    sin = torch.zeros_like(cos)
    loss, _ = _page_fisher_document_loss(
        queries,
        rows,
        query_positions=torch.tensor([7, 15]),
        value_encoder=value_encoder,
        left=left,
        right=right,
        bias=bias,
        cos=cos,
        sin=sin,
        page_size=2,
        pinned_prefix_pages=1,
    )
    loss.backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in (left, right, bias)
    )
