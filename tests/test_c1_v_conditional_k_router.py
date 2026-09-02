import torch

from basisserve.core.c1_v_conditional_k_router import (
    build_conditional_routing_sidecar,
    conditional_routing_query_projector,
    fit_affine_reduced_rank_map,
    residual_page_fisher_gram,
)
from basisserve.core.c1_k_routing_sidecar import routing_proxy_scores


def _moments(inputs: torch.Tensor, targets: torch.Tensor) -> dict[str, object]:
    return {
        "row_count": inputs.shape[0],
        "input_sum": inputs.sum(dim=0),
        "target_sum": targets.sum(dim=0),
        "input_gram": inputs.mT @ inputs,
        "input_target_gram": inputs.mT @ targets,
    }


def test_affine_reduced_rank_map_is_rank_bounded_and_nested() -> None:
    generator = torch.Generator().manual_seed(7)
    inputs = torch.randn(400, 6, generator=generator, dtype=torch.float64)
    full_weight = torch.randn(6, 5, generator=generator, dtype=torch.float64)
    bias = torch.randn(5, generator=generator, dtype=torch.float64)
    targets = inputs @ full_weight + bias + 0.02 * torch.randn(
        400, 5, generator=generator, dtype=torch.float64
    )
    rank_two = fit_affine_reduced_rank_map(**_moments(inputs, targets), rank=2)
    rank_five = fit_affine_reduced_rank_map(**_moments(inputs, targets), rank=5)
    error_two = (rank_two.apply(inputs) - targets).square().sum()
    error_five = (rank_five.apply(inputs) - targets).square().sum()
    assert torch.linalg.matrix_rank(rank_two.weight) <= 2
    assert error_five <= error_two
    torch.testing.assert_close(
        rank_five.apply(inputs).mean(dim=0),
        targets.mean(dim=0),
        rtol=1e-10,
        atol=1e-10,
    )


def test_residual_page_fisher_uses_exact_key_teacher() -> None:
    generator = torch.Generator().manual_seed(11)
    queries = torch.randn(3, 4, generator=generator, dtype=torch.float64)
    exact_key = torch.randn(8, 4, generator=generator, dtype=torch.float64)
    residual = torch.randn(8, 4, generator=generator, dtype=torch.float64)
    scaling = 0.5
    grams, energy = residual_page_fisher_gram(
        queries,
        exact_key,
        residual,
        scaling=scaling,
        page_size=2,
        excluded_prefix_pages=0,
    )

    probabilities = torch.softmax(scaling * queries @ exact_key.mT, dim=-1)
    probabilities = probabilities.reshape(3, 4, 2)
    mass = probabilities.sum(dim=-1)
    page_rows = torch.einsum(
        "hps,psd->hpd",
        probabilities,
        residual.reshape(4, 2, 4),
    ) / mass.unsqueeze(-1)
    mean = torch.einsum("hp,hpd->hd", mass, page_rows)
    centered = page_rows - mean.unsqueeze(1)
    expected_grams = torch.bmm(
        (centered * torch.sqrt(mass).unsqueeze(-1)).mT,
        centered * torch.sqrt(mass).unsqueeze(-1),
    )
    page_scores = scaling * torch.einsum("hd,hpd->hp", queries, page_rows)
    expected_energy = 0.5 * torch.sum(
        mass
        * (
            page_scores
            - torch.sum(mass * page_scores, dim=-1, keepdim=True)
        ).square()
    )
    torch.testing.assert_close(grams, expected_grams)
    torch.testing.assert_close(
        torch.tensor(energy, dtype=expected_energy.dtype),
        expected_energy,
    )


def test_zero_residual_has_zero_page_fisher_energy() -> None:
    queries = torch.randn(2, 4, dtype=torch.float64)
    exact_key = torch.randn(7, 4, dtype=torch.float64)
    grams, energy = residual_page_fisher_gram(
        queries,
        exact_key,
        torch.zeros_like(exact_key),
        scaling=0.5,
        page_size=3,
        excluded_prefix_pages=0,
    )
    assert torch.count_nonzero(grams) == 0
    assert energy == 0.0


def test_residual_page_fisher_conditions_on_non_sink_pages() -> None:
    generator = torch.Generator().manual_seed(19)
    queries = torch.randn(3, 4, generator=generator, dtype=torch.float64)
    exact_key = torch.randn(10, 4, generator=generator, dtype=torch.float64)
    residual = torch.randn(10, 4, generator=generator, dtype=torch.float64)
    conditioned = residual_page_fisher_gram(
        queries,
        exact_key,
        residual,
        scaling=0.5,
        page_size=2,
        excluded_prefix_pages=1,
    )
    sliced = residual_page_fisher_gram(
        queries,
        exact_key[2:],
        residual[2:],
        scaling=0.5,
        page_size=2,
        excluded_prefix_pages=0,
    )
    torch.testing.assert_close(conditioned[0], sliced[0])
    torch.testing.assert_close(
        torch.tensor(conditioned[1]),
        torch.tensor(sliced[1]),
    )


def test_conditional_sidecar_reproduces_base_plus_residual_scores() -> None:
    generator = torch.Generator().manual_seed(23)
    batch, groups, heads_per_group = 2, 2, 3
    tokens, value_rank, base_rank, head_dim, residual_rank = 5, 4, 3, 6, 2
    value = torch.randn(
        batch, groups, tokens, value_rank, generator=generator
    )
    exact_key = torch.randn(
        batch, groups, tokens, head_dim, generator=generator
    )
    query = torch.randn(
        batch, groups * heads_per_group, head_dim, generator=generator
    )
    base_left = torch.randn(groups, value_rank, base_rank, generator=generator)
    base_right = torch.randn(groups, base_rank, head_dim, generator=generator)
    base_bias = torch.randn(groups, head_dim, generator=generator)
    residual_encoder = torch.randn(
        groups, head_dim, residual_rank, generator=generator
    )
    residual_query = torch.randn(
        groups * heads_per_group,
        head_dim,
        residual_rank,
        generator=generator,
    )
    angles = torch.randn(batch, tokens, head_dim // 2, generator=generator)
    cos = torch.cat((angles.cos(), angles.cos()), dim=-1)
    sin = torch.cat((angles.sin(), angles.sin()), dim=-1)
    sidecar = build_conditional_routing_sidecar(
        value,
        exact_key,
        base_left=base_left,
        base_right=base_right,
        base_bias=base_bias,
        residual_encoder=residual_encoder,
        cos=cos,
        sin=sin,
    )
    projector = conditional_routing_query_projector(residual_query)

    base_pre = torch.einsum(
        "bhtv,hvr,hrd->bhtd", value, base_left, base_right
    ) + base_bias[None, :, None]
    half = head_dim // 2
    rotated = torch.cat((-base_pre[..., half:], base_pre[..., :half]), dim=-1)
    base_post = base_pre * cos[:, None] + rotated * sin[:, None]
    residual_code = torch.einsum(
        "bhtd,hdr->bhtr", exact_key - base_post, residual_encoder
    )
    scale = head_dim**-0.5
    for item in range(batch):
        observed = routing_proxy_scores(
            query[item],
            sidecar[item],
            projector,
            head_dim=head_dim,
        )
        expected = torch.empty_like(observed)
        for group in range(groups):
            first = group * heads_per_group
            stop = first + heads_per_group
            q = query[item, first:stop]
            expected[first:stop] = scale * (
                q @ base_post[item, group].mT
                + (torch.einsum("hd,hdr->hr", q, residual_query[first:stop]))
                @ residual_code[item, group].mT
            )
        torch.testing.assert_close(observed, expected)
