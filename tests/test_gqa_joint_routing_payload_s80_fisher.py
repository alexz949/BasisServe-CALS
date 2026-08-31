import torch

from basisserve.calibration.gqa_joint_routing_payload_s80_stats import (
    S80DirectResidualData,
)
from basisserve.core.gqa_joint_routing_payload_s80_fisher import (
    prepare_softmax_fisher_routing,
    softmax_fisher_adjoint,
    softmax_fisher_routing_diagnostics,
    softmax_fisher_routing_loss,
    softmax_fisher_transform,
)


def test_softmax_fisher_residual_matches_closed_form() -> None:
    generator = torch.Generator().manual_seed(17)
    teacher = torch.randn(3, 4, 11, generator=generator, dtype=torch.float64)
    delta = torch.randn(3, 4, 11, generator=generator, dtype=torch.float64)
    probability = torch.softmax(teacher, dim=-1)

    residual = softmax_fisher_transform(delta, probability)
    expected = 0.5 * (
        torch.sum(probability * delta.square(), dim=-1)
        - torch.sum(probability * delta, dim=-1).square()
    )

    torch.testing.assert_close(residual.square().sum(dim=-1), expected)


def test_softmax_fisher_ignores_row_constant_and_has_exact_adjoint() -> None:
    generator = torch.Generator().manual_seed(23)
    teacher = torch.randn(2, 5, 13, generator=generator, dtype=torch.float64)
    probability = torch.softmax(teacher, dim=-1)
    constant = torch.randn(2, 5, 1, generator=generator, dtype=torch.float64)
    probe = torch.randn(2, 5, 13, generator=generator, dtype=torch.float64)
    residual_probe = torch.randn(
        2,
        5,
        13,
        generator=generator,
        dtype=torch.float64,
    )

    torch.testing.assert_close(
        softmax_fisher_transform(constant.expand_as(teacher), probability),
        torch.zeros_like(teacher),
        atol=1e-14,
        rtol=1e-14,
    )
    left = torch.sum(
        softmax_fisher_transform(probe, probability) * residual_probe
    )
    right = torch.sum(
        probe * softmax_fisher_adjoint(residual_probe, probability)
    )
    torch.testing.assert_close(left, right)


def test_exact_key_subspace_has_zero_softmax_fisher_loss() -> None:
    generator = torch.Generator().manual_seed(29)
    documents = 3
    tokens = 7
    query_heads = 4
    kv_heads = 2
    value_dim = 2
    key_dim = 3
    joint_dim = value_dim + key_dim
    queries = torch.randn(
        documents,
        query_heads,
        key_dim,
        generator=generator,
        dtype=torch.float64,
    )
    rows = torch.randn(
        documents,
        tokens,
        kv_heads,
        joint_dim,
        generator=generator,
        dtype=torch.float64,
    )
    direct = S80DirectResidualData(queries, rows)
    mapping = torch.tensor([0, 0, 1, 1])
    prepared = prepare_softmax_fisher_routing(
        direct,
        head_to_kv_group=mapping,
        value_dim=value_dim,
        key_dim=key_dim,
        device="cpu",
        dtype=torch.float64,
    )
    encoder = torch.zeros(kv_heads, joint_dim, key_dim, dtype=torch.float64)
    encoder[:, value_dim:] = torch.eye(key_dim, dtype=torch.float64)
    query_factor = torch.eye(key_dim, dtype=torch.float64).expand(
        query_heads,
        -1,
        -1,
    )

    observed = softmax_fisher_routing_loss(
        prepared,
        routing_payload_encoders=encoder,
        routing_query_factors=query_factor,
    )

    assert abs(observed) < 1e-24


def test_exact_proxy_retrieves_every_page_at_full_budget() -> None:
    generator = torch.Generator().manual_seed(31)
    documents = 2
    tokens = 7
    query_heads = 4
    kv_heads = 2
    value_dim = 2
    key_dim = 3
    joint_dim = value_dim + key_dim
    queries = torch.randn(
        documents,
        query_heads,
        key_dim,
        generator=generator,
        dtype=torch.float64,
    )
    rows = torch.randn(
        documents,
        tokens,
        kv_heads,
        joint_dim,
        generator=generator,
        dtype=torch.float64,
    )
    mapping = torch.tensor([0, 0, 1, 1])
    prepared = prepare_softmax_fisher_routing(
        S80DirectResidualData(queries, rows),
        head_to_kv_group=mapping,
        value_dim=value_dim,
        key_dim=key_dim,
        device="cpu",
        dtype=torch.float64,
    )
    route_encoder = torch.zeros(kv_heads, joint_dim, key_dim, dtype=torch.float64)
    route_encoder[:, value_dim:] = torch.eye(key_dim, dtype=torch.float64)
    payload_encoder = torch.randn(
        kv_heads,
        joint_dim,
        1,
        generator=generator,
        dtype=torch.float64,
    )
    query_factor = torch.eye(key_dim, dtype=torch.float64).expand(
        query_heads,
        -1,
        -1,
    )
    decoder = torch.randn(
        query_heads,
        key_dim + 1,
        6,
        generator=generator,
        dtype=torch.float64,
    )

    diagnostics = softmax_fisher_routing_diagnostics(
        prepared,
        routing_payload_encoders=route_encoder,
        payload_only_encoders=payload_encoder,
        routing_query_factors=query_factor,
        payload_decoders=decoder,
        page_size=4,
        exact_token_budget=tokens,
    )

    assert diagnostics["raw_score_nmse"] < 1e-24
    assert diagnostics["softmax_fisher_nmse"] < 1e-24
    assert diagnostics["physical_page_recall_minimum"] == 1.0
    assert diagnostics["attention_mass_recall_minimum"] > 1 - 1e-12
    assert diagnostics["selected_token_fraction_mean"] == 1.0
    assert diagnostics["exact_refined_output_relative_mse"] < 1e-24
