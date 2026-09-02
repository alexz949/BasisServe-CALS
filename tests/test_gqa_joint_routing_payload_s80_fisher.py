import torch

from basisserve.calibration.gqa_joint_routing_payload_s80_stats import (
    S80DirectResidualData,
)
from basisserve.core.gqa_joint_routing_payload_s80_fisher import (
    compact_softmax_fisher_adapter_system,
    compact_softmax_fisher_encoder_diagonal,
    compact_softmax_fisher_loss,
    compact_softmax_fisher_roots,
    pack_symmetric_fisher_grams,
    page_softmax_fisher_gram,
    prepare_compact_softmax_fisher_routing,
    prepare_softmax_fisher_routing,
    softmax_fisher_adapter_system,
    softmax_fisher_encoder_diagonal,
    softmax_fisher_adjoint,
    softmax_fisher_routing_diagnostics,
    softmax_fisher_routing_loss,
    softmax_fisher_gram,
    softmax_fisher_transform,
)


def test_page_fisher_size_one_matches_token_fisher() -> None:
    generator = torch.Generator().manual_seed(11)
    queries = torch.randn(3, 4, generator=generator, dtype=torch.float64)
    rows = torch.randn(7, 9, generator=generator, dtype=torch.float64)
    token_grams, token_energy = softmax_fisher_gram(
        queries,
        rows,
        value_dim=5,
        scaling=0.5,
    )
    page_grams, page_energy = page_softmax_fisher_gram(
        queries,
        rows,
        value_dim=5,
        scaling=0.5,
        page_size=1,
    )
    torch.testing.assert_close(page_grams, token_grams)
    assert abs(page_energy - token_energy) < 1e-12


def test_page_fisher_removes_within_page_directions() -> None:
    generator = torch.Generator().manual_seed(13)
    queries = torch.randn(2, 3, generator=generator, dtype=torch.float64)
    rows = torch.randn(5, 7, generator=generator, dtype=torch.float64)
    grams, energy = page_softmax_fisher_gram(
        queries,
        rows,
        value_dim=4,
        scaling=3**-0.5,
        page_size=5,
    )
    torch.testing.assert_close(grams, torch.zeros_like(grams), atol=1e-14, rtol=0)
    assert abs(energy) < 1e-14


def _compact_fisher(prepared):
    rows = prepared.joint_rows_by_group.index_select(
        0,
        prepared.head_to_kv_group,
    )
    probabilities = prepared.probabilities_by_head
    means = torch.einsum("hdt,hdti->hdi", probabilities, rows)
    centered = rows - means.unsqueeze(2)
    grams = torch.einsum(
        "hdt,hdti,hdtj->hdij",
        probabilities,
        centered,
        centered,
    )
    return prepare_compact_softmax_fisher_routing(
        queries_by_head=prepared.queries_by_head,
        fisher_grams_packed_by_head=pack_symmetric_fisher_grams(grams),
        head_to_kv_group=prepared.head_to_kv_group,
        value_dim=prepared.value_dim,
        key_dim=prepared.key_dim,
        scaling=prepared.scaling,
        teacher_fisher_energy=prepared.teacher_fisher_energy,
        device=prepared.queries_by_head.device,
        dtype=prepared.queries_by_head.dtype,
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
        selection_routing_encoders=route_encoder,
        payload_routing_encoders=route_encoder,
        payload_only_encoders=payload_encoder,
        routing_query_factors=query_factor,
        payload_decoders=decoder,
        page_size=4,
        exact_token_budget=tokens,
    )

    assert diagnostics["raw_score_nmse"] < 1e-24
    assert diagnostics["softmax_fisher_nmse"] < 1e-24


def test_selection_and_payload_latents_can_be_decoupled() -> None:
    generator = torch.Generator().manual_seed(37)
    documents = 2
    tokens = 8
    query_heads = 4
    kv_heads = 2
    value_dim = 2
    key_dim = 3
    joint_dim = value_dim + key_dim
    mapping = torch.tensor([0, 0, 1, 1])
    prepared = prepare_softmax_fisher_routing(
        S80DirectResidualData(
            torch.randn(
                documents,
                query_heads,
                key_dim,
                generator=generator,
                dtype=torch.float64,
            ),
            torch.randn(
                documents,
                tokens,
                kv_heads,
                joint_dim,
                generator=generator,
                dtype=torch.float64,
            ),
        ),
        head_to_kv_group=mapping,
        value_dim=value_dim,
        key_dim=key_dim,
        device="cpu",
        dtype=torch.float64,
    )
    selector = torch.zeros(kv_heads, joint_dim, key_dim, dtype=torch.float64)
    selector[:, value_dim:] = torch.eye(key_dim, dtype=torch.float64)
    query_factor = torch.eye(key_dim, dtype=torch.float64).expand(
        query_heads,
        -1,
        -1,
    )
    payload_encoder = torch.randn(
        kv_heads,
        joint_dim,
        2,
        generator=generator,
        dtype=torch.float64,
    )
    decoder = torch.randn(
        query_heads,
        2,
        6,
        generator=generator,
        dtype=torch.float64,
    )

    diagnostics = softmax_fisher_routing_diagnostics(
        prepared,
        selection_routing_encoders=selector,
        payload_routing_encoders=payload_encoder[..., :0],
        payload_only_encoders=payload_encoder,
        routing_query_factors=query_factor,
        payload_decoders=decoder,
        page_size=4,
        exact_token_budget=tokens,
    )

    assert diagnostics["raw_score_nmse"] < 1e-24
    assert diagnostics["exact_refined_output_relative_mse"] < 1e-24
    assert diagnostics["physical_page_recall_minimum"] == 1.0
    assert diagnostics["attention_mass_recall_minimum"] > 1 - 1e-12
    assert diagnostics["selected_token_fraction_mean"] == 1.0
    assert diagnostics["exact_refined_output_relative_mse"] < 1e-24


def test_compact_fisher_statistics_match_raw_routing_systems() -> None:
    generator = torch.Generator().manual_seed(37)
    documents = 3
    tokens = 11
    query_heads = 4
    kv_heads = 1
    value_dim = 3
    key_dim = 3
    joint_dim = value_dim + key_dim
    route_rank = 2
    mapping = torch.zeros(query_heads, dtype=torch.long)
    direct = S80DirectResidualData(
        torch.randn(
            documents,
            query_heads,
            key_dim,
            generator=generator,
            dtype=torch.float64,
        ),
        torch.randn(
            documents,
            tokens,
            kv_heads,
            joint_dim,
            generator=generator,
            dtype=torch.float64,
        ),
    )
    raw = prepare_softmax_fisher_routing(
        direct,
        head_to_kv_group=mapping,
        value_dim=value_dim,
        key_dim=key_dim,
        device="cpu",
        dtype=torch.float64,
    )
    compact = _compact_fisher(raw)
    encoders = torch.randn(
        kv_heads,
        joint_dim,
        route_rank,
        generator=generator,
        dtype=torch.float64,
    )
    query_factors = torch.randn(
        query_heads,
        key_dim,
        route_rank,
        generator=generator,
        dtype=torch.float64,
    )

    raw_loss = softmax_fisher_routing_loss(
        raw,
        routing_payload_encoders=encoders,
        routing_query_factors=query_factors,
    )
    compact_loss = compact_softmax_fisher_loss(
        compact,
        routing_payload_encoders=encoders,
        routing_query_factors=query_factors,
    )
    torch.testing.assert_close(
        torch.tensor(compact_loss),
        torch.tensor(raw_loss),
    )

    projected_queries = torch.einsum(
        "hdk,hkr->dhr",
        raw.queries_by_head,
        query_factors,
    )
    raw_diagonal = softmax_fisher_encoder_diagonal(
        joint_rows=raw.joint_rows_by_group[0],
        projected_queries=projected_queries,
        probabilities=raw.probabilities_by_head.permute(1, 0, 2),
        scaling=raw.scaling,
    )
    compact_diagonal = compact_softmax_fisher_encoder_diagonal(
        compact,
        head_indices=torch.arange(query_heads),
        projected_queries=projected_queries,
    )
    torch.testing.assert_close(compact_diagonal, raw_diagonal)

    raw_system = softmax_fisher_adapter_system(
        raw,
        routing_payload_encoders=encoders,
        routing_query_factors=query_factors,
    )
    compact_system = compact_softmax_fisher_adapter_system(
        compact,
        routing_payload_encoders=encoders,
        routing_query_factors=query_factors,
    )
    for compact_value, raw_value in zip(compact_system, raw_system, strict=True):
        torch.testing.assert_close(compact_value, raw_value)

    roots = compact_softmax_fisher_roots(compact)
    reconstructed = roots @ roots.mT
    torch.testing.assert_close(
        reconstructed,
        compact.fisher_grams_by_head,
        atol=1e-12,
        rtol=1e-12,
    )
