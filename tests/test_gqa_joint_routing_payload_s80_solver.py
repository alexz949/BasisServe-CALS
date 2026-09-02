from __future__ import annotations

import torch

import basisserve.core.gqa_joint_routing_payload_s80 as s80_core
from basisserve.calibration.gqa_joint_routing_payload_s80_stats import (
    S80DirectResidualData,
    S80PayloadAccumulator,
    S80RoutingAccumulator,
    joint_token_features,
)
from basisserve.core.gqa_joint_routing_payload_s80 import (
    S80Factors,
    S80Layout,
    evaluate_s80_objective,
    fit_s80_joint,
    fold_s80_factors,
    gauge_canonicalize_s80,
    initialize_s80_from_c1_and_kq,
    routing_loss,
    s80_objective_from_statistics,
)
from basisserve.core.gqa_joint_routing_payload_s80_fisher import (
    pack_symmetric_fisher_grams,
    prepare_compact_softmax_fisher_routing,
    prepare_softmax_fisher_routing,
)


def _layout() -> S80Layout:
    return S80Layout(6, 4, 2, 3, 2, 3, 2)


def _problem():
    torch.manual_seed(20260901)
    layout = _layout()
    rows = 96
    routed_value = torch.randn(rows, 4, 3, dtype=torch.float64)
    routed_key = torch.randn(rows, 4, 2, dtype=torch.float64)
    dense_blocks = torch.randn(4, 3, 6, dtype=torch.float64)
    dense_output = torch.einsum("nhd,hdo->no", routed_value, dense_blocks)
    payload_accumulator = S80PayloadAccumulator(
        num_query_heads=4,
        value_dim=3,
        key_dim=2,
        accumulation_dtype=torch.float64,
    )
    payload_accumulator.update(routed_value, routed_key, dense_output)
    payload_statistics = payload_accumulator.finalize()

    routing_accumulator = S80RoutingAccumulator(
        head_to_kv_group=layout.head_to_kv_group(),
        value_dim=3,
        key_dim=2,
    )
    routing_inputs = []
    for shard_index in range(2):
        query = torch.randn(1, 4, 2, dtype=torch.float64)
        value = torch.randn(7, 2, 3, dtype=torch.float64)
        key = torch.randn(7, 2, 2, dtype=torch.float64)
        routing_accumulator.update_shard(
            query, value, key, metadata={"shard": shard_index}
        )
        routing_inputs.append((query, joint_token_features(value, key)))
    routing_statistics = routing_accumulator.finalize()
    objective = s80_objective_from_statistics(
        layout=layout,
        payload_statistics=payload_statistics,
        routing_statistics=routing_statistics,
        dense_o_proj_weight=dense_blocks.reshape(12, 6).mT.contiguous(),
        routing_weight=0.7,
    )
    factors = S80Factors(
        torch.randn(2, 5, 2, dtype=torch.float64),
        torch.randn(2, 5, 1, dtype=torch.float64),
        torch.randn(4, 3, 6, dtype=torch.float64),
        torch.randn(4, 2, 2, dtype=torch.float64),
    )
    direct = S80DirectResidualData(
        routing_queries=torch.stack([query[0] for query, _ in routing_inputs]),
        routing_joint_rows=torch.stack([joint for _, joint in routing_inputs]),
    )
    return (
        layout,
        objective,
        factors,
        routed_value,
        routed_key,
        dense_blocks,
        routing_inputs,
        direct,
    )


def _explicit_payload(layout, factors, value, key, dense_blocks):
    joint = joint_token_features(value, key)
    mapping = layout.head_to_kv_group()
    dense = torch.einsum("nhd,hdo->no", value, dense_blocks)
    predicted = sum(
        joint[:, head]
        @ factors.joint_encoders[int(mapping[head])]
        @ factors.payload_decoders[head]
        for head in range(layout.num_attention_heads)
    )
    return (dense - predicted).square().sum() / value.shape[0]


def _explicit_routing(layout, factors, routing_inputs):
    mapping = layout.head_to_kv_group()
    selector = torch.zeros(2, 5, dtype=factors.joint_encoders.dtype)
    selector[:, 3:] = torch.eye(2, dtype=selector.dtype)
    result = torch.zeros((), dtype=selector.dtype)
    for query, joint in routing_inputs:
        for head, group in enumerate(mapping.tolist()):
            delta = (
                factors.routing_query_factors[head]
                @ factors.routing_payload_encoders[group].mT
                - selector
            )
            result += (query[:, head] @ delta @ joint[:, group].mT).square().sum()
    return result


def _compact_fisher(layout: S80Layout, direct: S80DirectResidualData):
    raw = prepare_softmax_fisher_routing(
        direct,
        head_to_kv_group=layout.head_to_kv_group(),
        value_dim=layout.value_dim,
        key_dim=layout.key_dim,
        device="cpu",
        dtype=torch.float64,
    )
    rows = raw.joint_rows_by_group.index_select(0, raw.head_to_kv_group)
    means = torch.einsum("hdt,hdti->hdi", raw.probabilities_by_head, rows)
    centered = rows - means.unsqueeze(2)
    grams = torch.einsum(
        "hdt,hdti,hdtj->hdij",
        raw.probabilities_by_head,
        centered,
        centered,
    )
    return prepare_compact_softmax_fisher_routing(
        queries_by_head=raw.queries_by_head,
        fisher_grams_packed_by_head=pack_symmetric_fisher_grams(grams),
        head_to_kv_group=raw.head_to_kv_group,
        value_dim=raw.value_dim,
        key_dim=raw.key_dim,
        scaling=raw.scaling,
        teacher_fisher_energy=raw.teacher_fisher_energy,
        device="cpu",
        dtype=torch.float64,
    )


def test_covariance_root_system_preserves_gram_and_cross() -> None:
    layout, objective, factors, *_ = _problem()
    errors = torch.bmm(
        factors.joint_encoders.index_select(0, layout.head_to_kv_group()),
        factors.payload_decoders,
    ) - objective.payload_target
    heads = torch.tensor([0, 1])
    design, target = s80_core._payload_covariance_root_system(
        covariance=objective.payload.covariance,
        current_errors=errors,
        head_indices=heads,
    )
    covariance = objective.payload.covariance.index_select(0, heads).index_select(
        1, heads
    )
    expected_gram = covariance.permute(0, 2, 1, 3).reshape(10, 10)
    expected_cross = torch.cat(
        [
            -torch.einsum("hij,hjo->io", objective.payload.covariance[head], errors)
            for head in heads.tolist()
        ]
    )
    torch.testing.assert_close(design.mT @ design, expected_gram)
    torch.testing.assert_close(design.mT @ target, expected_cross)


def test_batched_kronecker_pcg_solves_all_systems() -> None:
    torch.manual_seed(20260909)
    shards, systems, left_dim, right_dim = 5, 3, 4, 2
    left_root = torch.randn(shards, systems, left_dim, left_dim, dtype=torch.float64)
    right_root = torch.randn(
        shards, systems, right_dim, right_dim, dtype=torch.float64
    )
    left = left_root @ left_root.mT
    right = right_root @ right_root.mT
    rhs = torch.randn(systems, left_dim, right_dim, dtype=torch.float64)
    damping = torch.full((systems,), 1e-3, dtype=torch.float64)
    solution, diagnostics = s80_core._batched_kronecker_pcg(
        left_factors=left,
        right_factors=right,
        rhs=rhs,
        absolute_damping=damping,
        relative_tolerance=1e-10,
        max_iterations=100,
    )
    residual = torch.stack(
        [
            sum(left[s, h] @ solution[h] @ right[s, h] for s in range(shards))
            + damping[h] * solution[h]
            - rhs[h]
            for h in range(systems)
        ]
    )
    assert float(torch.linalg.vector_norm(residual)) < 1e-8
    assert all(item.converged for item in diagnostics)


def test_statistics_match_explicit_objectives() -> None:
    layout, objective, factors, value, key, dense_blocks, routing_inputs, _ = _problem()
    measured = evaluate_s80_objective(
        layout=layout, objective=objective, factors=factors
    )
    expected_payload = _explicit_payload(layout, factors, value, key, dense_blocks)
    expected_routing = _explicit_routing(layout, factors, routing_inputs)
    assert abs(measured.payload - float(expected_payload)) < 1e-8
    assert abs(measured.routing - float(expected_routing)) < 1e-8
    assert abs(
        routing_loss(
            layout=layout,
            routing_payload_encoders=factors.routing_payload_encoders,
            routing_query_factors=factors.routing_query_factors,
            routing_statistics=objective.routing,
        )
        - float(expected_routing)
    ) < 1e-8


def test_structured_gauge_preserves_both_products() -> None:
    layout, _, factors, *_ = _problem()
    mapping = layout.head_to_kv_group()
    payload_before = torch.stack(
        [
            factors.joint_encoders[int(mapping[head])] @ factors.payload_decoders[head]
            for head in range(layout.num_attention_heads)
        ]
    )
    routing_before = torch.stack(
        [
            factors.routing_query_factors[head]
            @ factors.routing_payload_encoders[int(mapping[head])].mT
            for head in range(layout.num_attention_heads)
        ]
    )
    canonical, payload_error, routing_error = gauge_canonicalize_s80(
        layout=layout, factors=factors
    )
    payload_after = torch.stack(
        [
            canonical.joint_encoders[int(mapping[head])]
            @ canonical.payload_decoders[head]
            for head in range(layout.num_attention_heads)
        ]
    )
    routing_after = torch.stack(
        [
            canonical.routing_query_factors[head]
            @ canonical.routing_payload_encoders[int(mapping[head])].mT
            for head in range(layout.num_attention_heads)
        ]
    )
    torch.testing.assert_close(payload_after, payload_before)
    torch.testing.assert_close(routing_after, routing_before)
    assert payload_error < 1e-12
    assert routing_error < 1e-12
    identity = torch.eye(layout.joint_rank, dtype=torch.float64)
    for group in range(layout.num_key_value_heads):
        torch.testing.assert_close(
            canonical.joint_encoders[group].mT @ canonical.joint_encoders[group],
            identity,
        )


def test_initializer_preserves_c1_v80_in_frontloaded_coordinates() -> None:
    layout, objective, *_ = _problem()
    torch.manual_seed(20260903)
    value_encoder = torch.linalg.qr(torch.randn(2, 3, 3, dtype=torch.float64)).Q
    value_decoder = torch.randn(4, 3, 6, dtype=torch.float64)
    factors = initialize_s80_from_c1_and_kq(
        layout=layout,
        value_encoders=value_encoder,
        value_decoders=value_decoder,
        key_encoders=torch.randn(2, 2, 2, dtype=torch.float64),
        query_encoders=torch.randn(2, 2, 2, dtype=torch.float64),
        routing_statistics=objective.routing,
        cg_relative_tolerance=1e-8,
    )
    for head, group in enumerate(layout.head_to_kv_group().tolist()):
        torch.testing.assert_close(
            factors.joint_encoders[group] @ factors.payload_decoders[head],
            torch.cat(
                (
                    value_encoder[group] @ value_decoder[head],
                    torch.zeros(2, 6, dtype=torch.float64),
                )
            ),
        )


def test_fast_schedule_is_monotone_and_supports_all_u_modes() -> None:
    layout, objective, factors, *_, direct = _problem()
    for mode in ("frozen_u", "adapter_u", "full_u_final"):
        progress = []
        result = fit_s80_joint(
            layout=layout,
            objective=objective,
            validation_objective=objective,
            direct_residuals=direct,
            initial_factors=factors,
            outer_sweeps=3,
            u_mode=mode,
            lsqr_max_iterations=80,
            lsqr_relative_tolerance=1e-10,
            progress_callback=progress.append,
        )
        values = [result.initial_loss.total]
        for sweep in result.diagnostics.sweeps:
            values.extend(
                (sweep.fit_after_decoder.total, sweep.fit_after_encoders.total)
            )
            assert sweep.validation_after_encoders is not None
        values.extend(
            (
                result.diagnostics.loss_after_final_decoder.total,
                result.final_loss.total,
            )
        )
        assert all(right <= left + 1e-8 for left, right in zip(values, values[1:]))
        assert all(
            len(sweep.encoder_steps) == layout.num_key_value_heads
            for sweep in result.diagnostics.sweeps
        )
        assert len(progress) == 3
        assert result.diagnostics.validation_after_final_decoder is not None
        assert result.diagnostics.validation_after_routing_queries is not None
        expected_u_systems = 0 if mode == "frozen_u" else layout.num_attention_heads
        assert len(result.final_routing_query_steps) == expected_u_systems


def test_page_fisher_schedule_is_monotone_with_adapter_u() -> None:
    layout, objective, factors, *_, direct = _problem()
    fisher = _compact_fisher(layout, direct)
    result = fit_s80_joint(
        layout=layout,
        objective=objective,
        validation_objective=objective,
        fisher_statistics=fisher,
        validation_fisher_statistics=fisher,
        initial_factors=factors,
        routing_metric="page_fisher",
        outer_sweeps=3,
        u_mode="adapter_u",
        lsqr_max_iterations=100,
        lsqr_relative_tolerance=1e-10,
    )
    values = [result.initial_loss.total]
    for sweep in result.diagnostics.sweeps:
        values.extend((sweep.fit_after_decoder.total, sweep.fit_after_encoders.total))
    values.extend(
        (
            result.diagnostics.loss_after_final_decoder.total,
            result.final_loss.total,
        )
    )
    assert all(right <= left + 1e-8 for left, right in zip(values, values[1:]))
    assert result.routing_metric == "page_fisher"
    assert result.routing_normalizer > 0
    assert result.diagnostics.validation_after_routing_queries is not None


def test_folded_runtime_uses_the_same_frontloaded_latent() -> None:
    layout, objective, factors, *_, direct = _problem()
    result = fit_s80_joint(
        layout=layout,
        objective=objective,
        direct_residuals=direct,
        initial_factors=factors,
        lsqr_max_iterations=40,
    )
    torch.manual_seed(20260904)
    dense_v = torch.randn(6, 6, dtype=torch.float64)
    dense_v_bias = torch.randn(6, dtype=torch.float64)
    folded = fold_s80_factors(
        layout=layout,
        factors=result.factors,
        dense_v_proj_weight=dense_v,
        dense_v_proj_bias=dense_v_bias,
        output_dtype=torch.float64,
    )
    hidden = torch.randn(5, 6, dtype=torch.float64)
    post_rope_key = torch.randn(5, 2, 2, dtype=torch.float64)
    dense_value = (hidden @ dense_v.mT + dense_v_bias).reshape(5, 2, 3)
    joint = torch.cat((dense_value, post_rope_key), dim=-1)
    expected = torch.einsum("tgd,gdr->tgr", joint, result.factors.joint_encoders)
    runtime = (
        hidden @ folded.v_joint_proj_weight.mT + folded.v_joint_proj_bias
    ).reshape(5, 2, 3) + torch.einsum(
        "tgk,gkr->tgr", post_rope_key, folded.k_joint_encoder
    )
    torch.testing.assert_close(runtime, expected)
    torch.testing.assert_close(
        runtime[..., : layout.routing_rank],
        torch.einsum(
            "tgd,gdr->tgr", joint, result.factors.routing_payload_encoders
        ),
    )
