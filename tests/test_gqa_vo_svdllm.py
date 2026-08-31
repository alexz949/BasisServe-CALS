from __future__ import annotations

import torch

from basisserve.core.gqa_vo_svdllm import (
    FullOInputCovariance,
    GQAHeadwiseOInputCovariance,
    GQAIdentityOInputCovariance,
    GQAOInputCovariance,
    GQAVOLayout,
    OInputActivationSamples,
    activation_whitened_o_reconstruction_error,
    factorize_gqa_vo_activation_whitened,
    factorize_gqa_vo_headwise_activation_whitened,
    global_o_activation_whitened_errors,
    refine_gqa_vo_global_joint_linear,
    validate_projection_shapes,
)


def _small_layout(rank: int = 2) -> GQAVOLayout:
    return GQAVOLayout(
        hidden_size=6,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=3,
        rank=rank,
    )


def _random_inputs(layout: GQAVOLayout) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(17)
    o_weight = torch.randn(layout.hidden_size, layout.query_width, dtype=torch.float64)
    v_weight = torch.randn(layout.kv_width, layout.hidden_size, dtype=torch.float64)
    raw = torch.randn(
        layout.num_key_value_heads,
        layout.head_dim,
        layout.head_dim,
        dtype=torch.float64,
    )
    covariances = raw @ raw.transpose(-1, -2) + 0.25 * torch.eye(layout.head_dim)
    return o_weight, v_weight, covariances


def test_qwen3_8b_layout_widths() -> None:
    layout = GQAVOLayout(
        hidden_size=4096,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        rank=64,
    )

    assert layout.query_heads_per_kv_group == 4
    assert layout.query_width == 4096
    assert layout.kv_width == 1024
    assert layout.compressed_query_width == 2048
    assert layout.compressed_kv_width == 512
    assert layout.query_head_indices(3) == (12, 13, 14, 15)


def test_group_covariance_pools_query_heads_within_kv_group() -> None:
    layout = _small_layout()
    activations = torch.arange(2 * 4 * 3, dtype=torch.float64).reshape(2, 4, 3)
    collector = GQAOInputCovariance(layout)
    collector.update(activations.reshape(1, 2, layout.query_width))

    expected = []
    for group_index in range(layout.num_key_value_heads):
        start = group_index * layout.query_heads_per_kv_group
        group = activations[:, start : start + layout.query_heads_per_kv_group, :].reshape(-1, 3)
        expected.append(group.T @ group / group.shape[0])

    torch.testing.assert_close(collector.covariances(), torch.stack(expected))
    assert collector.rows.tolist() == [4, 4]


def test_identity_covariance_is_exact_and_has_no_activation_rows() -> None:
    layout = _small_layout()
    collector = GQAIdentityOInputCovariance(layout)

    expected = torch.eye(layout.head_dim, dtype=torch.float64).expand(
        layout.num_key_value_heads,
        -1,
        -1,
    )
    torch.testing.assert_close(collector.covariances(), expected)
    assert collector.rows is None


def test_identity_metric_matches_direct_group_svd() -> None:
    layout = _small_layout(rank=2)
    o_weight, v_weight, _ = _random_inputs(layout)
    covariances = GQAIdentityOInputCovariance(layout).covariances()
    factors = factorize_gqa_vo_activation_whitened(
        layout=layout,
        o_weight=o_weight,
        v_weight=v_weight,
        covariances=covariances,
        act_damp=0.0,
        orthonormalize_a=True,
        output_dtype=torch.float64,
        work_dtype=torch.float64,
    )

    for group in factors.groups:
        weight_rows = []
        for head_index in group.query_head_indices:
            start = head_index * layout.head_dim
            weight_rows.append(o_weight[:, start : start + layout.head_dim].T)
        weight_cat = torch.cat(weight_rows, dim=1)
        u, singular_values, vh = torch.linalg.svd(weight_cat, full_matrices=False)
        expected = (u[:, : layout.rank] * singular_values[: layout.rank]) @ vh[: layout.rank]
        actual = torch.cat(
            [decoder.T for decoder in group.D_pt_list], dim=1
        )
        actual = group.A_pt.T @ actual
        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
        assert abs(group.relative_whitened_error - group.relative_frobenius_error) < 1e-12


def test_headwise_covariance_preserves_query_head_metrics() -> None:
    layout = _small_layout()
    activations = torch.arange(2 * 4 * 3, dtype=torch.float64).reshape(2, 4, 3)
    collector = GQAHeadwiseOInputCovariance(layout)
    collector.update(activations.reshape(1, 2, layout.query_width))

    expected = []
    for group_index in range(layout.num_key_value_heads):
        group = []
        for head_index in layout.query_head_indices(group_index):
            rows = activations[:, head_index, :]
            group.append(rows.T @ rows / rows.shape[0])
        expected.append(torch.stack(group))

    torch.testing.assert_close(collector.covariances(), torch.stack(expected))
    assert collector.rows.tolist() == [[2, 2], [2, 2]]


def test_activation_sampler_bounds_and_spreads_rows() -> None:
    collector = OInputActivationSamples(3, max_rows=4, rows_per_update=2)
    first = torch.arange(18, dtype=torch.float32).reshape(2, 3, 3)
    second = first + 100
    third = first + 200
    collector.update(first)
    collector.update(second)
    collector.update(third)

    samples = collector.samples().float()
    assert samples.shape == (4, 3)
    torch.testing.assert_close(samples[0], first.reshape(-1, 3)[0])
    torch.testing.assert_close(samples[1], first.reshape(-1, 3)[-1])
    torch.testing.assert_close(samples[2], second.reshape(-1, 3)[0])
    torch.testing.assert_close(samples[3], second.reshape(-1, 3)[-1])
    assert collector.rows == 4


def test_full_covariance_keeps_cross_head_terms() -> None:
    layout = _small_layout()
    activations = torch.arange(3 * layout.query_width, dtype=torch.float32).reshape(
        1,
        3,
        layout.query_width,
    )
    collector = FullOInputCovariance(layout.query_width)
    collector.update(activations)

    rows = activations.reshape(-1, layout.query_width)
    expected = rows.T @ rows / rows.shape[0]
    torch.testing.assert_close(collector.covariance(), expected)
    assert collector.rows == 3


def test_full_rank_whitened_factorization_is_exact() -> None:
    layout = _small_layout(rank=3)
    o_weight, v_weight, covariances = _random_inputs(layout)
    factors = factorize_gqa_vo_activation_whitened(
        layout=layout,
        o_weight=o_weight,
        v_weight=v_weight,
        covariances=covariances,
        output_dtype=torch.float64,
        work_dtype=torch.float64,
    )

    for group in factors.groups:
        for head_index, decoder in zip(group.query_head_indices, group.D_pt_list, strict=True):
            start = head_index * layout.head_dim
            expected = o_weight[:, start : start + layout.head_dim]
            reconstructed = decoder @ group.A_pt
            torch.testing.assert_close(reconstructed, expected, rtol=1e-10, atol=1e-10)
        assert group.relative_frobenius_error < 1e-10
        assert group.relative_whitened_error < 1e-10


def test_full_rank_headwise_factorization_is_exact() -> None:
    layout = _small_layout(rank=3)
    o_weight, v_weight, _ = _random_inputs(layout)
    torch.manual_seed(23)
    raw = torch.randn(
        layout.num_key_value_heads,
        layout.query_heads_per_kv_group,
        layout.head_dim,
        layout.head_dim,
        dtype=torch.float64,
    )
    covariances = raw @ raw.transpose(-1, -2) + 0.25 * torch.eye(layout.head_dim)
    factors = factorize_gqa_vo_headwise_activation_whitened(
        layout=layout,
        o_weight=o_weight,
        v_weight=v_weight,
        covariances=covariances,
        optimization_steps=5,
        output_dtype=torch.float64,
        work_dtype=torch.float64,
    )

    for group in factors.groups:
        for head_index, decoder in zip(group.query_head_indices, group.D_pt_list, strict=True):
            start = head_index * layout.head_dim
            expected = o_weight[:, start : start + layout.head_dim]
            reconstructed = decoder @ group.A_pt
            torch.testing.assert_close(reconstructed, expected, rtol=1e-9, atol=1e-9)
        assert group.relative_whitened_error < 1e-9


def test_headwise_optimization_monotonically_improves_shared_basis() -> None:
    layout = GQAVOLayout(
        hidden_size=7,
        num_attention_heads=3,
        num_key_value_heads=1,
        head_dim=4,
        rank=2,
    )
    torch.manual_seed(101)
    o_weight = torch.randn(layout.hidden_size, layout.query_width, dtype=torch.float64)
    v_weight = torch.randn(layout.kv_width, layout.hidden_size, dtype=torch.float64)
    raw = torch.randn(
        layout.num_key_value_heads,
        layout.query_heads_per_kv_group,
        layout.head_dim,
        layout.head_dim,
        dtype=torch.float64,
    )
    scales = torch.tensor([0.2, 2.0, 7.0], dtype=torch.float64).view(1, 3, 1, 1)
    covariances = scales * (raw @ raw.transpose(-1, -2)) + 0.1 * torch.eye(layout.head_dim)
    factors = factorize_gqa_vo_headwise_activation_whitened(
        layout=layout,
        o_weight=o_weight,
        v_weight=v_weight,
        covariances=covariances,
        act_damp=1e-5,
        optimization_steps=30,
        step_size=0.5,
        tolerance=1e-12,
        output_dtype=torch.float64,
        work_dtype=torch.float64,
    )

    group = factors.groups[0]
    assert group.initial_relative_whitened_error is not None
    assert group.optimization_steps > 0
    assert group.relative_whitened_error < group.initial_relative_whitened_error


def test_headwise_factorization_keeps_one_shared_latent_per_kv_group() -> None:
    layout = _small_layout(rank=2)
    o_weight, v_weight, _ = _random_inputs(layout)
    torch.manual_seed(109)
    raw = torch.randn(
        layout.num_key_value_heads,
        layout.query_heads_per_kv_group,
        layout.head_dim,
        layout.head_dim,
        dtype=torch.float64,
    )
    covariances = raw @ raw.transpose(-1, -2) + 0.1 * torch.eye(layout.head_dim)
    factors = factorize_gqa_vo_headwise_activation_whitened(
        layout=layout,
        o_weight=o_weight,
        v_weight=v_weight,
        covariances=covariances,
        optimization_steps=5,
        output_dtype=torch.float64,
        work_dtype=torch.float64,
    )

    assert factors.v_proj_compressed_weight.shape == (
        layout.num_key_value_heads * layout.rank,
        layout.hidden_size,
    )
    assert factors.o_decoder_weight.shape == (
        layout.hidden_size,
        layout.num_attention_heads * layout.rank,
    )
    for group in factors.groups:
        assert group.A_pt.shape == (layout.rank, layout.head_dim)
        assert len(group.D_pt_list) == layout.query_heads_per_kv_group
        assert all(
            decoder.shape == (layout.hidden_size, layout.rank)
            for decoder in group.D_pt_list
        )


def test_global_joint_linear_refinement_reduces_full_output_error() -> None:
    layout = GQAVOLayout(
        hidden_size=8,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=3,
        rank=2,
    )
    torch.manual_seed(131)
    o_weight = torch.randn(layout.hidden_size, layout.query_width, dtype=torch.float64)
    v_weight = torch.randn(layout.kv_width, layout.hidden_size, dtype=torch.float64)
    samples = torch.randn(96, layout.query_width, dtype=torch.float64)
    head_rows = samples.reshape(
        -1,
        layout.num_key_value_heads,
        layout.query_heads_per_kv_group,
        layout.head_dim,
    )
    covariances = torch.einsum("ngqd,ngqe->gqde", head_rows, head_rows) / samples.shape[0]
    initial = factorize_gqa_vo_headwise_activation_whitened(
        layout=layout,
        o_weight=o_weight,
        v_weight=v_weight,
        covariances=covariances,
        optimization_steps=5,
        output_dtype=torch.float64,
        work_dtype=torch.float64,
    )
    with torch.inference_mode():
        refined = refine_gqa_vo_global_joint_linear(
            initial_factors=initial,
            o_weight=o_weight,
            v_weight=v_weight,
            activation_samples=samples,
            steps=80,
            batch_size=32,
            basis_learning_rate=2e-2,
            decoder_learning_rate=5e-3,
            eval_interval=5,
            output_dtype=torch.float64,
            factor_device="cpu",
            work_dtype=torch.float64,
        )

    assert refined.joint_initial_relative_output_error is not None
    assert refined.joint_final_relative_output_error is not None
    assert (
        refined.joint_final_relative_output_error
        < refined.joint_initial_relative_output_error
    )
    assert refined.v_proj_compressed_weight.shape == (
        layout.compressed_kv_width,
        layout.hidden_size,
    )
    assert refined.o_decoder_weight.shape == (
        layout.hidden_size,
        layout.compressed_query_width,
    )
    for group in refined.groups:
        eye = torch.eye(layout.rank, dtype=torch.float64)
        torch.testing.assert_close(
            group.A_pt @ group.A_pt.transpose(0, 1),
            eye,
            rtol=1e-8,
            atol=1e-8,
        )


def test_truncated_factor_shapes_and_head_order() -> None:
    layout = _small_layout(rank=2)
    o_weight, v_weight, covariances = _random_inputs(layout)
    factors = factorize_gqa_vo_activation_whitened(
        layout=layout,
        o_weight=o_weight,
        v_weight=v_weight,
        covariances=covariances,
        output_dtype=torch.float32,
    )

    assert factors.v_proj_compressed_weight.shape == (layout.compressed_kv_width, layout.hidden_size)
    assert factors.o_decoder_weight.shape == (layout.hidden_size, layout.compressed_query_width)
    assert [group.query_head_indices for group in factors.groups] == [(0, 1), (2, 3)]
    for group in factors.groups:
        assert group.A_pt.shape == (layout.rank, layout.head_dim)
        assert group.Wv_compressed_pt.shape == (layout.rank, layout.hidden_size)
        assert len(group.D_pt_list) == layout.query_heads_per_kv_group
        assert all(decoder.shape == (layout.hidden_size, layout.rank) for decoder in group.D_pt_list)


def test_folding_basis_into_v_adds_no_error() -> None:
    layout = _small_layout(rank=2)
    o_weight, v_weight, covariances = _random_inputs(layout)
    factors = factorize_gqa_vo_activation_whitened(
        layout=layout,
        o_weight=o_weight,
        v_weight=v_weight,
        covariances=covariances,
        output_dtype=torch.float64,
        work_dtype=torch.float64,
    )
    torch.manual_seed(29)
    hidden = torch.randn(7, layout.hidden_size, dtype=torch.float64)
    attention_probabilities = torch.softmax(torch.randn(2, 7, dtype=torch.float64), dim=-1)

    for group in factors.groups:
        start = group.group_index * layout.head_dim
        dense_v = hidden @ v_weight[start : start + layout.head_dim, :].T
        folded_v = hidden @ group.Wv_compressed_pt.T
        for decoder in group.D_pt_list:
            no_fold = (attention_probabilities @ dense_v) @ group.A_pt.T @ decoder.T
            folded = (attention_probabilities @ folded_v) @ decoder.T
            torch.testing.assert_close(folded, no_fold, rtol=1e-10, atol=1e-10)


def test_orthonormalize_a_preserves_product_and_conditions_basis() -> None:
    layout = _small_layout(rank=2)
    o_weight, v_weight, covariances = _random_inputs(layout)
    plain = factorize_gqa_vo_activation_whitened(
        layout=layout,
        o_weight=o_weight,
        v_weight=v_weight,
        covariances=covariances,
        output_dtype=torch.float64,
        work_dtype=torch.float64,
    )
    orthogonal = factorize_gqa_vo_activation_whitened(
        layout=layout,
        o_weight=o_weight,
        v_weight=v_weight,
        covariances=covariances,
        orthonormalize_a=True,
        output_dtype=torch.float64,
        work_dtype=torch.float64,
    )

    eye = torch.eye(layout.rank, dtype=torch.float64)
    for plain_group, ortho_group in zip(plain.groups, orthogonal.groups, strict=True):
        torch.testing.assert_close(ortho_group.A_pt @ ortho_group.A_pt.T, eye, rtol=1e-10, atol=1e-10)
        for plain_decoder, ortho_decoder in zip(
            plain_group.D_pt_list,
            ortho_group.D_pt_list,
            strict=True,
        ):
            torch.testing.assert_close(
                plain_decoder @ plain_group.A_pt,
                ortho_decoder @ ortho_group.A_pt,
                rtol=1e-10,
                atol=1e-10,
            )


def test_projection_shape_validation_accepts_non_square_qwen_geometry() -> None:
    layout = GQAVOLayout(
        hidden_size=10,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=3,
        rank=2,
    )
    validate_projection_shapes(
        layout,
        q_weight=torch.empty(12, 10),
        k_weight=torch.empty(6, 10),
        v_weight=torch.empty(6, 10),
        o_weight=torch.empty(10, 12),
    )


def test_global_whitened_spectrum_and_reconstruction_use_same_metric() -> None:
    weight = torch.diag(torch.tensor([4.0, 3.0, 2.0, 1.0], dtype=torch.float64))
    covariance = torch.eye(4, dtype=torch.float64)
    results = global_o_activation_whitened_errors(
        o_weight=weight,
        covariance=covariance,
        ranks=[1, 2],
        act_damp=0.0,
        work_dtype=torch.float64,
    )

    expected_rank1 = (torch.tensor(14.0 / 30.0, dtype=torch.float64)).sqrt()
    expected_rank2 = (torch.tensor(5.0 / 30.0, dtype=torch.float64)).sqrt()
    assert abs(results[0].relative_whitened_error - float(expected_rank1)) < 1e-12
    assert abs(results[1].relative_whitened_error - float(expected_rank2)) < 1e-12

    rank2_reconstruction = torch.diag(
        torch.tensor([4.0, 3.0, 0.0, 0.0], dtype=torch.float64)
    )
    measured = activation_whitened_o_reconstruction_error(
        o_weight=weight,
        reconstructed_o_weight=rank2_reconstruction,
        covariance=covariance,
        act_damp=0.0,
        work_dtype=torch.float64,
    )
    assert abs(measured - results[1].relative_whitened_error) < 1e-12
