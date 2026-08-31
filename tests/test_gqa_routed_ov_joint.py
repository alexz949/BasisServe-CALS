from __future__ import annotations

import torch

from basisserve.calibration.gqa_routed_ov_stats import (
    RoutedOVAccumulator,
    covariance_block_energy_diagnostics,
    flatten_covariance_blocks,
    unflatten_covariance,
)
from basisserve.core.gqa_routed_ov_joint import (
    add_isotropic_product_prior,
    block_matrix_product,
    combine_quadratics,
    conjugate_gradient_matrix,
    covariance_minimum_eigenvalue,
    encoder_group_half_gradient,
    encoder_hessian_vector_product,
    evaluate_quadratic,
    extract_value_coordinate_factors,
    fit_routed_ov_joint,
    fold_ragged_routed_ov_factors,
    fold_routed_ov_factors,
    function_prior_covariance,
    gauge_canonicalize,
    head_products,
    initialize_group_pooled_routed_svd,
    mask_routed_covariance,
    quadratic_from_target,
    solve_free_decoder,
    solve_free_decoder_ragged,
    trace_normalize_covariance,
)
from basisserve.core.gqa_vo_svdllm import GQAVOLayout


def _layout(rank: int = 2) -> GQAVOLayout:
    return GQAVOLayout(
        hidden_size=7,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=3,
        rank=rank,
    )


def _mapping(layout: GQAVOLayout) -> torch.Tensor:
    return (
        torch.arange(layout.num_attention_heads)
        // layout.query_heads_per_kv_group
    )


def _problem(
    layout: GQAVOLayout,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(20260718)
    rows = torch.randn(37, layout.query_width, dtype=torch.float64)
    flat = rows.T @ rows / rows.shape[0]
    blocks = unflatten_covariance(
        flat,
        num_query_heads=layout.num_attention_heads,
        head_dim=layout.head_dim,
    )
    target = torch.randn(
        layout.num_attention_heads,
        layout.head_dim,
        layout.hidden_size,
        dtype=torch.float64,
    )
    A = torch.randn(
        layout.num_key_value_heads,
        layout.head_dim,
        layout.rank,
        dtype=torch.float64,
    )
    D = torch.randn(
        layout.num_attention_heads,
        layout.rank,
        layout.hidden_size,
        dtype=torch.float64,
    )
    A, D, _ = gauge_canonicalize(A, D, _mapping(layout))
    return blocks, target, A, D


def test_dense_head_stacking_and_covariance_blocks() -> None:
    layout = _layout()
    torch.manual_seed(1)
    by_head = torch.randn(11, 4, 3, dtype=torch.float64)
    weights = torch.randn(4, 3, 7, dtype=torch.float64)
    expected = sum(by_head[:, head] @ weights[head] for head in range(4))
    actual = by_head.reshape(11, -1) @ weights.reshape(-1, 7)
    torch.testing.assert_close(actual, expected)

    flat = by_head.reshape(11, -1).T @ by_head.reshape(11, -1)
    blocks = unflatten_covariance(
        flat,
        num_query_heads=4,
        head_dim=3,
    )
    torch.testing.assert_close(flatten_covariance_blocks(blocks), flat)
    for h in range(4):
        for k in range(4):
            torch.testing.assert_close(
                blocks[h, k],
                by_head[:, h].T @ by_head[:, k],
            )


def test_routed_accumulator_matches_direct_and_masks() -> None:
    layout = _layout()
    torch.manual_seed(2)
    routed = torch.randn(2, 5, layout.query_width, dtype=torch.float64)
    weight = torch.randn(layout.query_width, layout.hidden_size, dtype=torch.float64)
    output = routed @ weight
    mask = torch.ones(2, 5, dtype=torch.bool)
    mask[0, -2:] = False
    accumulator = RoutedOVAccumulator(
        num_query_heads=layout.num_attention_heads,
        head_dim=layout.head_dim,
        accumulation_dtype=torch.float64,
    )
    accumulator.update(routed, output, valid_mask=mask)
    stats = accumulator.finalize()
    selected = routed.reshape(-1, layout.query_width)[mask.reshape(-1)]
    selected_output = output.reshape(-1, layout.hidden_size)[mask.reshape(-1)]
    torch.testing.assert_close(
        stats.flat_covariance(),
        selected.T @ selected / selected.shape[0],
    )
    assert abs(
        stats.dense_output_energy
        - float(selected_output.square().sum() / selected.shape[0])
    ) < 1e-10

    diagonal = mask_routed_covariance(
        stats.covariance_blocks,
        head_to_kv_group=_mapping(layout),
        mode="diagonal",
    )
    within = mask_routed_covariance(
        stats.covariance_blocks,
        head_to_kv_group=_mapping(layout),
        mode="within_group",
    )
    for h in range(4):
        for k in range(4):
            if h != k:
                assert torch.count_nonzero(diagonal[h, k]) == 0
            if _mapping(layout)[h] != _mapping(layout)[k]:
                assert torch.count_nonzero(within[h, k]) == 0
    assert covariance_minimum_eigenvalue(diagonal) >= -1e-10
    assert covariance_minimum_eigenvalue(within) >= -1e-10
    diagnostics = covariance_block_energy_diagnostics(
        stats.covariance_blocks,
        _mapping(layout),
    )
    assert abs(
        diagnostics["diagonal_energy"]
        + diagnostics["within_group_offdiagonal_energy"]
        + diagnostics["cross_group_energy"]
        - diagnostics["total_energy"]
    ) < 1e-9


def test_quadratic_objective_matches_direct_residual_and_mixture() -> None:
    layout = _layout()
    covariance, target, A, D = _problem(layout)
    mapping = _mapping(layout)
    objective = quadratic_from_target(
        covariance=covariance,
        target=target,
        name="routed",
        trace_normalize=False,
    )
    products = head_products(A, D, mapping)
    residual = target - products
    expected = torch.sum(residual * block_matrix_product(covariance, residual))
    actual = evaluate_quadratic(objective, A, D, mapping)
    assert abs(actual - float(expected)) < 1e-9

    prior_target = 0.75 * target
    prior = quadratic_from_target(
        covariance=covariance,
        target=prior_target,
        name="prior",
        trace_normalize=False,
    )
    mixed = combine_quadratics(prior, objective, right_weight=0.3)
    expected_mixture = 0.7 * torch.sum(
        (prior_target - products)
        * block_matrix_product(covariance, prior_target - products)
    ) + 0.3 * expected
    assert abs(evaluate_quadratic(mixed, A, D, mapping) - float(expected_mixture)) < 1e-9


def test_trace_normalization_and_function_prior() -> None:
    layout = _layout()
    torch.manual_seed(3)
    metrics = []
    for _ in range(layout.num_key_value_heads):
        raw = torch.randn(layout.head_dim, layout.head_dim, dtype=torch.float64)
        metrics.append(raw @ raw.T + torch.eye(layout.head_dim, dtype=torch.float64))
    metrics = torch.stack(metrics)
    prior = function_prior_covariance(
        metrics,
        head_to_kv_group=_mapping(layout),
    )
    normalized, _ = trace_normalize_covariance(prior)
    trace = sum(torch.trace(normalized[h, h]) for h in range(4))
    assert abs(float(trace) - layout.query_width) < 1e-10
    for h in range(4):
        torch.testing.assert_close(prior[h, h], metrics[_mapping(layout)[h]])


def test_free_decoder_matches_explicit_normal_equations() -> None:
    layout = _layout()
    covariance, target, A, _ = _problem(layout)
    mapping = _mapping(layout)
    objective = quadratic_from_target(
        covariance=covariance,
        target=target,
        name="full",
        trace_normalize=False,
    )
    D, diagnostics = solve_free_decoder(
        objective=objective,
        A_unique=A,
        head_to_kv_group=mapping,
        coupling_mode="full_layer",
        relative_jitter=0.0,
    )
    by_head = A.index_select(0, mapping)
    E = torch.zeros(
        layout.query_width,
        layout.compressed_query_width,
        dtype=torch.float64,
    )
    for h in range(layout.num_attention_heads):
        E[
            h * layout.head_dim : (h + 1) * layout.head_dim,
            h * layout.rank : (h + 1) * layout.rank,
        ] = by_head[h]
    flat_covariance = flatten_covariance_blocks(covariance)
    flat_target = target.reshape(layout.query_width, layout.hidden_size)
    jitter = diagnostics.absolute_jitters[0]
    expected = torch.linalg.solve(
        E.T @ flat_covariance @ E
        + jitter * torch.eye(layout.compressed_query_width, dtype=torch.float64),
        E.T @ flat_covariance @ flat_target,
    )
    torch.testing.assert_close(
        D.reshape(layout.compressed_query_width, layout.hidden_size),
        expected,
        rtol=1e-9,
        atol=1e-9,
    )


def test_isotropic_product_prior_matches_explicit_product_penalty() -> None:
    layout = _layout()
    covariance, target, A, D0 = _problem(layout)
    mapping = _mapping(layout)
    objective = quadratic_from_target(
        covariance=covariance,
        target=target,
        name="routed",
        trace_normalize=False,
    )
    center = head_products(A, D0, mapping)
    weight = 0.037
    combined = add_isotropic_product_prior(
        objective,
        center_product=center,
        absolute_weight=weight,
        name="routed_plus_product_prior",
    )
    torch.manual_seed(20260724)
    current_A = A + 0.1 * torch.randn_like(A)
    current_D = D0 + 0.1 * torch.randn_like(D0)
    product = head_products(current_A, current_D, mapping)
    expected = evaluate_quadratic(
        objective,
        current_A,
        current_D,
        mapping,
    ) + weight * float(torch.sum((product - center).square()))
    actual = evaluate_quadratic(
        combined,
        current_A,
        current_D,
        mapping,
    )
    torch.testing.assert_close(
        torch.tensor(actual),
        torch.tensor(expected),
        rtol=1e-10,
        atol=1e-10,
    )


def test_product_prior_fixed_orthonormal_encoder_is_decoder_ridge() -> None:
    layout = _layout()
    covariance, target, A, D0 = _problem(layout)
    mapping = _mapping(layout)
    objective = quadratic_from_target(
        covariance=covariance,
        target=target,
        name="routed",
        trace_normalize=False,
    )
    weight = 0.051
    combined = add_isotropic_product_prior(
        objective,
        center_product=head_products(A, D0, mapping),
        absolute_weight=weight,
    )
    actual, diagnostics = solve_free_decoder(
        objective=combined,
        A_unique=A,
        head_to_kv_group=mapping,
        coupling_mode="full_layer",
    )
    by_head = A.index_select(0, mapping)
    structured = torch.zeros(
        layout.query_width,
        layout.compressed_query_width,
        dtype=torch.float64,
    )
    for head in range(layout.num_attention_heads):
        structured[
            head * layout.head_dim : (head + 1) * layout.head_dim,
            head * layout.rank : (head + 1) * layout.rank,
        ] = by_head[head]
    flat_covariance = flatten_covariance_blocks(covariance)
    flat_target = target.reshape(layout.query_width, layout.hidden_size)
    hessian = structured.T @ flat_covariance @ structured
    rhs = structured.T @ flat_covariance @ flat_target
    center = D0.reshape(
        layout.compressed_query_width,
        layout.hidden_size,
    )
    jitter = diagnostics.absolute_jitters[0]
    expected = torch.linalg.solve(
        hessian
        + (weight + jitter)
        * torch.eye(layout.compressed_query_width, dtype=torch.float64),
        rhs + weight * center,
    )
    torch.testing.assert_close(
        actual.reshape(layout.compressed_query_width, layout.hidden_size),
        expected,
        rtol=1e-9,
        atol=1e-9,
    )


def test_product_prior_one_sweep_cg16_reaches_final_redecoder() -> None:
    layout = _layout()
    covariance, target, A, D0 = _problem(layout)
    mapping = _mapping(layout)
    routed = quadratic_from_target(
        covariance=covariance,
        target=target,
        name="routed",
        trace_normalize=False,
    )
    combined = add_isotropic_product_prior(
        routed,
        center_product=head_products(A, D0, mapping),
        absolute_weight=0.03,
        name="stage_b",
    )
    result = fit_routed_ov_joint(
        objective=combined,
        initial_A=A,
        initial_D=D0,
        head_to_kv_group=mapping,
        coupling_mode="full_layer",
        maximum_sweeps=1,
        minimum_sweeps=1,
        relative_objective_tolerance=0.0,
        patience=1,
        encoder_relative_damping=1e-8,
        cg_relative_tolerance=1e-8,
        cg_max_iterations=16,
        cg_fixed_iterations=True,
        maximum_backtracks=10,
        checkpoint_callback=lambda *_args: None,
        final_decoder_solve=True,
        verify_encoder_step_objective=True,
        work_dtype=torch.float64,
        work_device="cpu",
    )
    assert len(result.sweeps) == 1
    assert result.final_decoder is not None
    assert result.checkpoints[-1].boundary == "after_redecoder"
    assert result.checkpoints[-1].sweep == 1
    assert result.final_loss <= result.decoder_only_loss + 1e-10
    assert result.final_loss <= result.sweeps[0].loss_after_encoders + 1e-10
    assert all(
        step.cg.fixed_iterations
        for step in result.sweeps[0].encoder_steps
        if step.cg.initial_residual_norm != 0.0
    )


def test_ragged_free_decoder_matches_explicit_normal_equations() -> None:
    layout = _layout(rank=3)
    covariance, target, _, _ = _problem(layout)
    mapping = _mapping(layout)
    ranks = (1, 2)
    torch.manual_seed(20260721)
    A = torch.zeros(2, 3, 3, dtype=torch.float64)
    A[0, :, :1] = torch.randn(3, 1, dtype=torch.float64)
    A[1, :, :2] = torch.randn(3, 2, dtype=torch.float64)
    objective = quadratic_from_target(
        covariance=covariance,
        target=target,
        name="ragged",
        trace_normalize=False,
    )
    actual, diagnostics = solve_free_decoder_ragged(
        objective=objective,
        A_unique=A,
        head_to_kv_group=mapping,
        group_ranks=ranks,
    )

    head_ranks = [ranks[int(group)] for group in mapping]
    offsets = [0]
    for rank in head_ranks:
        offsets.append(offsets[-1] + rank)
    E = torch.zeros(layout.query_width, offsets[-1], dtype=torch.float64)
    for head, rank in enumerate(head_ranks):
        E[
            head * layout.head_dim : (head + 1) * layout.head_dim,
            offsets[head] : offsets[head + 1],
        ] = A[int(mapping[head]), :, :rank]
    flat_covariance = flatten_covariance_blocks(covariance)
    flat_target = target.reshape(layout.query_width, layout.hidden_size)
    jitter = diagnostics.absolute_jitters[0]
    expected = torch.linalg.solve(
        E.T @ flat_covariance @ E
        + jitter * torch.eye(offsets[-1], dtype=torch.float64),
        E.T @ flat_covariance @ flat_target,
    )
    packed = torch.cat(
        [actual[head, :rank] for head, rank in enumerate(head_ranks)],
        dim=0,
    )
    torch.testing.assert_close(packed, expected, rtol=1e-9, atol=1e-9)
    assert diagnostics.matrix_dimensions == (offsets[-1],)
    assert diagnostics.relative_residuals[0] < 1e-9
    assert torch.count_nonzero(actual[0, 1:]) == 0
    assert torch.count_nonzero(actual[2, 2:]) == 0


def test_uniform_ragged_decoder_regresses_to_uniform_solver() -> None:
    layout = _layout(rank=2)
    covariance, target, A, _ = _problem(layout)
    mapping = _mapping(layout)
    objective = quadratic_from_target(
        covariance=covariance,
        target=target,
        name="uniform_regression",
        trace_normalize=False,
    )
    uniform, _ = solve_free_decoder(
        objective=objective,
        A_unique=A,
        head_to_kv_group=mapping,
        coupling_mode="full_layer",
    )
    ragged, _ = solve_free_decoder_ragged(
        objective=objective,
        A_unique=A,
        head_to_kv_group=mapping,
        group_ranks=(2, 2),
    )
    torch.testing.assert_close(ragged, uniform, rtol=1e-10, atol=1e-10)


def test_group_pooled_routed_svd_solves_damped_surrogate() -> None:
    layout = _layout(rank=2)
    covariance, target, _, _ = _problem(layout)
    mapping = _mapping(layout)
    ranks = (1, 2)
    ridge = 1e-6
    initialized = initialize_group_pooled_routed_svd(
        covariance=covariance,
        target=target,
        head_to_kv_group=mapping,
        group_ranks=ranks,
        covariance_ridge=ridge,
    )
    assert tuple(initialized.A_unique.shape) == (2, 3, 2)
    assert initialized.group_ranks == ranks
    assert torch.count_nonzero(initialized.A_unique[0, :, 1:]) == 0

    for group, rank in enumerate(ranks):
        heads = torch.nonzero(mapping == group, as_tuple=False).flatten()
        pooled = covariance[heads, heads].mean(dim=0)
        pooled = 0.5 * (pooled + pooled.T)
        scale = float(torch.trace(pooled)) / layout.head_dim
        damped = pooled + ridge * scale * torch.eye(
            layout.head_dim,
            dtype=torch.float64,
        )
        cholesky = torch.linalg.cholesky(damped)
        concatenated = target.index_select(0, heads).permute(1, 0, 2).reshape(
            layout.head_dim,
            -1,
        )
        whitened = cholesky.T @ concatenated
        singular_values = torch.linalg.svdvals(whitened)
        expected_loss = float(singular_values[rank:].square().sum())

        encoder = initialized.A_unique[group, :, :rank]
        decoder = torch.linalg.solve(
            encoder.T @ damped @ encoder,
            encoder.T @ damped @ concatenated,
        )
        residual = concatenated - encoder @ decoder
        actual_loss = float(torch.sum(residual * (damped @ residual)))
        assert abs(actual_loss - expected_loss) < 1e-8
        torch.testing.assert_close(
            encoder.T @ encoder,
            torch.eye(rank, dtype=torch.float64),
            rtol=1e-10,
            atol=1e-10,
        )
        diagnostics = initialized.groups[group]
        assert diagnostics.head_indices == tuple(heads.tolist())
        assert diagnostics.effective_relative_ridge == ridge
        assert diagnostics.weighted_tail_energy_fraction >= 0
        assert diagnostics.metric_orthogonality_error < 1e-10
        assert diagnostics.euclidean_orthogonality_error < 1e-10


def test_group_pooled_routed_svd_and_cg16_do_not_use_backward(
    monkeypatch,
) -> None:
    layout = _layout(rank=2)
    covariance, target, _, _ = _problem(layout)
    mapping = _mapping(layout)

    def forbidden(*args, **kwargs):
        raise AssertionError("current-student backward is forbidden")

    monkeypatch.setattr(torch.Tensor, "backward", forbidden)
    monkeypatch.setattr(torch.autograd, "grad", forbidden)
    initialized = initialize_group_pooled_routed_svd(
        covariance=covariance,
        target=target,
        head_to_kv_group=mapping,
        group_ranks=(layout.rank,) * layout.num_key_value_heads,
        covariance_ridge=1e-7,
    )
    objective = quadratic_from_target(
        covariance=covariance,
        target=target,
        name="a3_independent_routed",
        trace_normalize=False,
    )
    initial_decoder, _ = solve_free_decoder(
        objective=objective,
        A_unique=initialized.A_unique,
        head_to_kv_group=mapping,
        coupling_mode="full_layer",
    )
    result = fit_routed_ov_joint(
        objective=objective,
        initial_A=initialized.A_unique,
        initial_D=initial_decoder,
        head_to_kv_group=mapping,
        coupling_mode="full_layer",
        maximum_sweeps=1,
        minimum_sweeps=1,
        cg_relative_tolerance=1e-8,
        cg_max_iterations=16,
        cg_fixed_iterations=True,
        final_decoder_solve=True,
        verify_encoder_step_objective=True,
        work_dtype=torch.float64,
    )
    assert result.final_loss <= result.decoder_only_loss + 1e-9
    assert result.final_decoder is not None


def test_encoder_gradient_and_hessian_vector_product_match_autograd() -> None:
    layout = _layout()
    covariance, target, A, D = _problem(layout)
    mapping = _mapping(layout)
    objective = quadratic_from_target(
        covariance=covariance,
        target=target,
        name="full",
        trace_normalize=False,
    )
    group = 0
    heads = tuple(torch.nonzero(mapping == group, as_tuple=False).flatten().tolist())
    variable = A[group].clone().requires_grad_(True)
    assembled = A.clone()
    assembled[group] = variable
    products = head_products(assembled, D, mapping)
    residual = target - products
    loss = torch.sum(residual * block_matrix_product(covariance, residual))
    loss.backward()
    assert variable.grad is not None

    epsilon = 1e-5
    direction = torch.randn_like(variable)
    plus = variable.detach() + epsilon * direction
    minus = variable.detach() - epsilon * direction

    def gradient_at(value: torch.Tensor) -> torch.Tensor:
        current = A.clone()
        current[group] = value
        current.requires_grad_(True)
        current_products = head_products(current, D, mapping)
        current_residual = target - current_products
        current_loss = torch.sum(
            current_residual * block_matrix_product(covariance, current_residual)
        )
        return torch.autograd.grad(current_loss, current)[0][group]

    finite_difference = (gradient_at(plus) - gradient_at(minus)) / (2 * epsilon)
    hessian_direction = encoder_hessian_vector_product(
        covariance=covariance,
        D_heads=D,
        head_indices=heads,
        delta=direction,
    )
    torch.testing.assert_close(
        finite_difference,
        2.0 * hessian_direction,
        rtol=2e-5,
        atol=2e-5,
    )
    other = torch.randn_like(direction)
    h_other = encoder_hessian_vector_product(
        covariance=covariance,
        D_heads=D,
        head_indices=heads,
        delta=other,
    )
    assert abs(
        float(torch.sum(direction * h_other))
        - float(torch.sum(hessian_direction * other))
    ) < 1e-9
    assert float(torch.sum(direction * hessian_direction)) >= -1e-9


def test_cg_coordinate_solution_matches_explicit_hessian() -> None:
    layout = _layout()
    covariance, _, _, D = _problem(layout)
    mapping = _mapping(layout)
    heads = tuple(torch.nonzero(mapping == 0, as_tuple=False).flatten().tolist())
    width = layout.head_dim * layout.rank

    def operator(value: torch.Tensor) -> torch.Tensor:
        return encoder_hessian_vector_product(
            covariance=covariance,
            D_heads=D,
            head_indices=heads,
            delta=value,
        )

    basis = torch.eye(width, dtype=torch.float64).reshape(
        width,
        layout.head_dim,
        layout.rank,
    )
    explicit = torch.stack([operator(item).reshape(-1) for item in basis], dim=1)
    rhs = torch.randn(layout.head_dim, layout.rank, dtype=torch.float64)
    damping = 1e-5
    actual, diagnostics = conjugate_gradient_matrix(
        operator,
        rhs,
        relative_tolerance=1e-12,
        max_iterations=100,
        absolute_damping=damping,
    )
    expected = torch.linalg.solve(
        explicit + damping * torch.eye(width, dtype=torch.float64),
        rhs.reshape(-1),
    ).reshape_as(rhs)
    torch.testing.assert_close(actual, expected, rtol=1e-8, atol=1e-8)
    assert diagnostics.converged


def test_fixed_budget_cg_does_not_stop_at_requested_tolerance() -> None:
    diagonal = torch.tensor([1.0, 2.0, 4.0], dtype=torch.float64)

    def operator(value: torch.Tensor) -> torch.Tensor:
        return diagonal * value

    rhs = torch.ones(3, dtype=torch.float64)
    early_solution, early = conjugate_gradient_matrix(
        operator,
        rhs,
        relative_tolerance=0.9,
        max_iterations=2,
    )
    fixed_solution, fixed = conjugate_gradient_matrix(
        operator,
        rhs,
        relative_tolerance=0.9,
        max_iterations=2,
        fixed_iterations=True,
    )
    assert early.iterations == 1
    assert fixed.iterations == 2
    assert not early.fixed_iterations
    assert fixed.fixed_iterations
    exact = rhs / diagonal
    assert torch.linalg.vector_norm(fixed_solution - exact) < torch.linalg.vector_norm(
        early_solution - exact
    )


def test_jacobi_preconditioned_cg_solves_ill_scaled_diagonal_in_one_step() -> None:
    diagonal = torch.tensor(
        [[1e-8, 1e-2], [1e4, 1e8]],
        dtype=torch.float64,
    )
    damping = 1e-6

    def operator(value: torch.Tensor) -> torch.Tensor:
        return diagonal * value

    rhs = torch.ones_like(diagonal)
    actual, diagnostics = conjugate_gradient_matrix(
        operator,
        rhs,
        relative_tolerance=1e-12,
        max_iterations=4,
        absolute_damping=damping,
        preconditioner=lambda value: value / (diagonal + damping),
    )
    torch.testing.assert_close(actual, rhs / (diagonal + damping))
    assert diagnostics.converged
    assert diagnostics.iterations == 1


def test_alpha_zero_product_anchor_and_monotonic_als() -> None:
    layout = _layout()
    covariance, _, A, D = _problem(layout)
    mapping = _mapping(layout)
    anchor_target = head_products(A, D, mapping)
    prior = quadratic_from_target(
        covariance=covariance,
        target=anchor_target,
        name="anchor",
        trace_normalize=True,
    )
    result = fit_routed_ov_joint(
        objective=prior,
        initial_A=A,
        initial_D=D,
        head_to_kv_group=mapping,
        coupling_mode="full_layer",
        maximum_sweeps=2,
        minimum_sweeps=1,
        patience=1,
        work_dtype=torch.float64,
    )
    assert result.initial_loss < 1e-9
    assert result.final_loss < 1e-8
    torch.testing.assert_close(
        head_products(result.A_unique, result.D_heads, mapping),
        anchor_target,
        rtol=1e-8,
        atol=1e-8,
    )
    losses = [result.decoder_only_loss] + [
        sweep.loss_after_encoders for sweep in result.sweeps
    ]
    assert all(right <= left + 1e-9 for left, right in zip(losses, losses[1:]))


def test_ragged_alpha_zero_and_thin_qr_preserve_source_product() -> None:
    layout = _layout(rank=3)
    covariance, _, _, _ = _problem(layout)
    mapping = _mapping(layout)
    ranks = (1, 3)
    torch.manual_seed(20260722)
    A = torch.zeros(2, 3, 3, dtype=torch.float64)
    D = torch.zeros(4, 3, layout.hidden_size, dtype=torch.float64)
    A[0, :, :1] = torch.randn(3, 1, dtype=torch.float64)
    A[1] = torch.eye(3, dtype=torch.float64)
    D[:2, :1] = torch.randn(2, 1, layout.hidden_size, dtype=torch.float64)
    D[2:] = torch.randn(2, 3, layout.hidden_size, dtype=torch.float64)
    source_product = head_products(A, D, mapping)
    prior = quadratic_from_target(
        covariance=covariance,
        target=source_product,
        name="ragged_source_prior",
        trace_normalize=True,
    )
    result = fit_routed_ov_joint(
        objective=prior,
        initial_A=A,
        initial_D=D,
        head_to_kv_group=mapping,
        group_ranks=ranks,
        coupling_mode="full_layer",
        maximum_sweeps=1,
        minimum_sweeps=1,
        cg_max_iterations=4,
        cg_fixed_iterations=True,
        encoder_group_indices=(0,),
        component_objectives={"prior": prior},
        final_decoder_solve=True,
        verify_encoder_step_objective=True,
        work_dtype=torch.float64,
    )
    torch.testing.assert_close(
        head_products(result.A_unique, result.D_heads, mapping),
        source_product,
        rtol=1e-8,
        atol=1e-8,
    )
    assert result.checkpoints[-1].decoder_relative_stationarity < 1e-8

    dense_v = torch.randn(layout.kv_width, layout.hidden_size, dtype=torch.float64)
    folded = fold_ragged_routed_ov_factors(
        dense_v_proj_weight=dense_v,
        A_unique=result.A_unique,
        D_heads=result.D_heads,
        head_to_kv_group=mapping,
        group_ranks=ranks,
        output_dtype=torch.float64,
    )
    assert [tuple(item.shape) for item in folded.v_group_weights] == [
        (1, layout.hidden_size),
        (3, layout.hidden_size),
    ]
    assert [tuple(item.shape) for item in folded.o_group_weights] == [
        (layout.hidden_size, 2),
        (layout.hidden_size, 6),
    ]
    assert folded.maximum_qr_product_error < 1e-12
    assert folded.maximum_dense_fold_error < 1e-12


def test_solver_checkpoints_final_redecoder_and_attribution_identity() -> None:
    layout = _layout()
    covariance, target, A, D = _problem(layout)
    mapping = _mapping(layout)
    routed = quadratic_from_target(
        covariance=covariance,
        target=target,
        name="routed",
        trace_normalize=True,
    )
    prior = quadratic_from_target(
        covariance=covariance,
        target=0.8 * target,
        name="prior",
        trace_normalize=True,
    )
    alpha = 0.3
    objective = combine_quadratics(
        prior,
        routed,
        right_weight=alpha,
        name="mixed",
    )
    captured: dict[tuple[str, int], tuple[torch.Tensor, torch.Tensor]] = {}

    def capture(checkpoint, checkpoint_A, checkpoint_D) -> None:
        captured[(checkpoint.boundary, checkpoint.sweep)] = (
            checkpoint_A.detach().clone(),
            checkpoint_D.detach().clone(),
        )

    legacy = fit_routed_ov_joint(
        objective=objective,
        initial_A=A,
        initial_D=D,
        head_to_kv_group=mapping,
        coupling_mode="full_layer",
        maximum_sweeps=2,
        minimum_sweeps=2,
        patience=2,
        work_dtype=torch.float64,
    )
    final = fit_routed_ov_joint(
        objective=objective,
        initial_A=A,
        initial_D=D,
        head_to_kv_group=mapping,
        coupling_mode="full_layer",
        maximum_sweeps=2,
        minimum_sweeps=2,
        patience=2,
        component_objectives={"prior": prior, "routed": routed},
        checkpoint_callback=capture,
        final_decoder_solve=True,
        verify_encoder_step_objective=True,
        work_dtype=torch.float64,
    )
    assert [
        (checkpoint.boundary, checkpoint.sweep)
        for checkpoint in final.checkpoints
    ] == [
        ("anchor", 0),
        ("decoder_only", 0),
        ("after_encoder", 1),
        ("after_redecoder", 1),
        ("after_encoder", 2),
        ("after_redecoder", 2),
    ]
    legacy_A, legacy_D = captured[("after_encoder", 2)]
    torch.testing.assert_close(legacy.A_unique, legacy_A)
    torch.testing.assert_close(legacy.D_heads, legacy_D)
    assert abs(legacy.final_loss - final.legacy_final_loss) < 1e-10
    assert final.final_loss <= final.legacy_final_loss + 1e-10
    assert final.final_decoder is not None
    assert final.checkpoints[-1].decoder_relative_stationarity < 1e-8

    for checkpoint in final.checkpoints:
        losses = dict(checkpoint.component_losses)
        expected = (1.0 - alpha) * losses["prior"] + alpha * losses["routed"]
        assert abs(checkpoint.loss - expected) < 1e-9

    attribution = final.attribution
    assert abs(attribution.identity_error) < 1e-9
    assert abs(
        attribution.decoder_reduction
        + attribution.encoder_reduction
        - attribution.total_reduction
    ) < 1e-9
    assert attribution.steps[-1].loss_after_redecoder is not None
    assert abs(attribution.endpoint_loss - final.final_loss) < 1e-10
    for sweep in final.sweeps:
        for step in sweep.encoder_steps:
            assert abs(
                step.predicted_change - step.realized_change
            ) < 1e-9
            assert abs(
                step.realized_change - (step.new_loss - step.old_loss)
            ) < 1e-12


def test_full_rank_exactness_and_folded_execution() -> None:
    layout = _layout(rank=3)
    mapping = _mapping(layout)
    covariance, target, _, _ = _problem(layout)
    A = torch.eye(layout.head_dim, dtype=torch.float64).repeat(
        layout.num_key_value_heads,
        1,
        1,
    )
    D = target.clone()
    objective = quadratic_from_target(
        covariance=covariance,
        target=target,
        name="rank_full",
        trace_normalize=False,
    )
    assert evaluate_quadratic(objective, A, D, mapping) < 1e-9

    torch.manual_seed(4)
    dense_v = torch.randn(layout.kv_width, layout.hidden_size, dtype=torch.float64)
    folded = fold_routed_ov_factors(
        layout=layout,
        dense_v_proj_weight=dense_v,
        A_unique=A,
        D_heads=D,
        head_to_kv_group=mapping,
        output_dtype=torch.float64,
    )
    recovered_A, recovered_D, diagnostics = extract_value_coordinate_factors(
        layout=layout,
        dense_v_proj_weight=dense_v,
        compressed_v_proj_weight=folded.v_proj_compressed_weight,
        compressed_o_proj_weight=folded.o_decoder_weight,
        work_dtype=torch.float64,
    )
    assert diagnostics["maximum_value_encoder_recovery_error"] < 1e-10
    torch.testing.assert_close(
        head_products(recovered_A, recovered_D, mapping),
        target,
        rtol=1e-9,
        atol=1e-9,
    )


def test_cross_head_coupling_can_strictly_improve_full_metric() -> None:
    # Two anti-correlated heads admit coordinated residual cancellation that
    # independent diagonal fitting cannot see.
    layout = GQAVOLayout(
        hidden_size=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=2,
        rank=1,
    )
    mapping = torch.tensor([0, 1])
    rho = 0.95
    eye = torch.eye(2, dtype=torch.float64)
    routed_cross = rho * torch.tensor(
        [[0.0, 1.0], [1.0, 0.0]],
        dtype=torch.float64,
    )
    covariance = torch.stack(
        [
            torch.stack([eye, routed_cross]),
            torch.stack([routed_cross.T, eye]),
        ]
    )
    target = torch.stack(
        [
            torch.eye(2, dtype=torch.float64),
            torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float64),
        ]
    )
    A = torch.tensor(
        [
            [[1.0], [0.0]],
            [[1.0], [0.0]],
        ],
        dtype=torch.float64,
    )
    full = quadratic_from_target(
        covariance=covariance,
        target=target,
        name="full",
        trace_normalize=False,
    )
    diagonal_covariance = mask_routed_covariance(
        covariance,
        head_to_kv_group=mapping,
        mode="diagonal",
    )
    diagonal = quadratic_from_target(
        covariance=diagonal_covariance,
        target=target,
        name="diagonal",
        trace_normalize=False,
    )
    diagonal_D, _ = solve_free_decoder(
        objective=diagonal,
        A_unique=A,
        head_to_kv_group=mapping,
        coupling_mode="diagonal",
    )
    full_D, _ = solve_free_decoder(
        objective=full,
        A_unique=A,
        head_to_kv_group=mapping,
        coupling_mode="full_layer",
    )
    assert evaluate_quadratic(full, A, full_D, mapping) < evaluate_quadratic(
        full,
        A,
        diagonal_D,
        mapping,
    )
