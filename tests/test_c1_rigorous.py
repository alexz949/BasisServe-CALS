from __future__ import annotations

import math

import torch

from basisserve.analysis.c1_rigorous import (
    decoder_union_energy_ranks,
    embed_c1_in_lr_allreduce,
    fit_strong_lr_allreduce_rank_bank,
    pairwise_decoder_subspace_metrics,
    permuted_private_products_weight,
    private_products_weight,
    relative_output_mse,
    ring_allgather_bytes_per_rank,
    ring_allreduce_bytes_per_rank,
    rotate_private_factors,
    simulate_c1_allgather,
    simulate_c1_local_decode_allreduce,
    simulate_lr_allreduce,
    wire_matched_allreduce_rank,
)
from basisserve.core.tp_source_wo_c1 import TPSourceWOLayout
from basisserve.core.tp_source_wo_fit import (
    TPSourceWOFitConfig,
    fit_tp_source_wo_c1,
)
from evaluation.eval_qwen3_8b_wo_c1_cross_source_covariance_ppl import (
    materialize_private_weight,
)
from evaluation.eval_qwen3_8b_wo_c1_lr_ar_quality import _materialize_weight
from evaluation.fit_qwen3_8b_wo_c1_independent_local import (
    _evaluate_factors,
    _objectives,
    fit_independent_local,
)
from evaluation.fit_qwen3_8b_wo_c1_cross_source_covariance import (
    refit_fixed_encoders,
)
from evaluation.run_c1_synthetic_controlled_angles import run_trial
from evaluation.summarize_qwen3_8b_wo_c1_uniform_rank_sweep import (
    relative_reduction,
)


def _generator() -> torch.Generator:
    return torch.Generator().manual_seed(20260828)


def test_relative_reduction_accepts_an_exact_zero_result() -> None:
    assert relative_reduction(0.0, 0.25) == 1.0


def test_capacity_matched_lr_allreduce_contains_c1_exactly() -> None:
    generator = _generator()
    source_widths = (4, 3, 5)
    source_ranks = (2, 1, 3)
    output_width = 7
    rows = 11
    encoders = tuple(
        torch.randn(width, rank, generator=generator, dtype=torch.float64)
        for width, rank in zip(source_widths, source_ranks)
    )
    decoders = tuple(
        torch.randn(rank, output_width, generator=generator, dtype=torch.float64)
        for rank in source_ranks
    )
    activations = tuple(
        torch.randn(rows, width, generator=generator, dtype=torch.float64)
        for width in source_widths
    )

    lr_encoders, shared_decoder = embed_c1_in_lr_allreduce(encoders, decoders)
    c1_output = simulate_c1_allgather(activations, encoders, decoders)
    lr_output = simulate_lr_allreduce(
        activations,
        lr_encoders,
        shared_decoder,
    )

    assert shared_decoder.shape == (sum(source_ranks), output_width)
    for source, (c1_encoder, c1_decoder, lr_encoder) in enumerate(
        zip(encoders, decoders, lr_encoders)
    ):
        torch.testing.assert_close(
            lr_encoder @ shared_decoder,
            c1_encoder @ c1_decoder,
            rtol=1.0e-14,
            atol=1.0e-14,
            msg=f"source {source} was not embedded in disjoint coordinates",
        )
    # The functions are algebraically identical.  Their GEMMs use a different
    # accumulation order, so execution equality is limited by FP64 rounding.
    torch.testing.assert_close(lr_output, c1_output, rtol=1.0e-13, atol=1.0e-13)


def test_c1_allgather_and_local_decode_allreduce_are_the_same_function() -> None:
    generator = _generator()
    source_width = 5
    source_rank = 3
    output_width = 7
    rows = 13
    activations = tuple(
        torch.randn(rows, source_width, generator=generator, dtype=torch.float64)
        for _ in range(4)
    )
    encoders = tuple(
        torch.randn(source_width, source_rank, generator=generator, dtype=torch.float64)
        for _ in range(4)
    )
    decoders = tuple(
        torch.randn(source_rank, output_width, generator=generator, dtype=torch.float64)
        for _ in range(4)
    )

    gathered = simulate_c1_allgather(activations, encoders, decoders)
    reduced = simulate_c1_local_decode_allreduce(activations, encoders, decoders)
    torch.testing.assert_close(gathered, reduced, rtol=1.0e-13, atol=1.0e-13)


def test_global_lr_solver_matches_or_beats_capacity_matched_c1() -> None:
    generator = _generator()
    source_widths = (4, 3, 5)
    source_ranks = (2, 1, 3)
    output_width = 7
    encoders = tuple(
        torch.randn(width, rank, generator=generator, dtype=torch.float64)
        for width, rank in zip(source_widths, source_ranks)
    )
    decoders = tuple(
        torch.randn(rank, output_width, generator=generator, dtype=torch.float64)
        for rank in source_ranks
    )
    c1_weight = private_products_weight(encoders, decoders)
    samples = torch.randn(
        37,
        sum(source_widths),
        generator=generator,
        dtype=torch.float64,
    )
    covariance = samples.transpose(0, 1) @ samples / samples.shape[0]
    factors = fit_strong_lr_allreduce_rank_bank(
        c1_weight,
        covariance,
        covariance,
        ranks=(sum(source_ranks),),
        covariance_damping=0.0,
        work_dtype=torch.float64,
        factor_dtype=torch.float64,
    )[0]

    lr_error = relative_output_mse(
        c1_weight,
        factors.reconstructed_weight(),
        covariance,
    )
    assert lr_error < 1.0e-24
    assert lr_error <= 1.0e-20  # The corresponding C1 target has zero error.
    assert float(factors.metrics["relative_eigen_residual"]) < 1.0e-12


def test_wo_only_capacity_lr_contains_fitted_tp_source_c1() -> None:
    generator = _generator()
    layout = TPSourceWOLayout(
        input_width=8,
        output_width=6,
        tp_size=2,
        source_rank=2,
    )
    weight = torch.randn(6, 8, generator=generator, dtype=torch.float64)
    fit_samples = torch.randn(31, 8, generator=generator, dtype=torch.float64)
    heldout_samples = torch.randn(29, 8, generator=generator, dtype=torch.float64)
    fit_covariance = fit_samples.transpose(0, 1) @ fit_samples
    heldout_covariance = heldout_samples.transpose(0, 1) @ heldout_samples
    c1 = fit_tp_source_wo_c1(
        weight,
        fit_covariance,
        heldout_covariance,
        layout,
        config=TPSourceWOFitConfig(
            encoder_sweeps=1,
            covariance_damping=0.0,
        ),
        work_dtype=torch.float64,
        factor_dtype=torch.float64,
    )
    capacity = fit_strong_lr_allreduce_rank_bank(
        weight,
        fit_covariance,
        heldout_covariance,
        ranks=(layout.gathered_width,),
        covariance_damping=0.0,
        work_dtype=torch.float64,
        factor_dtype=torch.float64,
    )[0]

    assert float(capacity.metrics["fit_damped_relative_mse"]) <= (
        c1.fit_relative_mse + 1.0e-12
    )
    assert len(c1.diagnostics["encoder_solves"]) == layout.tp_size
    assert all(
        row["solver"] == "exact_single_source_two_sided_cholesky"
        for row in c1.diagnostics["encoder_solves"]
    )


def test_independent_local_and_joint_fits_win_their_own_objectives() -> None:
    generator = _generator()
    layout = TPSourceWOLayout(
        input_width=8,
        output_width=6,
        tp_size=2,
        source_rank=2,
    )
    weight = torch.randn(6, 8, generator=generator, dtype=torch.float64)
    fit = torch.randn(97, 8, generator=generator, dtype=torch.float64)
    heldout = torch.randn(89, 8, generator=generator, dtype=torch.float64)
    fit[:, 4:] += 0.7 * fit[:, :4]
    heldout[:, 4:] += 0.7 * heldout[:, :4]
    fit_covariance = fit.transpose(0, 1) @ fit
    heldout_covariance = heldout.transpose(0, 1) @ heldout
    objectives, target, mapping, _ = _objectives(
        weight,
        fit_covariance,
        heldout_covariance,
        layout,
        covariance_damping=1.0e-5,
    )
    local = fit_independent_local(
        objectives,
        target,
        mapping,
        source_rank=layout.source_rank,
    )
    joint = fit_tp_source_wo_c1(
        weight,
        fit_covariance,
        heldout_covariance,
        layout,
        config=TPSourceWOFitConfig(
            encoder_sweeps=0,
            covariance_damping=1.0e-5,
        ),
        work_dtype=torch.float64,
        factor_dtype=torch.float64,
    )
    joint_metrics = _evaluate_factors(
        objectives,
        joint.source_encoders,
        joint.source_decoders,
        mapping,
    )

    assert local.source_encoders.shape == (2, 4, 2)
    assert local.source_decoders.shape == (2, 2, 6)
    assert local.diagnostics["decoder_solve"]["partition_sizes"] == (1, 1)
    assert local.metrics["local_fit_damped_relative_mse"] <= (
        joint_metrics["local_fit_damped_relative_mse"] + 1.0e-12
    )
    assert joint_metrics["final_fit_damped_relative_mse"] <= (
        local.metrics["final_fit_damped_relative_mse"] + 1.0e-12
    )


def test_fixed_encoder_covariance_refits_win_their_own_objectives() -> None:
    generator = _generator()
    layout = TPSourceWOLayout(
        input_width=8,
        output_width=6,
        tp_size=2,
        source_rank=2,
    )
    weight = torch.randn(6, 8, generator=generator, dtype=torch.float64)
    fit = torch.randn(97, 8, generator=generator, dtype=torch.float64)
    heldout = torch.randn(89, 8, generator=generator, dtype=torch.float64)
    fit[:, 4:] += 0.8 * fit[:, :4]
    heldout[:, 4:] += 0.8 * heldout[:, :4]
    objectives, _, mapping, _ = _objectives(
        weight,
        fit.transpose(0, 1) @ fit,
        heldout.transpose(0, 1) @ heldout,
        layout,
        covariance_damping=1.0e-5,
    )
    fixed_encoders = torch.randn(2, 4, 2, generator=generator, dtype=torch.float64)
    refits = refit_fixed_encoders(objectives, fixed_encoders, mapping)
    full = _evaluate_factors(objectives, fixed_encoders, refits.full_decoders, mapping)
    block = _evaluate_factors(
        objectives, fixed_encoders, refits.block_diagonal_decoders, mapping
    )

    assert refits.full_diagnostics["partition_sizes"] == (2,)
    assert refits.block_diagonal_diagnostics["partition_sizes"] == (1, 1)
    assert block["local_fit_damped_relative_mse"] <= (
        full["local_fit_damped_relative_mse"] + 1.0e-12
    )
    assert full["final_fit_damped_relative_mse"] <= (
        block["final_fit_damped_relative_mse"] + 1.0e-12
    )


def test_full_and_block_decoder_refits_match_without_cross_covariance() -> None:
    generator = _generator()
    layout = TPSourceWOLayout(
        input_width=8,
        output_width=6,
        tp_size=2,
        source_rank=2,
    )
    weight = torch.randn(6, 8, generator=generator, dtype=torch.float64)
    first = torch.randn(97, 4, generator=generator, dtype=torch.float64)
    second = torch.randn(101, 4, generator=generator, dtype=torch.float64)
    covariance = torch.zeros(8, 8, dtype=torch.float64)
    covariance[:4, :4] = first.transpose(0, 1) @ first
    covariance[4:, 4:] = second.transpose(0, 1) @ second
    objectives, _, mapping, _ = _objectives(
        weight,
        covariance,
        covariance,
        layout,
        covariance_damping=1.0e-5,
    )
    fixed_encoders = torch.randn(2, 4, 2, generator=generator, dtype=torch.float64)
    refits = refit_fixed_encoders(objectives, fixed_encoders, mapping)

    torch.testing.assert_close(
        refits.full_decoders,
        refits.block_diagonal_decoders,
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_latent_orthogonal_rotation_preserves_private_product() -> None:
    generator = _generator()
    encoder = torch.randn(6, 3, generator=generator, dtype=torch.float64)
    decoder = torch.randn(3, 8, generator=generator, dtype=torch.float64)
    orthogonal = torch.linalg.qr(
        torch.randn(3, 3, generator=generator, dtype=torch.float64)
    ).Q

    rotated_encoders, rotated_decoders = rotate_private_factors(
        (encoder,), (decoder,), (orthogonal,)
    )
    rotated_encoder = rotated_encoders[0]
    rotated_decoder = rotated_decoders[0]

    torch.testing.assert_close(
        rotated_encoder @ rotated_decoder,
        encoder @ decoder,
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_latent_rotation_rejects_nonorthogonal_basis_change() -> None:
    encoder = torch.eye(3, dtype=torch.float64)
    decoder = torch.eye(3, dtype=torch.float64)
    nonorthogonal = 2.0 * torch.eye(3, dtype=torch.float64)
    try:
        rotate_private_factors((encoder,), (decoder,), (nonorthogonal,))
    except ValueError as error:
        assert "not orthogonal" in str(error)
    else:
        raise AssertionError("nonorthogonal latent rotation was accepted")


def test_paired_source_permutation_moves_complete_private_products() -> None:
    encoders = (
        torch.tensor([[1.0], [2.0]], dtype=torch.float64),
        torch.tensor([[3.0], [4.0]], dtype=torch.float64),
        torch.tensor([[5.0], [6.0]], dtype=torch.float64),
    )
    decoders = (
        torch.tensor([[1.0, 10.0]], dtype=torch.float64),
        torch.tensor([[2.0, 20.0]], dtype=torch.float64),
        torch.tensor([[3.0, 30.0]], dtype=torch.float64),
    )

    identity = permuted_private_products_weight(encoders, decoders, (0, 1, 2))
    expected_identity = private_products_weight(encoders, decoders)
    torch.testing.assert_close(identity, expected_identity, rtol=0.0, atol=0.0)

    permuted = permuted_private_products_weight(encoders, decoders, (2, 0, 1))
    expected = torch.cat(
        (
            encoders[2] @ decoders[2],
            encoders[0] @ decoders[0],
            encoders[1] @ decoders[1],
        ),
        dim=0,
    ).transpose(0, 1)
    torch.testing.assert_close(permuted, expected, rtol=0.0, atol=0.0)


def test_paired_source_permutation_rejects_non_bijections() -> None:
    encoders = (torch.eye(2), torch.eye(2))
    decoders = (torch.eye(2), torch.eye(2))
    for invalid in ((0,), (0, 0), (0, 2)):
        try:
            permuted_private_products_weight(encoders, decoders, invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid permutation was accepted: {invalid}")


def test_value_folding_identity_fp64_and_fp32() -> None:
    generator = _generator()
    for dtype, tolerance in ((torch.float64, 1.0e-12), (torch.float32, 2.0e-6)):
        attention = torch.randn(5, 17, generator=generator, dtype=dtype).softmax(dim=-1)
        values = torch.randn(17, 8, generator=generator, dtype=dtype)
        encoder = torch.randn(8, 3, generator=generator, dtype=dtype)

        post_attention = (attention @ values) @ encoder
        folded = attention @ (values @ encoder)

        torch.testing.assert_close(
            folded,
            post_attention,
            rtol=tolerance,
            atol=tolerance,
        )


def test_source_ordering_requires_matching_decoder_order() -> None:
    first_latent = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
    second_latent = torch.tensor([[3.0]], dtype=torch.float64)
    first_decoder = torch.tensor([[1.0, 0.0], [0.0, 2.0]], dtype=torch.float64)
    second_decoder = torch.tensor([[4.0, 5.0]], dtype=torch.float64)

    gathered = torch.cat((first_latent, second_latent), dim=1)
    decoder = torch.cat((first_decoder, second_decoder), dim=0)
    expected = gathered @ decoder
    wrong = torch.cat((second_latent, first_latent), dim=1) @ decoder
    repaired = torch.cat((second_latent, first_latent), dim=1) @ torch.cat(
        (second_decoder, first_decoder), dim=0
    )

    assert not torch.equal(wrong, expected)
    torch.testing.assert_close(repaired, expected, rtol=0.0, atol=0.0)


def test_ideal_ring_budget_matching() -> None:
    for source_rank in (512, 640, 768, 896, 1024):
        source_ranks = (source_rank,) * 4
        ar_rank = wire_matched_allreduce_rank(source_ranks)
        assert ar_rank == 2 * source_rank
        assert sum(source_ranks) <= 4096

        allreduce = ring_allreduce_bytes_per_rank(
            rows=8,
            rank=ar_rank,
            tp_size=4,
            dtype_bytes=2,
        )
        allgather = ring_allgather_bytes_per_rank(
            rows=8,
            source_ranks=source_ranks,
            dtype_bytes=2,
        )
        assert allreduce == allgather == 48.0 * source_rank


def test_synthetic_subspace_angle_endpoints() -> None:
    rank = 2
    output_width = 4
    identity = torch.eye(rank, dtype=torch.float64)
    first_decoder = torch.eye(output_width, dtype=torch.float64)[:rank]
    shared_decoder = first_decoder.clone()
    orthogonal_decoder = torch.eye(output_width, dtype=torch.float64)[rank:]
    covariance = torch.eye(2 * rank, dtype=torch.float64)

    shared_weight = private_products_weight(
        (identity, identity),
        (first_decoder, shared_decoder),
    )
    orthogonal_weight = private_products_weight(
        (identity, identity),
        (first_decoder, orthogonal_decoder),
    )
    shared_fit = fit_strong_lr_allreduce_rank_bank(
        shared_weight,
        covariance,
        covariance,
        ranks=(rank,),
        covariance_damping=0.0,
        work_dtype=torch.float64,
        factor_dtype=torch.float64,
    )[0]
    orthogonal_fit = fit_strong_lr_allreduce_rank_bank(
        orthogonal_weight,
        covariance,
        covariance,
        ranks=(rank,),
        covariance_damping=0.0,
        work_dtype=torch.float64,
        factor_dtype=torch.float64,
    )[0]

    shared_error = relative_output_mse(
        shared_weight,
        shared_fit.reconstructed_weight(),
        covariance,
    )
    orthogonal_error = relative_output_mse(
        orthogonal_weight,
        orthogonal_fit.reconstructed_weight(),
        covariance,
    )
    assert shared_error < 1.0e-24
    assert math.isclose(orthogonal_error, 0.5, rel_tol=1.0e-12, abs_tol=1.0e-12)


def test_decoder_subspace_metrics_shared_and_orthogonal_endpoints() -> None:
    rank = 2
    output_width = 8
    shared = torch.eye(output_width, dtype=torch.float64)[:rank]
    shared_metrics = pairwise_decoder_subspace_metrics((shared,) * 4)
    assert all(float(row["maximum_angle_degrees"]) < 1.0e-5 for row in shared_metrics)
    assert all(
        math.isclose(float(row["projection_overlap"]), 1.0, abs_tol=1.0e-12)
        for row in shared_metrics
    )
    assert decoder_union_energy_ranks((shared,) * 4)["energy_ranks"]["95%"] == rank

    orthogonal = tuple(
        torch.eye(output_width, dtype=torch.float64)[
            source * rank : (source + 1) * rank
        ]
        for source in range(4)
    )
    orthogonal_metrics = pairwise_decoder_subspace_metrics(orthogonal)
    assert all(
        math.isclose(
            float(row["minimum_angle_degrees"]),
            90.0,
            abs_tol=1.0e-12,
        )
        for row in orthogonal_metrics
    )
    assert all(float(row["projection_overlap"]) == 0.0 for row in orthogonal_metrics)
    union = decoder_union_energy_ranks(orthogonal)
    assert union["algebraic_rank"] == 4 * rank
    assert union["energy_ranks"]["95%"] == 4 * rank


def test_controlled_angle_exact_solvers_follow_analytic_curve() -> None:
    trials = [
        run_trial(
            angle_degrees=angle,
            seed=20260828,
            tp_size=4,
            source_width=16,
            source_rank=4,
            output_width=32,
            rows=128,
            wire_dtype_bytes=2,
        )
        for angle in (0.0, 45.0, 90.0)
    ]
    errors = []
    for trial in trials:
        assert trial["status"] == "pass"
        assert (
            trial["budget"]["c1_allgather_bytes_per_rank_per_row"]
            == (trial["budget"]["lr_allreduce_bytes_per_rank_per_row"])
        )
        assert trial["c1_allgather"]["heldout_summed_output_relative_mse"] <= 1.0e-24
        measured = float(trial["lr_allreduce"]["heldout_summed_output_relative_mse"])
        expected = float(trial["lr_allreduce"]["analytic_relative_mse"])
        assert math.isclose(measured, expected, rel_tol=1.0e-11, abs_tol=2.0e-12)
        errors.append(measured)
    assert errors[0] <= 1.0e-24
    assert errors[0] <= errors[1] <= errors[2]
    assert math.isclose(errors[2], 0.5, rel_tol=1.0e-12, abs_tol=2.0e-12)


def test_phase1_quality_materializes_private_c1_in_source_order() -> None:
    generator = _generator()
    encoders = torch.randn(3, 5, 2, generator=generator, dtype=torch.float64)
    decoders = torch.randn(3, 2, 7, generator=generator, dtype=torch.float64)
    payload = {
        "c1_source_encoders": encoders,
        "c1_source_decoders": decoders,
    }

    actual = _materialize_weight(payload, "wo_c1_ag", device=torch.device("cpu"))
    expected = private_products_weight(tuple(encoders), tuple(decoders)).float()

    torch.testing.assert_close(actual, expected, rtol=1.0e-6, atol=1.0e-6)


def test_experiment_j_ppl_materializes_each_decoder_with_fixed_encoders() -> None:
    encoders = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 2.0]],
            [[3.0, 0.0], [0.0, 4.0]],
        ]
    )
    full = torch.tensor(
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
        ]
    )
    block = -full
    payload = {
        "fixed_source_encoders": encoders,
        "full_source_decoders": full,
        "block_diagonal_source_decoders": block,
    }

    full_weight = materialize_private_weight(
        payload,
        decoder_key="full_source_decoders",
        device=torch.device("cpu"),
    )
    block_weight = materialize_private_weight(
        payload,
        decoder_key="block_diagonal_source_decoders",
        device=torch.device("cpu"),
    )
    expected = torch.bmm(encoders, full).reshape(4, 3).transpose(0, 1)
    torch.testing.assert_close(full_weight, expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(block_weight, -expected, rtol=0.0, atol=0.0)


def test_phase1_quality_materializes_lr_allreduce_weight() -> None:
    generator = _generator()
    input_factor = torch.randn(9, 4, generator=generator, dtype=torch.float64)
    decoder = torch.randn(4, 6, generator=generator, dtype=torch.float64)
    payload = {
        "lr_wire_input_factor": input_factor,
        "lr_wire_shared_decoder": decoder,
    }

    actual = _materialize_weight(
        payload,
        "wo_lr_ar_wire",
        device=torch.device("cpu"),
    )
    expected = decoder.float().transpose(0, 1) @ input_factor.float().transpose(0, 1)

    torch.testing.assert_close(actual, expected, rtol=1.0e-6, atol=1.0e-6)
