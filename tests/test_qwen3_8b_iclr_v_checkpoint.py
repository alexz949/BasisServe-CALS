from __future__ import annotations

import pytest
import torch

from evaluation import build_llama31_8b_palu_m_checkpoint as palu_builder
from evaluation.build_qwen3_8b_iclr_v_checkpoint import (
    activate_experiment_profile,
    factorize_palu_projection,
    factorize_weight_svd_projection,
    fisher_layer_ranks,
    run_id,
)
from evaluation.eval_qwen3_8b_iclr_quality import (
    TASKS,
    activate_quality_profile,
    _cuda_device,
    summarize_commonsense,
)
from evaluation.summarize_iclr_quality import _run_ids, _trend_checks


def test_iclr_run_ids_match_the_experiment_matrix() -> None:
    try:
        activate_experiment_profile("qwen3_8b")
        assert run_id("dense", None, 1) == "Q3-8B-Dense"
        assert run_id("weight-svd", 80, 1) == "Q3-8B-SVD-R80"
        assert run_id("palu-fisher", 96, 1) == "Q3-8B-PALUM-R96"
        assert run_id("palu-fisher", 64, 4) == "Q3-8B-PALUG4-R64"
        activate_experiment_profile("llama31_8b")
        assert run_id("dense", None, 1) == "L31-8B-Dense"
        assert run_id("palu-fisher", 80, 2) == "L31-8B-PALUG2-R80"
    finally:
        activate_experiment_profile("qwen3_8b")


def test_weight_svd_factorization_is_per_physical_head() -> None:
    torch.manual_seed(43)
    weight = torch.randn(8, 7, dtype=torch.float32)
    writer, decoder, diagnostics = factorize_weight_svd_projection(
        weight,
        ranks=[4, 4],
        output_dtype=torch.float32,
    )

    assert writer.shape == (8, 7)
    assert decoder.shape == (2, 4, 4)
    reconstructed = torch.cat(
        [
            decoder[group] @ writer[group * 4 : (group + 1) * 4]
            for group in range(2)
        ]
    )
    torch.testing.assert_close(reconstructed, weight, rtol=2e-5, atol=2e-5)
    assert diagnostics["relative_frobenius_error"] == pytest.approx(0.0, abs=2e-5)


def test_palu_factorization_uses_the_activation_weighted_objective() -> None:
    torch.manual_seed(47)
    weight = torch.randn(8, 7, dtype=torch.float64)
    raw = torch.randn(7, 7, dtype=torch.float64)
    covariance = raw @ raw.T + torch.eye(7, dtype=torch.float64)
    cholesky = torch.linalg.cholesky(covariance)
    writer, decoder, diagnostics = factorize_palu_projection(
        weight,
        cholesky,
        ranks=[4, 4],
        output_dtype=torch.float64,
    )

    reconstructed = torch.cat(
        [
            decoder[group] @ writer[group * 4 : (group + 1) * 4]
            for group in range(2)
        ]
    )
    torch.testing.assert_close(reconstructed, weight, rtol=1e-10, atol=1e-10)
    assert diagnostics["relative_frobenius_error"] < 1e-10
    assert diagnostics["relative_activation_weighted_error"] < 1e-10


@pytest.mark.parametrize(
    ("head_group_size", "expected_groups", "maximum_group_rank"),
    [(1, 8, 128), (2, 4, 256), (4, 2, 512)],
)
def test_fisher_schedule_uses_equivalent_rank_geometry(
    head_group_size: int,
    expected_groups: int,
    maximum_group_rank: int,
) -> None:
    try:
        palu_builder.activate_model_profile("qwen3_8b")
        fisher = {
            f"model.layers.{layer}.self_attn.v_proj": float(layer + 1)
            for layer in range(palu_builder.NUM_LAYERS)
        }
        ranks, rank_sum, total_rank = fisher_layer_ranks(
            fisher,
            equivalent_rank=96,
            head_group_size=head_group_size,
        )
        assert len(ranks) == 36
        assert total_rank == 36 * 8 * 128
        assert rank_sum == sum(sum(layer) for layer in ranks)
        assert all(len(layer) == expected_groups for layer in ranks)
        assert all(len(set(layer)) == 1 for layer in ranks)
        assert all(
            0 < rank <= maximum_group_rank and rank % 32 == 0
            for layer in ranks
            for rank in layer
        )
    finally:
        palu_builder.activate_model_profile("llama31_8b")


def test_commonsense_summary_prefers_acc_norm_then_acc() -> None:
    results = {}
    expected = []
    for index, task in enumerate(TASKS):
        value = 0.4 + index / 100
        expected.append(value)
        if index % 2:
            results[task] = {"acc,none": value, "acc_stderr,none": 0.01}
        else:
            results[task] = {
                "acc_norm,none": value,
                "acc,none": value - 0.1,
            }
    summarized = summarize_commonsense({"results": results})
    assert summarized is not None
    rows, average = summarized
    assert [row["metric"] for row in rows] == [
        "acc_norm" if index % 2 == 0 else "acc"
        for index in range(len(TASKS))
    ]
    assert average == pytest.approx(sum(expected) / len(expected))


def test_quality_profiles_pin_model_specific_formats() -> None:
    from evaluation import eval_qwen3_8b_iclr_quality as quality

    try:
        activate_quality_profile("llama31_8b")
        assert quality.CHECKPOINT_FORMAT == "basisserve.llama31_8b.iclr_v_factors.v1"
        assert quality.FORMAT == "basisserve.llama31_8b.iclr_quality.v1"
        assert quality.STAGE_FORMATS["c4"] == (
            "basisserve.llama31_8b.iclr_quality.c4_validation.v1"
        )
    finally:
        activate_quality_profile("qwen3_8b")


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0, 0), ("2", 2), ("cuda:3", 3), ("cpu", None), ("disk", None)],
)
def test_cuda_device_map_values_are_normalized(value: object, expected: int | None) -> None:
    assert _cuda_device(value) == expected


def test_matrix_trend_audit_checks_dense_svd_and_palu_geometry() -> None:
    prefix = "Q3-8B"
    results = {}
    for run_id in _run_ids(prefix):
        if run_id.endswith("-Dense"):
            metrics = (7.0, 9.0, 0.70)
        elif "-SVD-" in run_id:
            metrics = (100.0, 120.0, 0.40)
        elif "-PALUM-" in run_id:
            metrics = (12.0, 14.0, 0.60)
        elif "-PALUG2-" in run_id:
            metrics = (10.0, 12.0, 0.63)
        else:
            metrics = (8.0, 10.0, 0.67)
        results[run_id] = {
            "metrics": {
                "wikitext2_ppl": metrics[0],
                "c4_validation_128_ppl": metrics[1],
                "average_accuracy": metrics[2],
            }
        }

    checks = _trend_checks(results, prefix)
    assert checks
    assert all(checks.values())
