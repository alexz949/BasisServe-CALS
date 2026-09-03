from __future__ import annotations

import hashlib
import json

import pytest

from evaluation.build_qwen3_8b_c1_two_sided_factorized_kl_schedule import (
    allocate_layer_schedule,
    local_error_curves_from_factor_results,
    predict_factorized_costs,
    predict_two_sided_factorized_costs,
)
from evaluation import eval_qwen3_32b_c1_wikitext as evaluator


def test_one_probe_factorization_recovers_sensitivity_and_exact_budget() -> None:
    ranks = (32, 64, 96, 128)
    errors = (
        {32: 0.4, 64: 0.2, 96: 0.1, 128: 0.0},
        {32: 0.4, 64: 0.2, 96: 0.1, 128: 0.0},
    )
    costs, sensitivities = predict_factorized_costs(
        errors,
        (-1.0, -0.1),
        candidate_ranks=ranks,
        anchor_rank=64,
        probe_rank=96,
    )

    assert sensitivities == pytest.approx((10.0, 1.0))
    assert costs[0][64] == pytest.approx(0.0)
    assert costs[0][32] == pytest.approx(2.0)
    assert costs[0][96] == pytest.approx(-1.0)
    schedule, total_cost = allocate_layer_schedule(
        costs,
        candidate_ranks=ranks,
        anchor_rank=64,
    )
    assert schedule == (96, 32)
    assert sum(schedule) == 2 * 64
    assert total_cost == pytest.approx(-0.8)


def test_one_probe_factorization_clips_nonpositive_gain_to_zero() -> None:
    costs, sensitivities = predict_factorized_costs(
        ({32: 0.4, 64: 0.2, 96: 0.1, 128: 0.0},),
        (0.1,),
        candidate_ranks=(32, 64, 96, 128),
        anchor_rank=64,
        probe_rank=96,
    )

    assert sensitivities == (0.0,)
    assert costs == ({32: 0.0, 64: 0.0, 96: -0.0, 128: -0.0},)


def test_two_sided_factorization_uses_distinct_compression_and_expansion_slopes() -> None:
    costs, compression, expansion = predict_two_sided_factorized_costs(
        (
            {
                32: 0.5,
                48: 0.3,
                64: 0.2,
                80: 0.15,
                96: 0.1,
                112: 0.05,
                128: 0.0,
            },
        ),
        (0.6,),
        (-0.1,),
        candidate_ranks=(32, 48, 64, 80, 96, 112, 128),
        anchor_rank=64,
        compression_probe_rank=32,
        expansion_probe_rank=96,
    )

    assert compression == pytest.approx((2.0,))
    assert expansion == pytest.approx((1.0,))
    assert costs[0] == pytest.approx(
        {
            32: 0.6,
            48: 0.2,
            64: 0.0,
            80: -0.05,
            96: -0.1,
            112: -0.15,
            128: -0.2,
        }
    )


def test_local_error_curves_default_to_heldout_factor_dtype_mse() -> None:
    def result(rank: int, fit: tuple[float, float], heldout: tuple[float, float]):
        return {
            "records": [
                {
                    "fit_config": {"cache_rank_per_head": rank},
                    "fit": {"factor_dtype_relative_mse": fit[layer]},
                    "heldout": {"factor_dtype_relative_mse": heldout[layer]},
                }
                for layer in range(2)
            ]
        }

    results = {
        32: result(32, (0.4, 0.5), (0.6, 0.7)),
        64: result(64, (0.2, 0.3), (0.25, 0.35)),
    }
    curves = local_error_curves_from_factor_results(
        results,
        candidate_ranks=(32, 64, 128),
        num_layers=2,
    )

    assert curves == (
        {32: 0.6, 64: 0.25, 128: 0.0},
        {32: 0.7, 64: 0.35, 128: 0.0},
    )


def test_external_schedule_is_authenticated_and_budget_checked(tmp_path) -> None:
    evaluator.activate_model_profile("qwen3_8b")
    try:
        allocation_result = tmp_path / "result.json"
        allocation_result.write_text("{}\n", encoding="utf-8")
        digest = hashlib.sha256(allocation_result.read_bytes()).hexdigest()
        schedule = [[64] * 8 for _ in range(36)]
        payload = {
            "format": evaluator.EXTERNAL_SCHEDULE_FORMAT,
            "status": "complete",
            "schedule_name": "factorized_test",
            "schedule": schedule,
            "accounting": {
                "source_rank_sum": 36 * 8 * 64,
                "source_rank_histogram": {"64": 36 * 8},
            },
            "source": {"allocation_result_sha256": digest},
        }
        path = tmp_path / "schedule.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        result = {
            "selection": {
                "candidate_ranks": [32, 48, 64, 80, 96, 112, 128],
                "target_source_rank_sum": 36 * 8 * 64,
            },
            "schedules": {
                "uniform_anchor": {},
                "mean_dp": {},
                "ucb_dp": {},
            },
        }

        name, row, loaded = evaluator._load_external_layer_schedule(
            path,
            allocation_result_path=allocation_result,
            result=result,
        )
        assert name == "factorized_test"
        assert row["schedule"] == schedule
        assert loaded == payload

        payload["accounting"]["source_rank_sum"] -= 1
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match="accounting violates"):
            evaluator._load_external_layer_schedule(
                path,
                allocation_result_path=allocation_result,
                result=result,
            )
    finally:
        evaluator.activate_model_profile("qwen3_32b")
