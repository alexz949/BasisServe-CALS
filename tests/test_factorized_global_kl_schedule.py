from __future__ import annotations

import hashlib
import json

import pytest

from evaluation.build_qwen3_8b_c1_factorized_global_kl_schedule import (
    allocate_layer_schedule,
    predict_factorized_costs,
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


def test_one_probe_factorization_rejects_nonpositive_sensitivity() -> None:
    with pytest.raises(ValueError, match="sensitivity is not positive"):
        predict_factorized_costs(
            ({32: 0.4, 64: 0.2, 96: 0.1, 128: 0.0},),
            (0.1,),
            candidate_ranks=(32, 64, 96, 128),
            anchor_rank=64,
            probe_rank=96,
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
