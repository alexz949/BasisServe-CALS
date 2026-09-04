from __future__ import annotations

import argparse
import json

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
from evaluation.merge_iclr_quality_shards import merge as merge_quality_shards
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
        activate_experiment_profile("qwen3_32b")
        assert run_id("dense", None, 1) == "Q3-32B-Dense"
        assert run_id("weight-svd", 64, 1) == "Q3-32B-SVD-R64"
        activate_experiment_profile("llama31_70b")
        assert run_id("palu-fisher", 96, 4) == "L31-70B-PALUG4-R96"
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


def test_commonsense_summary_accepts_a_task_shard() -> None:
    tasks = ("hellaswag", "piqa")
    summarized = summarize_commonsense(
        {
            "results": {
                "hellaswag": {"acc_norm,none": 0.6},
                "piqa": {"acc,none": 0.7},
            }
        },
        tasks,
    )
    assert summarized is not None
    rows, average = summarized
    assert [row["task"] for row in rows] == list(tasks)
    assert average == pytest.approx(0.65)


def test_parallel_quality_shards_merge_into_the_canonical_result(tmp_path) -> None:
    from evaluation import eval_qwen3_8b_iclr_quality as quality

    model_dir = tmp_path / "model"
    checkpoint_dir = tmp_path / "checkpoint"
    output_dir = tmp_path / "quality"
    model_dir.mkdir()
    checkpoint_dir.mkdir()
    output_dir.mkdir()
    (model_dir / "config.json").write_text("{}\n", encoding="utf-8")
    run_id = "Q3-32B-Dense"
    try:
        activate_quality_profile("qwen3_32b")
        manifest = {
            "format": quality.CHECKPOINT_FORMAT,
            "status": "complete",
            "run_id": run_id,
            "model": {
                "config_sha256": quality._sha256(model_dir / "config.json"),
            },
            "compression": {
                "method": "dense",
                "method_label": "Dense",
                "realized_retained_v_ratio": 1.0,
            },
        }
        manifest_path = checkpoint_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        manifest_sha256 = quality._sha256(manifest_path)
        checkpoint = {
            "directory": str(checkpoint_dir),
            "manifest_sha256": manifest_sha256,
            "artifact_sha256": None,
            "format": quality.CHECKPOINT_FORMAT,
        }
        base = {
            "status": "complete",
            "run_id": run_id,
            "checkpoint": checkpoint,
            "compression": manifest["compression"],
        }
        for stage, ppl in (("wikitext2", 7.0), ("c4", 9.0)):
            payload = {
                **base,
                "format": quality.STAGE_FORMATS[stage],
                "metrics": {"ppl": ppl},
            }
            (output_dir / f"{stage}.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
        task_shards = ((TASKS[0],), TASKS[1:])
        for index, tasks in enumerate(task_shards):
            rows = [
                {"task": task, "metric": "acc", "value": 0.5 + offset / 100}
                for offset, task in enumerate(tasks)
            ]
            payload = {
                **base,
                "format": quality.STAGE_FORMATS["commonsense"],
                "protocol": {
                    "tasks": list(tasks),
                    "quality_shard_index": index,
                    "quality_shard_count": 2,
                },
                "task_accuracy": rows,
                "evaluation": {
                    "results": {task: {"acc,none": 0.5} for task in tasks}
                },
                "runtime": {"quality_shard_index": index},
                "environment": {
                    "cuda_devices": ["NVIDIA L40S", "NVIDIA L40S"],
                    "slurm_job_id": str(index),
                },
                "elapsed_seconds": float(index + 1),
            }
            (output_dir / f"commonsense-shard-{index:02d}.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
        args = argparse.Namespace(
            profile="qwen3_32b",
            run_id=run_id,
            model=str(model_dir),
            checkpoint_dir=str(checkpoint_dir),
            output_dir=str(output_dir),
            shard_count=2,
            gpus_per_shard=2,
            lm_eval_batch_size=8,
        )
        assert merge_quality_shards(args) == 0
        result = json.loads((output_dir / "result.json").read_text(encoding="utf-8"))
        assert result["status"] == "complete"
        assert result["environment"]["evaluation_layout"] == (
            "parallel_quality_shards"
        )
        assert [row["task"] for row in result["metrics"]["task_accuracy"]] == list(
            TASKS
        )
    finally:
        activate_quality_profile("qwen3_8b")


def test_quality_profiles_pin_model_specific_formats() -> None:
    from evaluation import eval_qwen3_8b_iclr_quality as quality

    try:
        activate_quality_profile("llama31_8b")
        assert quality.CHECKPOINT_FORMAT == "basisserve.llama31_8b.iclr_v_factors.v1"
        assert quality.FORMAT == "basisserve.llama31_8b.iclr_quality.v1"
        assert quality.STAGE_FORMATS["c4"] == (
            "basisserve.llama31_8b.iclr_quality.c4_validation.v1"
        )
        activate_quality_profile("qwen3_32b")
        assert quality.CHECKPOINT_FORMAT == "basisserve.qwen3_32b.iclr_v_factors.v1"
        assert quality.FORMAT == "basisserve.qwen3_32b.iclr_quality.v1"
        activate_quality_profile("llama31_70b")
        assert quality.CHECKPOINT_FORMAT == "basisserve.llama31_70b.iclr_v_factors.v1"
        assert quality.FORMAT == "basisserve.llama31_70b.iclr_quality.v1"
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
    for matrix_run_id in _run_ids(prefix):
        if matrix_run_id.endswith("-Dense"):
            metrics = (7.0, 9.0, 0.70)
        elif "-SVD-" in matrix_run_id:
            metrics = (100.0, 120.0, 0.40)
        elif "-PALUM-" in matrix_run_id:
            metrics = (12.0, 14.0, 0.60)
        elif "-PALUG2-" in matrix_run_id:
            metrics = (10.0, 12.0, 0.63)
        else:
            metrics = (8.0, 10.0, 0.67)
        results[matrix_run_id] = {
            "metrics": {
                "wikitext2_ppl": metrics[0],
                "c4_validation_128_ppl": metrics[1],
                "average_accuracy": metrics[2],
            }
        }

    checks = _trend_checks(results, prefix)
    assert checks
    assert all(checks.values())
