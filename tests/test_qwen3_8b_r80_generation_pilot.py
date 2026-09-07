from __future__ import annotations

from evaluation.eval_qwen3_8b_r80_generation_pilot import (
    EXPECTED_ARMS,
    _stage_matches,
    clear_model_generation_length_override,
    summarize_task,
)


def test_uniform_c1_arm_is_part_of_r80_pilot() -> None:
    assert EXPECTED_ARMS["Q3-8B-C1U-R80"] == {
        "method": "c1-uniform",
        "equivalent_rank_target": 80,
    }


def test_summarize_ifeval_metrics() -> None:
    evaluation = {
        "results": {
            "ifeval": {
                "prompt_level_strict_acc,none": 0.4,
                "inst_level_strict_acc,none": 0.5,
                "prompt_level_loose_acc,none": 0.6,
                "inst_level_loose_acc,none": 0.7,
            }
        }
    }
    assert summarize_task("ifeval", evaluation) == {
        "prompt_level_strict_accuracy": 0.4,
        "instruction_level_strict_accuracy": 0.5,
        "prompt_level_loose_accuracy": 0.6,
        "instruction_level_loose_accuracy": 0.7,
    }


def test_summarize_gsm8k_metrics() -> None:
    evaluation = {
        "results": {
            "gsm8k": {
                "exact_match,strict-match": 0.3,
                "exact_match,flexible-extract": 0.35,
            }
        }
    }
    assert summarize_task("gsm8k", evaluation) == {
        "strict_exact_match": 0.3,
        "flexible_exact_match": 0.35,
    }


def test_stage_resume_requires_matching_limit_and_checkpoint() -> None:
    payload = {
        "format": "basisserve.qwen3_8b.r80_generation_pilot.task.v1",
        "status": "complete",
        "run_id": "Q3-8B-C1-R80",
        "task": "ifeval",
        "checkpoint": {"manifest_sha256": "abc"},
        "protocol": {"limit": None, "max_gen_toks": 1280},
    }
    assert _stage_matches(
        payload,
        run_id="Q3-8B-C1-R80",
        task="ifeval",
        checkpoint_manifest_sha256="abc",
        limit=None,
    )
    assert not _stage_matches(
        payload,
        run_id="Q3-8B-C1-R80",
        task="ifeval",
        checkpoint_manifest_sha256="abc",
        limit=0.1,
    )


def test_model_generation_length_override_is_removed() -> None:
    class _GenerationConfig:
        max_new_tokens = 2048

    class _Model:
        generation_config = _GenerationConfig()

    model = _Model()
    assert clear_model_generation_length_override(model) == 2048
    assert model.generation_config.max_new_tokens is None
