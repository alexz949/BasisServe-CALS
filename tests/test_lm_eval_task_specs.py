from __future__ import annotations

import pytest

from evaluation.lm_eval_task_specs import (
    parse_task_fewshot,
    task_evaluation_specifications,
)


def test_default_protocol_preserves_global_zero_shot() -> None:
    tasks, global_num_fewshot, task_num_fewshot = (
        task_evaluation_specifications(["boolq", "piqa"], None)
    )
    assert tasks == ["boolq", "piqa"]
    assert global_num_fewshot == 0
    assert task_num_fewshot is None


def test_mixed_task_fewshot_defaults_unmentioned_tasks_to_zero() -> None:
    names = ["gsm8k", "mathqa", "ifeval", "mmlu"]
    tasks, global_num_fewshot, task_num_fewshot = (
        task_evaluation_specifications(names, "gsm8k=5,mmlu=5")
    )
    assert global_num_fewshot is None
    assert task_num_fewshot == {
        "gsm8k": 5,
        "mathqa": 0,
        "ifeval": 0,
        "mmlu": 5,
    }
    assert tasks == [
        {"task": "gsm8k", "num_fewshot": 5},
        {"task": "mathqa", "num_fewshot": 0},
        {"task": "ifeval", "num_fewshot": 0},
        {"task": "mmlu", "num_fewshot": 5},
    ]


@pytest.mark.parametrize(
    "raw,match",
    (
        ("gsm8k", "task=count"),
        ("gsm8k=five", "few-shot count"),
        ("gsm8k=-1", "non-negative"),
        ("piqa=1", "absent from --tasks"),
        ("gsm8k=5,gsm8k=0", "duplicate"),
    ),
)
def test_task_fewshot_parser_rejects_invalid_mappings(raw: str, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        parse_task_fewshot(raw, ["gsm8k"])
