from __future__ import annotations

import pytest

from evaluation.ruler_v1 import (
    RULER_TASKS,
    paired_summary,
    parse_tasks,
    ruler_prompt,
    sample_score,
    summarize_arm,
)


def test_parse_tasks_preserves_official_order() -> None:
    assert parse_tasks("all") == RULER_TASKS
    assert [task.name for task in parse_tasks("qa_2,niah_single_1")] == [
        "qa_2",
        "niah_single_1",
    ]
    with pytest.raises(ValueError, match="unknown"):
        parse_tasks("niah_single_1,missing")


def test_base_prompt_keeps_official_answer_prefix_spacing() -> None:
    record = {"input": "Question?", "answer_prefix": " Answer:"}
    assert ruler_prompt(record) == "Question? Answer:"


def test_ruler_all_and_part_substring_scores_match_official_definition() -> None:
    references = ["Alpha", "Beta"]
    assert sample_score("alpha and beta", references, "all") == 1.0
    assert sample_score("Only ALPHA", references, "all") == 0.5
    assert sample_score("Only beta", references, "part") == 1.0
    assert sample_score("neither", references, "part") == 0.0


def test_task_balanced_and_paired_summaries() -> None:
    tasks = parse_tasks("niah_single_1,qa_1")
    records = [
        {
            "task": "niah_single_1",
            "arms": {"dense": {"score": 1.0}, "sparse": {"score": 0.5}},
        },
        {
            "task": "niah_single_1",
            "arms": {"dense": {"score": 0.0}, "sparse": {"score": 1.0}},
        },
        {
            "task": "qa_1",
            "arms": {"dense": {"score": 1.0}, "sparse": {"score": 1.0}},
        },
    ]
    dense = summarize_arm(records, "dense", tasks)
    assert dense["tasks"]["niah_single_1"]["accuracy"] == 0.5
    assert dense["tasks"]["qa_1"]["accuracy"] == 1.0
    assert dense["task_balanced_accuracy"] == 0.75
    paired = paired_summary(records, "dense", "sparse", tasks)
    assert paired["all_samples"]["sparse_improvements"] == 1
    assert paired["all_samples"]["sparse_regressions"] == 1
    assert paired["all_samples"]["ties"] == 1
