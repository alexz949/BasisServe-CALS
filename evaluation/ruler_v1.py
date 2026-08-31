"""Small, dependency-free helpers for the official RULER-v1 protocol."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class RulerTask:
    name: str
    family: str
    tokens_to_generate: int
    match_type: str


RULER_TASKS = (
    RulerTask("niah_single_1", "niah", 128, "all"),
    RulerTask("niah_single_2", "niah", 128, "all"),
    RulerTask("niah_single_3", "niah", 128, "all"),
    RulerTask("niah_multikey_1", "niah", 128, "all"),
    RulerTask("niah_multikey_2", "niah", 128, "all"),
    RulerTask("niah_multikey_3", "niah", 128, "all"),
    RulerTask("niah_multivalue", "niah", 128, "all"),
    RulerTask("niah_multiquery", "niah", 128, "all"),
    RulerTask("vt", "variable_tracking", 30, "all"),
    RulerTask("cwe", "common_words_extraction", 120, "all"),
    RulerTask("fwe", "freq_words_extraction", 50, "all"),
    RulerTask("qa_1", "qa", 32, "part"),
    RulerTask("qa_2", "qa", 32, "part"),
)

TASK_BY_NAME = {task.name: task for task in RULER_TASKS}


def parse_tasks(raw: str) -> tuple[RulerTask, ...]:
    """Resolve ``all`` or a comma-separated ordered RULER task subset."""

    if raw == "all":
        return RULER_TASKS
    names = tuple(name.strip() for name in raw.split(",") if name.strip())
    if not names or len(set(names)) != len(names):
        raise ValueError("RULER tasks must be a nonempty unique list")
    unknown = set(names) - set(TASK_BY_NAME)
    if unknown:
        raise ValueError(f"unknown RULER tasks: {sorted(unknown)}")
    return tuple(TASK_BY_NAME[name] for name in names)


def load_task_records(
    data_dir: Path,
    task: RulerTask,
    samples: int,
) -> list[dict[str, Any]]:
    """Load and validate one official generated RULER JSONL file."""

    path = data_dir / task.name / "validation.jsonl"
    if samples <= 0:
        raise ValueError("samples per task must be positive")
    if not path.is_file():
        raise FileNotFoundError(path)
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from error
            if not isinstance(record.get("input"), str):
                raise TypeError(f"RULER input is not text at {path}:{line_number}")
            outputs = record.get("outputs")
            if (
                not isinstance(outputs, list)
                or not outputs
                or not all(isinstance(output, str) for output in outputs)
            ):
                raise TypeError(
                    f"RULER outputs must be nonempty strings at {path}:{line_number}"
                )
            if not isinstance(record.get("answer_prefix", ""), str):
                raise TypeError(
                    f"RULER answer prefix is not text at {path}:{line_number}"
                )
            records.append(record)
    if len(records) < samples:
        raise ValueError(
            f"RULER task {task.name} has {len(records)} rows, requested {samples}"
        )
    return records[:samples]


def ruler_prompt(record: Mapping[str, Any]) -> str:
    """Return the official base-model completion prompt."""

    return str(record["input"]) + str(record.get("answer_prefix", ""))


def sample_score(
    prediction: str,
    references: Sequence[str],
    match_type: str,
) -> float:
    """Match NVIDIA RULER's case-insensitive substring metrics."""

    if not references:
        raise ValueError("RULER references must be nonempty")
    normalized = prediction.strip().lower()
    matches = [reference.lower() in normalized for reference in references]
    if match_type == "all":
        return sum(matches) / len(matches)
    if match_type == "part":
        return float(any(matches))
    raise ValueError(f"unsupported RULER match type: {match_type}")


def summarize_arm(
    records: Iterable[Mapping[str, Any]],
    arm: str,
    tasks: Sequence[RulerTask],
) -> dict[str, Any]:
    """Compute task-balanced RULER accuracy for one arm."""

    materialized = list(records)
    by_task = {}
    for task in tasks:
        scores = [
            float(record["arms"][arm]["score"])
            for record in materialized
            if record["task"] == task.name
        ]
        if not scores:
            raise ValueError(f"no completed RULER rows for task {task.name}")
        by_task[task.name] = {
            "samples": len(scores),
            "accuracy": sum(scores) / len(scores),
        }
    return {
        "arm": arm,
        "task_balanced_accuracy": sum(
            row["accuracy"] for row in by_task.values()
        )
        / len(by_task),
        "tasks": by_task,
    }


def paired_summary(
    records: Iterable[Mapping[str, Any]],
    dense_arm: str,
    sparse_arm: str,
    tasks: Sequence[RulerTask],
) -> dict[str, Any]:
    """Count paired sparse improvements and regressions."""

    materialized = list(records)

    def summarize_subset(subset: list[Mapping[str, Any]]) -> dict[str, Any]:
        dense_scores = [float(row["arms"][dense_arm]["score"]) for row in subset]
        sparse_scores = [float(row["arms"][sparse_arm]["score"]) for row in subset]
        deltas = [
            sparse - dense
            for dense, sparse in zip(dense_scores, sparse_scores, strict=True)
        ]
        return {
            "samples": len(subset),
            "dense_mean": sum(dense_scores) / len(subset),
            "sparse_mean": sum(sparse_scores) / len(subset),
            "mean_delta": sum(deltas) / len(subset),
            "sparse_improvements": sum(delta > 0 for delta in deltas),
            "sparse_regressions": sum(delta < 0 for delta in deltas),
            "ties": sum(delta == 0 for delta in deltas),
        }

    by_task = {}
    for task in tasks:
        subset = [row for row in materialized if row["task"] == task.name]
        if not subset:
            raise ValueError(f"no paired RULER rows for task {task.name}")
        by_task[task.name] = summarize_subset(subset)
    return {"all_samples": summarize_subset(materialized), "tasks": by_task}


__all__ = [
    "RULER_TASKS",
    "RulerTask",
    "load_task_records",
    "paired_summary",
    "parse_tasks",
    "ruler_prompt",
    "sample_score",
    "summarize_arm",
]
