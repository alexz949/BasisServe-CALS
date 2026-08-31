"""Build lm-eval task specifications with optional per-task few-shot counts."""

from __future__ import annotations

from typing import Any, Sequence


def parse_task_fewshot(
    raw: str,
    tasks: Sequence[str],
) -> dict[str, int]:
    """Parse ``task=count`` entries and default unmentioned tasks to zero."""

    selected = set(tasks)
    parsed: dict[str, int] = {}
    entries = [entry.strip() for entry in raw.split(",") if entry.strip()]
    if not entries:
        raise ValueError("--task-fewshot must contain at least one task=count entry")
    for entry in entries:
        if entry.count("=") != 1:
            raise ValueError(
                f"invalid --task-fewshot entry {entry!r}; expected task=count"
            )
        task, count_raw = (part.strip() for part in entry.split("=", 1))
        if not task or task not in selected:
            raise ValueError(
                f"--task-fewshot task {task!r} is absent from --tasks"
            )
        if task in parsed:
            raise ValueError(f"--task-fewshot contains duplicate task {task!r}")
        try:
            count = int(count_raw)
        except ValueError as error:
            raise ValueError(
                f"invalid few-shot count {count_raw!r} for task {task!r}"
            ) from error
        if count < 0:
            raise ValueError("few-shot counts must be non-negative")
        parsed[task] = count
    return {task: parsed.get(task, 0) for task in tasks}


def task_evaluation_specifications(
    tasks: Sequence[str],
    raw_task_fewshot: str | None,
) -> tuple[list[str | dict[str, Any]], int | None, dict[str, int] | None]:
    """Return lm-eval task inputs, its global override, and protocol metadata."""

    if raw_task_fewshot is None:
        return list(tasks), 0, None
    task_num_fewshot = parse_task_fewshot(raw_task_fewshot, tasks)
    task_specs: list[str | dict[str, Any]] = [
        {"task": task, "num_fewshot": task_num_fewshot[task]}
        for task in tasks
    ]
    return task_specs, None, task_num_fewshot


__all__ = ["parse_task_fewshot", "task_evaluation_specifications"]
