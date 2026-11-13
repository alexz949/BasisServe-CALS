#!/usr/bin/env python3
"""Summarize one model's nominal-V64 calibration-seed sensitivity experiment.

Two kinds of uncertainty are reported separately.

A. Calibration-seed variation: mean and sample SD (ddof=1) of each metric over
   the fitted calibration seeds. Dense is one fixed reference, not a seed.
B. MCQ test-sample uncertainty: a task-stratified percentile bootstrap over
   evaluation examples. One draw of example indices is shared by every method
   and calibration seed, so method differences are paired. The intervals are
   conditional on the fitted models; they are not calibration-seed intervals.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


TASKS = (
    "arc_easy",
    "arc_challenge",
    "hellaswag",
    "piqa",
    "winogrande",
    "boolq",
    "openbookqa",
)
TASK_LABELS = {
    "arc_easy": "ARC-E",
    "arc_challenge": "ARC-C",
    "hellaswag": "HellaS.",
    "piqa": "PIQA",
    "winogrande": "WinoG.",
    "boolq": "BoolQ",
    "openbookqa": "OBQA",
}
METHODS = {
    "basiskv": "BasisKV",
    "palu_m": "PaLU M-LRD",
    "palu_g4": "PaLU G-LRD4",
}
COMPARISONS = (("basiskv", "palu_m"), ("basiskv", "palu_g4"))
# Tasks whose examples carry an explicit shared source; all other tasks are
# resampled by example.
RESAMPLING_UNITS = {
    "boolq": "passage",
    "hellaswag": "source_id",
}


def _check(condition: bool, message: str) -> bool:
    if condition:
        return True
    print(f"[Error] {message}", file=sys.stderr, flush=True)
    return False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resampling_unit(record: Mapping[str, Any]) -> str:
    kind = RESAMPLING_UNITS.get(str(record["task"]), "example")
    if kind == "passage":
        return str(record["passage_sha256"])
    if kind == "source_id":
        return str(record["native"]["source_id"])
    return str(record["doc_id"])


def load_examples(path: Path) -> dict[str, list[dict[str, Any]]]:
    """Read one arm's per-example records, ordered by example within each task."""

    by_task: dict[str, list[dict[str, Any]]] = {task: [] for task in TASKS}
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        by_task[record["task"]].append(record)
    for records in by_task.values():
        records.sort(key=lambda record: int(record["doc_id"]))
    return by_task


def paired_examples(arms: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]]) -> bool:
    """Every arm must score the same examples, identified by ID and content hash."""

    reference_name = next(iter(arms))
    reference = arms[reference_name]
    paired = True
    for name, examples in arms.items():
        for task in TASKS:
            same = [
                (record["doc_id"], record["doc_hash"], record["metric"])
                for record in examples[task]
            ] == [
                (record["doc_id"], record["doc_hash"], record["metric"])
                for record in reference[task]
            ]
            paired = _check(
                same and len(examples[task]) > 0,
                f"{name}/{task} examples do not pair with {reference_name}",
            ) and paired
    return paired


def unit_tables(
    arms: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, np.ndarray]]]:
    """Per task: examples per resampling unit, and each arm's correct count per unit."""

    reference = next(iter(arms.values()))
    sizes: dict[str, np.ndarray] = {}
    correct: dict[str, dict[str, np.ndarray]] = {name: {} for name in arms}
    for task in TASKS:
        units = [resampling_unit(record) for record in reference[task]]
        index = {unit: position for position, unit in enumerate(dict.fromkeys(units))}
        membership = np.array([index[unit] for unit in units])
        sizes[task] = np.bincount(membership, minlength=len(index)).astype(np.float64)
        for name, examples in arms.items():
            scores = np.array([float(record["correct"]) for record in examples[task]])
            correct[name][task] = np.bincount(
                membership, weights=scores, minlength=len(index)
            )
    return sizes, correct


def bootstrap_macro(
    sizes: Mapping[str, np.ndarray],
    correct: Mapping[str, Mapping[str, np.ndarray]],
    *,
    replicates: int,
    seed: int,
    chunk: int = 250,
) -> dict[str, np.ndarray]:
    """Task-stratified bootstrap of the equal-weight macro accuracy of every arm.

    Units are drawn with replacement inside each task, keeping the number of
    units fixed. The same draw scores every arm.
    """

    generator = np.random.default_rng(seed)
    macro = {name: np.zeros(replicates) for name in correct}
    for task in TASKS:
        unit_count = len(sizes[task])
        probabilities = np.full(unit_count, 1.0 / unit_count)
        for start in range(0, replicates, chunk):
            stop = min(start + chunk, replicates)
            counts = generator.multinomial(
                unit_count, probabilities, size=stop - start
            ).astype(np.float64)
            examples = counts @ sizes[task]
            for name in correct:
                macro[name][start:stop] += (counts @ correct[name][task]) / examples
    return {name: values / len(TASKS) for name, values in macro.items()}


def interval(values: np.ndarray) -> list[float]:
    low, high = np.percentile(values, [2.5, 97.5])
    return [float(low), float(high)]


def point_macro(
    sizes: Mapping[str, np.ndarray], correct: Mapping[str, np.ndarray]
) -> tuple[dict[str, float], float]:
    accuracy = {
        task: float(correct[task].sum() / sizes[task].sum()) for task in TASKS
    }
    return accuracy, sum(accuracy.values()) / len(TASKS)


def _mean_sd(values: Sequence[float]) -> dict[str, float]:
    array = np.array(values, dtype=np.float64)
    return {"mean": float(array.mean()), "sd": float(array.std(ddof=1))}


def _layer_ranks(compression: Mapping[str, Any]) -> list[int]:
    """One rank per layer: per physical head for BasisKV/M-LRD, per group for G-LRD."""

    return [int(ranks[0]) for ranks in compression["layer_ranks"]]


def _arm(root: Path, relative: str, method: str, seed: int | None) -> dict[str, Any] | None:
    quality = root / relative / "quality"
    result_path = quality / "result.json"
    if not _check(result_path.is_file(), f"missing result: {result_path}"):
        return None
    result = _read_json(result_path)
    examples_path = quality / str(result["commonsense_examples"]["file"])
    manifest_path = root / relative / "checkpoint" / "manifest.json"
    manifest = _read_json(manifest_path)
    consistent = all(
        (
            _check(result.get("status") == "complete", f"incomplete result: {result_path}"),
            _check(
                _sha256(examples_path) == result["commonsense_examples"]["sha256"],
                f"example-record hash mismatch: {examples_path}",
            ),
            _check(
                result["checkpoint"]["manifest_sha256"] == _sha256(manifest_path),
                f"result was evaluated on another checkpoint: {result_path}",
            ),
        )
    )
    if not consistent:
        return None
    compression = manifest["compression"]
    return {
        "name": relative,
        "method": method,
        "seed": seed,
        "result": result,
        "manifest": manifest,
        "manifest_sha256": _sha256(manifest_path),
        "examples": load_examples(examples_path),
        "examples_sha256": result["commonsense_examples"]["sha256"],
        "retained_v": float(compression["realized_retained_v_ratio"]),
        "layer_ranks": [] if method == "dense" else _layer_ranks(compression),
        "wikitext2_ppl": float(result["metrics"]["wikitext2_ppl"]),
        "c4_ppl": float(result["metrics"]["c4_validation_128_ppl"]),
        "task_accuracy": {
            row["task"]: float(row["value"]) for row in result["metrics"]["task_accuracy"]
        },
        "macro": float(result["metrics"]["average_accuracy"]),
    }


def _calibration_provenance(root: Path, seed: int) -> dict[str, Any]:
    calib = root / "calib" / f"s{seed}"
    record: dict[str, Any] = {"seed": seed}
    for name in ("windows_fit", "windows_fit_heldout", "windows_kl"):
        manifest = _read_json(calib / name / "manifest.json")
        record[name] = {
            "sampling_seed": manifest["sampling"]["seed"],
            "split": manifest["dataset"]["split"],
            "shape": manifest["artifact"]["shape"],
            "sha256": manifest["artifact"]["sha256"],
            "document_ids": [row["document_id"] for row in manifest["records"]],
        }
    fit = set(record["windows_fit_heldout"]["document_ids"])
    profile = record["windows_kl"]["document_ids"][320:464]
    record["kl_windows_320_463_documents_shared_with_fit_heldout"] = len(fit & set(profile))
    for name in ("windows_fit", "windows_fit_heldout", "windows_kl"):
        record[name]["document_ids_sha256"] = hashlib.sha256(
            "\n".join(record[name].pop("document_ids")).encode("utf-8")
        ).hexdigest()
    record["covariance_snapshot_manifest_sha256"] = _sha256(calib / "cov" / "manifest.json")
    record["whitening_sha256"] = _read_json(calib / "whitening" / "manifest.json")["artifact"]["sha256"]
    record["fisher_sha256"] = _sha256(calib / "fisher" / "fisher.json")
    return record


def _pp(value: float) -> str:
    return f"{100.0 * value:+.2f}"


def _pct(value: float) -> str:
    return f"{100.0 * value:.2f}"


def _markdown(summary: Mapping[str, Any], arms: Sequence[Mapping[str, Any]], notes: str) -> str:
    seeds = summary["calibration_seeds"]
    lines = [
        f"# {summary['model'].split('/')[-1]} nominal V64: calibration-seed sensitivity and MCQ confidence intervals",
        "",
        f"Calibration seeds: {', '.join(map(str, seeds))} (C4 train windows resampled per seed; "
        "WikiText-2, the C4-validation PPL bank and the MCQ test sets are fixed). "
        "Dense is evaluated once as a fixed reference.",
        "",
        "## 1. Mean ± sample SD over calibration seeds",
        "",
        "SD uses ddof=1 over the fitted seeds and describes the observed calibration variation only.",
        "",
        "| Method | Retained V | WT2 PPL | C4 PPL | MCQ macro (%) | "
        + " | ".join(TASK_LABELS[task] for task in TASKS)
        + " |",
        "| --- | ---: | ---: | ---: | ---: | " + " | ".join("---:" for _ in TASKS) + " |",
    ]
    dense = next(arm for arm in arms if arm["method"] == "dense")
    lines.append(
        f"| Dense (fixed reference) | 1.0000 | {dense['wikitext2_ppl']:.3f} | {dense['c4_ppl']:.3f} | "
        f"{_pct(dense['macro'])} | "
        + " | ".join(_pct(dense["task_accuracy"][task]) for task in TASKS)
        + " |"
    )
    for method, label in METHODS.items():
        stats = summary["seed_variation"][method]
        lines.append(
            f"| {label} | {stats['retained_v']['mean']:.4f} | "
            f"{stats['wikitext2_ppl']['mean']:.3f} ± {stats['wikitext2_ppl']['sd']:.3f} | "
            f"{stats['c4_ppl']['mean']:.3f} ± {stats['c4_ppl']['sd']:.3f} | "
            f"{_pct(stats['macro']['mean'])} ± {_pct(stats['macro']['sd'])} | "
            + " | ".join(
                f"{_pct(stats['task_accuracy'][task]['mean'])} ± {_pct(stats['task_accuracy'][task]['sd'])}"
                for task in TASKS
            )
            + " |"
        )
    lines += [
        "",
        "## 2. Per-seed results",
        "",
        "| Method | Seed | Retained V | WT2 PPL | C4 PPL | MCQ macro (%) | 95% CI (test samples) | "
        + " | ".join(TASK_LABELS[task] for task in TASKS)
        + " |",
        "| --- | ---: | ---: | ---: | ---: | ---: | :---: | " + " | ".join("---:" for _ in TASKS) + " |",
    ]
    for arm in arms:
        low, high = summary["macro_ci"][arm["name"]]["ci95"]
        label = "Dense" if arm["method"] == "dense" else METHODS[arm["method"]]
        lines.append(
            f"| {label} | {arm['seed'] if arm['seed'] is not None else '-'} | {arm['retained_v']:.4f} | "
            f"{arm['wikitext2_ppl']:.4f} | {arm['c4_ppl']:.4f} | {_pct(arm['macro'])} | "
            f"[{_pct(low)}, {_pct(high)}] | "
            + " | ".join(_pct(arm["task_accuracy"][task]) for task in TASKS)
            + " |"
        )
    lines += ["", "Layer ranks (one per layer; per physical KV head for BasisKV and M-LRD, per 4-head group for G-LRD4):", ""]
    for arm in arms:
        if arm["method"] != "dense":
            lines.append(f"- {METHODS[arm['method']]} s{arm['seed']}: `{arm['layer_ranks']}`")
    bootstrap = summary["bootstrap"]
    lines += [
        "",
        "## 3. MCQ test-sample confidence intervals",
        "",
        f"Task-stratified percentile bootstrap, {bootstrap['replicates']} replicates, seed {bootstrap['seed']}. "
        "Each replicate redraws the resampling units of every task with replacement (unit count unchanged), "
        "computes the seven task accuracies and averages them with equal weight. The same draw is applied to "
        "every method and calibration seed.",
        "",
        "| Task | Resampling unit | Units | Examples |",
        "| --- | --- | ---: | ---: |",
    ]
    for task in TASKS:
        unit = bootstrap["units"][task]
        lines.append(f"| {TASK_LABELS[task]} | {unit['kind']} | {unit['units']} | {unit['examples']} |")
    lines += [
        "",
        "Paired differences in MCQ macro accuracy (percentage points):",
        "",
        "| Comparison | Seed | Difference (pp) | 95% CI (pp) |",
        "| --- | ---: | ---: | :---: |",
    ]
    for left, right in COMPARISONS:
        for seed in seeds:
            row = summary["paired_differences"][f"{left}-{right}"][str(seed)]
            lines.append(
                f"| {METHODS[left]} − {METHODS[right]} | {seed} | {_pp(row['difference'])} | "
                f"[{_pp(row['ci95'][0])}, {_pp(row['ci95'][1])}] |"
            )
    lines += [
        "",
        "Three-seed averages. Each replicate first scores every seed, then averages the seeds. "
        "**These intervals are test-sample uncertainty conditional on the three fitted models of each method; "
        "they are not confidence intervals over the population of calibration seeds.**",
        "",
        "| Quantity | Value | 95% CI |",
        "| --- | ---: | :---: |",
    ]
    for method, label in METHODS.items():
        row = summary["seed_average"]["macro"][method]
        lines.append(
            f"| {label} MCQ macro (%) | {_pct(row['value'])} | [{_pct(row['ci95'][0])}, {_pct(row['ci95'][1])}] |"
        )
    for left, right in COMPARISONS:
        row = summary["seed_average"]["paired_differences"][f"{left}-{right}"]
        lines.append(
            f"| {METHODS[left]} − {METHODS[right]} (pp) | {_pp(row['difference'])} | "
            f"[{_pp(row['ci95'][0])}, {_pp(row['ci95'][1])}] |"
        )
    low, high = summary["macro_ci"]["dense"]["ci95"]
    lines += ["", f"Dense MCQ macro: {_pct(dense['macro'])} [{_pct(low)}, {_pct(high)}].", ""]
    if notes:
        lines += [notes.rstrip(), ""]
    return "\n".join(lines)


def summarize(args: argparse.Namespace) -> int:
    root = Path(args.root).expanduser().resolve()
    seeds = [int(seed) for seed in args.seeds.split(",")]
    arms = [_arm(root, "dense", "dense", None)]
    for method in METHODS:
        arms += [_arm(root, f"{method}/s{seed}", method, seed) for seed in seeds]
    if any(arm is None for arm in arms):
        return 2
    examples = {arm["name"]: arm["examples"] for arm in arms}
    if not paired_examples(examples):
        return 2
    sizes, correct = unit_tables(examples)
    reproduced = True
    for arm in arms:
        accuracy, macro = point_macro(sizes, correct[arm["name"]])
        reproduced = _check(
            abs(macro - arm["macro"]) < 1.0e-9
            and all(abs(accuracy[task] - arm["task_accuracy"][task]) < 1.0e-9 for task in TASKS),
            f"{arm['name']}: per-example accuracy differs from the evaluator aggregate",
        ) and reproduced
    if not reproduced:
        return 2

    macro = bootstrap_macro(
        sizes, correct, replicates=args.bootstrap_replicates, seed=args.bootstrap_seed
    )
    by_name = {arm["name"]: arm for arm in arms}
    summary: dict[str, Any] = {
        "format": "basisserve.v64_calibration_seed_summary.v1",
        "model": arms[0]["result"]["model"]["huggingface_repo"],
        "calibration_seeds": seeds,
        "bootstrap": {
            "replicates": args.bootstrap_replicates,
            "seed": args.bootstrap_seed,
            "interval": "percentile 2.5/97.5",
            "units": {
                task: {
                    "kind": RESAMPLING_UNITS.get(task, "example"),
                    "units": int(len(sizes[task])),
                    "examples": int(sizes[task].sum()),
                }
                for task in TASKS
            },
        },
        "macro_ci": {
            name: {"value": by_name[name]["macro"], "ci95": interval(values)}
            for name, values in macro.items()
        },
        "seed_variation": {},
        "paired_differences": {},
        "seed_average": {"macro": {}, "paired_differences": {}},
        "arms": {},
    }
    for method in METHODS:
        members = [by_name[f"{method}/s{seed}"] for seed in seeds]
        summary["seed_variation"][method] = {
            "retained_v": _mean_sd([arm["retained_v"] for arm in members]),
            "wikitext2_ppl": _mean_sd([arm["wikitext2_ppl"] for arm in members]),
            "c4_ppl": _mean_sd([arm["c4_ppl"] for arm in members]),
            "macro": _mean_sd([arm["macro"] for arm in members]),
            "task_accuracy": {
                task: _mean_sd([arm["task_accuracy"][task] for arm in members])
                for task in TASKS
            },
        }
        averaged = np.mean([macro[f"{method}/s{seed}"] for seed in seeds], axis=0)
        summary["seed_average"]["macro"][method] = {
            "value": float(np.mean([arm["macro"] for arm in members])),
            "ci95": interval(averaged),
        }
    for left, right in COMPARISONS:
        key = f"{left}-{right}"
        per_seed = {
            seed: macro[f"{left}/s{seed}"] - macro[f"{right}/s{seed}"] for seed in seeds
        }
        summary["paired_differences"][key] = {
            str(seed): {
                "difference": by_name[f"{left}/s{seed}"]["macro"]
                - by_name[f"{right}/s{seed}"]["macro"],
                "ci95": interval(values),
            }
            for seed, values in per_seed.items()
        }
        summary["seed_average"]["paired_differences"][key] = {
            "difference": float(
                np.mean(
                    [
                        by_name[f"{left}/s{seed}"]["macro"]
                        - by_name[f"{right}/s{seed}"]["macro"]
                        for seed in seeds
                    ]
                )
            ),
            "ci95": interval(np.mean(list(per_seed.values()), axis=0)),
        }
    for arm in arms:
        result = arm["result"]
        compression = arm["manifest"]["compression"]
        summary["arms"][arm["name"]] = {
            "method": arm["method"],
            "calibration_seed": arm["seed"],
            "run_id": result["run_id"],
            "command": result["command"],
            "model": result["model"],
            "environment": result["environment"],
            "checkpoint_manifest_sha256": arm["manifest_sha256"],
            "factor_artifact_sha256": result["checkpoint"]["artifact_sha256"],
            "layer_factor_sha256": [
                layer["sha256"] for layer in arm["manifest"].get("layers", ()) if "sha256" in layer
            ],
            "allocation": compression.get("allocation"),
            "layer_ranks": arm["layer_ranks"],
            "rank_sum_across_layers": compression["rank_sum_across_layers"],
            "realized_retained_v_ratio": arm["retained_v"],
            "wikitext2_ppl": arm["wikitext2_ppl"],
            "c4_validation_128_ppl": arm["c4_ppl"],
            "task_accuracy": arm["task_accuracy"],
            "macro_accuracy": arm["macro"],
            "stage_sha256": result["stages"],
            "examples_sha256": arm["examples_sha256"],
        }
    summary["calibration"] = {str(seed): _calibration_provenance(root, seed) for seed in seeds}
    summary["evaluation_inputs"] = {
        "c4_validation_windows_sha256": _sha256(
            root / "eval_inputs" / "c4_validation_128x2048" / "windows.safetensors"
        ),
        "wikitext2_tokens": {
            arm["name"]: _read_json(root / arm["name"] / "quality" / "wikitext2.json")["metrics"]["tokens"]
            for arm in arms
        },
    }

    (root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    np.savez_compressed(root / "bootstrap_macro.npz", **{name.replace("/", "__"): values for name, values in macro.items()})
    with (root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["method", "calibration_seed", "statistic", "retained_v", "wikitext2_ppl", "c4_ppl", "mcq_macro",
             "mcq_macro_ci95_low", "mcq_macro_ci95_high", *TASKS]
        )
        for arm in arms:
            low, high = summary["macro_ci"][arm["name"]]["ci95"]
            writer.writerow(
                [arm["method"], arm["seed"] if arm["seed"] is not None else "", "value", arm["retained_v"],
                 arm["wikitext2_ppl"], arm["c4_ppl"], arm["macro"], low, high,
                 *[arm["task_accuracy"][task] for task in TASKS]]
            )
        for method in METHODS:
            stats = summary["seed_variation"][method]
            average = summary["seed_average"]["macro"][method]["ci95"]
            for statistic in ("mean", "sd"):
                writer.writerow(
                    [method, "", f"seed_{statistic}", stats["retained_v"][statistic],
                     stats["wikitext2_ppl"][statistic], stats["c4_ppl"][statistic], stats["macro"][statistic],
                     average[0] if statistic == "mean" else "", average[1] if statistic == "mean" else "",
                     *[stats["task_accuracy"][task][statistic] for task in TASKS]]
                )
    notes_path = root / "NOTES.md"
    notes = notes_path.read_text(encoding="utf-8") if notes_path.is_file() else ""
    (root / "SUMMARY.md").write_text(_markdown(summary, arms, notes), encoding="utf-8")
    print(f"[Result] wrote {root / 'SUMMARY.md'}", flush=True)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261113)
    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(summarize(parse_args()))
