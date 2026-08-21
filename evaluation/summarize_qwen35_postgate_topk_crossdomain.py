#!/usr/bin/env python3
"""Merge Qwen3.5 WikiText-2/MCQ post-gate Top-K quality results."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import statistics
from typing import Any, Mapping, Sequence

from safetensors.torch import load_file
import torch
from torch import Tensor


FORMAT = "basisserve.qwen35.postgate_topk_crossdomain.v1"
SUMMARY_FORMAT = "basisserve.qwen35.postgate_topk_crossdomain_summary.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-json")
    parser.add_argument("--output-md")
    return parser.parse_args()


def _atomic_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _load_results(root: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    payloads: list[dict[str, Any]] = []
    for path in sorted(root.glob("*/results.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("format") != FORMAT:
            raise ValueError(f"incompatible result {path}")
        payload["_result_path"] = str(path)
        payloads.append(payload)
    ppl = [payload for payload in payloads if payload.get("ppl") is not None]
    if len(ppl) != 1:
        raise ValueError(f"expected exactly one WikiText result, found {len(ppl)}")
    mcq: dict[str, dict[str, Any]] = {}
    for payload in payloads:
        if payload.get("mcq") is None:
            continue
        tasks = payload["mcq"]["dense"]["tasks"]
        if len(tasks) != 1:
            raise ValueError("the merger expects one MCQ task per result directory")
        name = str(tasks[0]["task"])
        if name in mcq:
            raise ValueError(f"duplicate MCQ task {name}")
        mcq[name] = payload
    if not mcq:
        raise ValueError("no MCQ result was found")
    return ppl[0], mcq


def _variant_order(ppl: Mapping[str, Any]) -> tuple[str, ...]:
    values = tuple(str(row["name"]) for row in ppl["variants"])
    candidate_keys = set(ppl["ppl"]["candidates"])
    if set(values) != candidate_keys:
        raise ValueError("variant metadata and PPL candidate keys differ")
    return values


def _profile_similarity(left: Tensor, right: Tensor, *, top_fraction: float = 0.1) -> dict[str, float]:
    if left.ndim != 1 or right.shape != left.shape:
        raise ValueError("selection profiles must be matched vectors")
    left = left.double()
    right = right.double()
    cosine = float(
        torch.dot(left, right)
        / (left.norm() * right.norm()).clamp_min(1e-30)
    )
    k = min(left.numel(), max(1, int(round(left.numel() * top_fraction))))
    left_top = set(torch.topk(left, k=k, sorted=False).indices.tolist())
    right_top = set(torch.topk(right, k=k, sorted=False).indices.tolist())
    jaccard = len(left_top & right_top) / max(len(left_top | right_top), 1)
    return {
        "cosine": cosine,
        "top10pct_jaccard": jaccard,
        "mean_absolute_probability_shift": float((left - right).abs().mean()),
        "maximum_absolute_probability_shift": float((left - right).abs().max()),
    }


def _frequency_path(payload: Mapping[str, Any]) -> Path:
    result_path = Path(str(payload["_result_path"]))
    recorded = Path(str(payload["selection_frequency"]["path"]))
    if recorded.is_file():
        return recorded
    fallback = result_path.parent / "selection_frequency.safetensors"
    if not fallback.is_file():
        raise FileNotFoundError(fallback)
    return fallback


def _profile_prefix(dataset: str, variant: str) -> str:
    return f"{dataset}__{variant}__"


def _profile_by_suffix(bank: Mapping[str, Tensor], prefix: str) -> dict[str, Tensor]:
    return {key[len(prefix) :]: value for key, value in bank.items() if key.startswith(prefix)}


def _profile_comparisons(
    ppl: Mapping[str, Any],
    mcq: Mapping[str, Mapping[str, Any]],
    variants: Sequence[str],
) -> list[dict[str, Any]]:
    wiki_bank = load_file(_frequency_path(ppl), device="cpu")
    rows: list[dict[str, Any]] = []
    for task_name, payload in sorted(mcq.items()):
        task_bank = load_file(_frequency_path(payload), device="cpu")
        for variant in variants:
            wiki = _profile_by_suffix(
                wiki_bank,
                _profile_prefix("wikitext2", variant),
            )
            task = _profile_by_suffix(
                task_bank,
                _profile_prefix(f"mcq_{task_name}", variant),
            )
            if not wiki or set(wiki) != set(task):
                raise ValueError(
                    f"WikiText/{task_name} profiles differ for variant {variant}"
                )
            layer_rows = [
                {"profile": suffix, **_profile_similarity(wiki[suffix], task[suffix])}
                for suffix in sorted(wiki)
            ]
            rows.append(
                {
                    "task": task_name,
                    "variant": variant,
                    "layers": layer_rows,
                    "layer_count": len(layer_rows),
                    "mean_cosine": statistics.fmean(row["cosine"] for row in layer_rows),
                    "minimum_cosine": min(row["cosine"] for row in layer_rows),
                    "mean_top10pct_jaccard": statistics.fmean(
                        row["top10pct_jaccard"] for row in layer_rows
                    ),
                    "minimum_top10pct_jaccard": min(
                        row["top10pct_jaccard"] for row in layer_rows
                    ),
                    "mean_absolute_probability_shift": statistics.fmean(
                        row["mean_absolute_probability_shift"] for row in layer_rows
                    ),
                }
            )
    return rows


def _mcq_summary(
    mcq: Mapping[str, Mapping[str, Any]],
    variants: Sequence[str],
) -> dict[str, Any]:
    task_names = tuple(sorted(mcq))
    dense_tasks: dict[str, dict[str, Any]] = {}
    for task, payload in mcq.items():
        dense_tasks[task] = payload["mcq"]["dense"]["tasks"][0]
    dense_macro = statistics.fmean(float(dense_tasks[task]["accuracy"]) for task in task_names)
    candidates: dict[str, Any] = {}
    for variant in variants:
        task_rows: dict[str, Any] = {}
        for task in task_names:
            source = mcq[task]["mcq"]["candidates"][variant]["tasks"][0]
            paired = source["paired_vs_dense"]
            task_rows[task] = {
                "accuracy": float(source["accuracy"]),
                "accuracy_delta": float(paired["accuracy_delta"]),
                "prediction_agreement": float(paired["prediction_agreement"]),
                "dense_correct_to_candidate_wrong": int(
                    paired["dense_correct_to_candidate_wrong"]
                ),
                "candidate_rescues": int(paired["candidate_rescues"]),
                "choice_score_delta_rmse": float(paired["choice_score_delta_rmse"]),
            }
        candidates[variant] = {
            "macro_accuracy": statistics.fmean(row["accuracy"] for row in task_rows.values()),
            "macro_accuracy_delta": statistics.fmean(
                row["accuracy_delta"] for row in task_rows.values()
            ),
            "macro_prediction_agreement": statistics.fmean(
                row["prediction_agreement"] for row in task_rows.values()
            ),
            "total_dense_correct_to_candidate_wrong": sum(
                row["dense_correct_to_candidate_wrong"] for row in task_rows.values()
            ),
            "total_candidate_rescues": sum(
                row["candidate_rescues"] for row in task_rows.values()
            ),
            "tasks": task_rows,
        }
    return {
        "task_order": list(task_names),
        "dense": {
            "macro_accuracy": dense_macro,
            "tasks": {
                task: {
                    "accuracy": float(dense_tasks[task]["accuracy"]),
                    "correct": int(dense_tasks[task]["correct"]),
                    "answered": int(dense_tasks[task]["answered"]),
                }
                for task in task_names
            },
        },
        "candidates": candidates,
    }


def _markdown(summary: Mapping[str, Any]) -> str:
    variants = summary["variant_order"]
    ppl = summary["ppl"]
    mcq = summary["mcq"]
    tasks = mcq["task_order"]
    lines = [
        "# Qwen3.5 post-gate Top-K: WikiText2 + MCQ",
        "",
        (
            "All candidates use exact checkpoint gates/states and TP8 source-local "
            "Top-K immediately before the output projection. The implementation is "
            "a dense-zero-fill quality oracle, not a sparse-kernel timing result."
        ),
        "",
        "## WikiText2 PPL",
        "",
        "| Variant | PPL | Delta NLL | Paired SE | Top-1 agreement |",
        "|---|---:|---:|---:|---:|",
        (
            f"| dense | {ppl['dense']['perplexity']:.6f} | 0 | 0 | 1 |"
        ),
    ]
    for variant in variants:
        row = ppl["candidates"][variant]
        lines.append(
            f"| {variant} | {row['perplexity']:.6f} | "
            f"{row['delta_mean_nll']:+.6f} | "
            f"{row['paired_window_standard_error']:.6f} | "
            f"{row['top1_agreement']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## MCQ accuracy",
            "",
            "| Variant | Macro | Delta | Agreement | " + " | ".join(tasks) + " |",
            "|---|---:|---:|---:|" + "---:|" * len(tasks),
        ]
    )
    dense = mcq["dense"]
    lines.append(
        f"| dense | {dense['macro_accuracy']:.4f} | 0 | 1 | "
        + " | ".join(f"{dense['tasks'][task]['accuracy']:.4f}" for task in tasks)
        + " |"
    )
    for variant in variants:
        row = mcq["candidates"][variant]
        lines.append(
            f"| {variant} | {row['macro_accuracy']:.4f} | "
            f"{100 * row['macro_accuracy_delta']:+.2f} pp | "
            f"{row['macro_prediction_agreement']:.4f} | "
            + " | ".join(f"{row['tasks'][task]['accuracy']:.4f}" for task in tasks)
            + " |"
        )
    lines.extend(
        [
            "",
            "## WikiText2-to-MCQ selection-profile shift",
            "",
            "| Variant | Task | Mean cosine | Min cosine | Mean top-10% Jaccard | Min Jaccard |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in summary["profile_comparisons"]:
        lines.append(
            f"| {row['variant']} | {row['task']} | {row['mean_cosine']:.4f} | "
            f"{row['minimum_cosine']:.4f} | {row['mean_top10pct_jaccard']:.4f} | "
            f"{row['minimum_top10pct_jaccard']:.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    root = Path(args.input_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    output_json = (
        root / "combined_summary.json"
        if args.output_json is None
        else Path(args.output_json).expanduser().resolve()
    )
    output_md = (
        root / "combined_summary.md"
        if args.output_md is None
        else Path(args.output_md).expanduser().resolve()
    )
    if output_json.exists() or output_md.exists():
        raise FileExistsError("refusing to overwrite an existing combined summary")
    ppl_payload, mcq_payloads = _load_results(root)
    variants = _variant_order(ppl_payload)
    for task, payload in mcq_payloads.items():
        if set(payload["mcq"]["candidates"]) != set(variants):
            raise ValueError(f"task {task} has a different variant set")
    summary = {
        "format": SUMMARY_FORMAT,
        "schema_version": 1,
        "input_root": str(root),
        "variant_order": list(variants),
        "ppl": ppl_payload["ppl"],
        "mcq": _mcq_summary(mcq_payloads, variants),
        "profile_comparisons": _profile_comparisons(
            ppl_payload,
            mcq_payloads,
            variants,
        ),
        "source_results": [
            ppl_payload["_result_path"],
            *[mcq_payloads[name]["_result_path"] for name in sorted(mcq_payloads)],
        ],
    }
    _atomic_text(output_json, json.dumps(summary, indent=2, sort_keys=True) + "\n")
    _atomic_text(output_md, _markdown(summary))
    print(f"[Saved] {output_json}")
    print(f"[Saved] {output_md}")


if __name__ == "__main__":
    main()
