#!/usr/bin/env python3
"""Merge disjoint window shards from the KQ-routing PPL evaluator."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import statistics
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.eval_c1_block_scheduled_ppl import _distribution  # noqa: E402
from evaluation.eval_qwen3_8b_c1_kq_routing_ppl import (  # noqa: E402
    FORMAT,
    _markdown,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _combine_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    samples = sum(int(row.get("samples", 1)) for row in rows)
    tokens = sum(int(row["evaluated_tokens"]) for row in rows)
    nll_sum = sum(float(row["nll_sum"]) for row in rows)
    mean_nll = nll_sum / tokens
    return {
        "samples": samples,
        "evaluated_tokens": tokens,
        "nll_sum": nll_sum,
        "mean_nll": mean_nll,
        "ppl": math.exp(mean_nll),
    }


def _combine_comparisons(
    rows: list[dict[str, Any]],
    references: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    reference = _combine_results(references)
    candidate = _combine_results(candidates)
    sample_deltas = [float(row["mean_nll_delta"]) for row in rows]
    agreements = sum(
        int(
            row.get(
                "top1_agreements",
                round(
                    float(row["top1_agreement"])
                    * int(result["evaluated_tokens"])
                ),
            )
        )
        for row, result in zip(rows, references, strict=True)
    )
    comparisons = sum(
        int(row.get("top1_comparisons", result["evaluated_tokens"]))
        for row, result in zip(rows, references, strict=True)
    )
    return {
        "mean_nll_delta": candidate["mean_nll"] - reference["mean_nll"],
        "ppl_ratio": math.exp(candidate["mean_nll"] - reference["mean_nll"]),
        "top1_agreement": agreements / comparisons,
        "paired_sample_standard_error": (
            statistics.stdev(sample_deltas) / math.sqrt(len(sample_deltas))
            if len(sample_deltas) > 1
            else None
        ),
        "sample_mean_nll_delta": _distribution(sample_deltas),
    }


def _combine_runtime(rows: list[dict[str, float]]) -> dict[str, float]:
    result = {}
    for name in rows[0]:
        values = [float(row[name]) for row in rows]
        result[name] = (
            max(values) if name == "resident_selector_metadata_bytes" else sum(values)
        )
    result["selected_token_fraction"] = (
        result["selected_tokens"] / result["physical_valid_tokens"]
        if result["physical_valid_tokens"]
        else 0.0
    )
    result["query_selected_token_fraction"] = (
        result["query_selected_tokens"] / result["query_valid_tokens"]
        if result["query_valid_tokens"]
        else 0.0
    )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    return parser


def aggregate(args: argparse.Namespace) -> None:
    input_paths = [path.expanduser().resolve() for path in args.inputs]
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in input_paths]
    if any(
        payload.get("format") != FORMAT or payload.get("status") != "complete"
        for payload in payloads
    ):
        raise ValueError("every shard must be a complete compatible PPL result")
    shard_indices = []
    claimed_indices: set[int] = set()
    for payload in payloads:
        start = int(payload["metadata"]["window_start"])
        samples = int(payload["metadata"]["samples"])
        indices = set(range(start, start + samples))
        overlap = claimed_indices & indices
        if overlap:
            raise ValueError(
                "PPL shards overlap at window indices "
                + ", ".join(str(index) for index in sorted(overlap))
            )
        shard_indices.append(indices)
        claimed_indices.update(indices)
    window_indices = sorted(claimed_indices)
    if window_indices != list(range(window_indices[0], window_indices[-1] + 1)):
        raise ValueError("PPL shards must form one contiguous window range")
    invariant_names = (
        "model_config_sha256",
        "c1_export_sha256",
        "routing_factor_result_sha256",
        "routing_factor_tensor_sha256",
        "windows_sha256",
        "sequence_length",
        "prefill_tokens",
        "block_length",
        "page_size",
        "ranks",
        "nominal_token_budgets",
        "force_last_page",
        "layers",
        "dtype",
    )
    first_metadata = payloads[0]["metadata"]
    for payload in payloads[1:]:
        for name in invariant_names:
            if payload["metadata"][name] != first_metadata[name]:
                raise ValueError(f"PPL shard metadata differs for {name}")

    corpus_coverage = None
    windows_manifest_path = Path(first_metadata["windows"]).parent / "manifest.json"
    if (
        windows_manifest_path.is_file()
        and _sha256(windows_manifest_path)
        == first_metadata["windows_manifest_sha256"]
    ):
        windows_manifest = json.loads(
            windows_manifest_path.read_text(encoding="utf-8")
        )
        tokenized_total = int(windows_manifest["dataset"]["tokenized_total"])
        covered_tokens = len(window_indices) * int(first_metadata["sequence_length"])
        all_complete_blocks = (
            int(windows_manifest["sampling"]["start_token"]) == 0
            and window_indices
            == list(range(int(windows_manifest["sampling"]["samples"])))
            and covered_tokens
            == (tokenized_total // int(first_metadata["sequence_length"]))
            * int(first_metadata["sequence_length"])
        )
        corpus_coverage = {
            "all_complete_blocks": all_complete_blocks,
            "covered_tokens": covered_tokens,
            "tokenized_total": tokenized_total,
            "trailing_tokens": tokenized_total - covered_tokens,
        }

    sample_records = [
        sample for payload in payloads for sample in payload["samples"]
    ]
    sample_records.sort(key=lambda sample: int(sample["sample_index"]))
    if [int(sample["sample_index"]) for sample in sample_records] != window_indices:
        raise ValueError("sample records do not exactly cover the shard window range")
    baseline_configs = [sample["configurations"][0] for sample in sample_records]
    bf16_rows = [row["bf16_dense"] for row in baseline_configs]
    c1_rows = [row["c1_exact_qk"] for row in baseline_configs]
    c1_comparisons = [row["c1_vs_bf16"] for row in baseline_configs]
    baseline = {
        "bf16_dense": _combine_results(bf16_rows),
        "c1_exact_qk": _combine_results(c1_rows),
        "c1_vs_bf16": _combine_comparisons(
            c1_comparisons, bf16_rows, c1_rows
        ),
    }

    configurations = {
        (
            int(row["routing_rank"]),
            int(row["nominal_token_budget"]),
        )
        for row in payloads[0]["aggregate"]
    }
    aggregate_rows = []
    for rank, budget in sorted(configurations):
        shard_rows = []
        for payload in payloads:
            matches = [
                row
                for row in payload["aggregate"]
                if int(row["routing_rank"]) == rank
                and int(row["nominal_token_budget"]) == budget
            ]
            if len(matches) != 1:
                raise ValueError("each shard must contain every routing configuration")
            shard_rows.append(matches[0])
        sample_rows = []
        for sample in sample_records:
            matches = [
                row
                for row in sample["configurations"]
                if int(row["routing_rank"]) == rank
                and int(row["nominal_token_budget"]) == budget
            ]
            if len(matches) != 1:
                raise ValueError(
                    "each sample must contain every routing configuration"
                )
            sample_rows.append(matches[0])
        sparse_rows = [row["sparse"] for row in sample_rows]
        aggregate_rows.append(
            {
                "routing_rank": rank,
                "nominal_token_budget": budget,
                "persistent_gpu_scalar_ratio": shard_rows[0][
                    "persistent_gpu_scalar_ratio"
                ],
                "sparse": _combine_results(sparse_rows),
                "sparse_vs_c1": _combine_comparisons(
                    [row["sparse_vs_c1"] for row in sample_rows],
                    c1_rows,
                    sparse_rows,
                ),
                "sparse_vs_bf16": _combine_comparisons(
                    [row["sparse_vs_bf16"] for row in sample_rows],
                    bf16_rows,
                    sparse_rows,
                ),
                "runtime_logical": _combine_runtime(
                    [row["runtime_logical"] for row in shard_rows]
                ),
            }
        )

    metadata = dict(first_metadata)
    metadata.update(
        {
            "window_start": window_indices[0],
            "window_indices": window_indices,
            "samples": sum(int(payload["metadata"]["samples"]) for payload in payloads),
            "parallel_shards": len(payloads),
            "shards": [
                {"path": str(path), "sha256": _sha256(path)}
                for path in input_paths
            ],
        }
    )
    if corpus_coverage is not None:
        metadata["corpus_coverage"] = corpus_coverage
    limitations = list(payloads[0]["limitations"])
    if corpus_coverage and corpus_coverage["all_complete_blocks"]:
        limitations[0] = (
            "full corpus is evaluated as independent fixed-length blocks; "
            "the incomplete trailing block is dropped"
        )
    result = {
        "format": FORMAT,
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(sys.argv),
        "metadata": metadata,
        "baseline": baseline,
        "aggregate": aggregate_rows,
        "samples": sample_records,
        "elapsed_seconds": max(float(payload["elapsed_seconds"]) for payload in payloads),
        "limitations": limitations,
    }
    output_json = args.output_json.expanduser().resolve()
    output_markdown = args.output_markdown.expanduser().resolve()
    _atomic_text(output_json, json.dumps(result, indent=2, sort_keys=True) + "\n")
    _atomic_text(output_markdown, _markdown(result))
    print(json.dumps({"baseline": baseline, "aggregate": aggregate_rows}, indent=2))


if __name__ == "__main__":
    aggregate(_parser().parse_args())
