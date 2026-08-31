#!/usr/bin/env python3
"""Replay captured Qwen3 C1 tensors through the Reverse ShadowKV oracle."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any

from safetensors.torch import load_file
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basisserve.core.c1_k_reverse_shadow import (  # noqa: E402
    ReverseShadowConfig,
    build_c1_decoder_gram,
    build_c1_page_metadata,
    build_post_rope_k_landmarks,
    c1_k_reverse_shadow_attention,
    reverse_shadow_quality_statistics,
)
from evaluation.eval_qwen3_c1_k_refine_oracle import (  # noqa: E402
    _load_c1_decoder,
    _load_capture_layer,
    _sha256,
)


FORMAT = "basisserve.c1_k_reverse_shadow_oracle.v5"

_HIGHER_IS_BETTER = {
    "exact_attention_mass_selected",
    "exact_top_page_recall",
    "exact_top_mass_page_recall",
    "exact_top_token_recall",
    "attention_top_k_overlap",
    "c1_decoded_output_cosine_similarity",
}
_QUALITY_METRICS = (
    "exact_attention_mass_selected",
    "exact_top_mass_page_recall",
    "attention_kl_sparse_to_exact",
    "attention_probability_l1",
    "attention_max_probability_error",
    "attention_top_k_overlap",
    "c1_latent_relative_l2",
    "c1_decoded_output_relative_l2",
    "c1_decoded_output_error_l2",
    "c1_decoded_output_reference_l2",
    "c1_decoded_output_max_absolute_error",
    "c1_decoded_output_cosine_similarity",
)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _parse_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def _parse_strings(value: str) -> list[str]:
    return [item for item in value.split(",") if item]


def _capture_layers(tensors: dict[str, torch.Tensor]) -> list[int]:
    result = []
    for name in tensors:
        pieces = name.split(".")
        if len(pieces) == 3 and pieces[0] == "layers" and pieces[2] == "query":
            result.append(int(pieces[1]))
    return sorted(set(result))


def _timed(callable_, *, device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    value = callable_()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return value, time.perf_counter() - started


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(max(math.ceil(fraction * len(ordered)) - 1, 0), len(ordered) - 1)
    return ordered[index]


def _metric_summary(
    records: list[dict[str, Any]], metric: str
) -> dict[str, Any] | None:
    observations = [
        (float(row[metric]), int(row["layer"]), int(row["example"]))
        for row in records
        if metric in row and math.isfinite(float(row[metric]))
    ]
    if not observations:
        return None
    values = [value for value, _, _ in observations]
    result: dict[str, Any] = {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
    }
    if metric in _HIGHER_IS_BETTER:
        tail_value, tail_layer, tail_example = min(observations)
        result.update(
            {
                "p10": _percentile(values, 0.10),
                "p05": _percentile(values, 0.05),
                "minimum": tail_value,
                "minimum_layer": tail_layer,
                "minimum_example": tail_example,
            }
        )
    else:
        tail_value, tail_layer, tail_example = max(observations)
        result.update(
            {
                "p90": _percentile(values, 0.90),
                "p95": _percentile(values, 0.95),
                "maximum": tail_value,
                "maximum_layer": tail_layer,
                "maximum_example": tail_example,
            }
        )
    return result


def _window_metric_summary(
    records: list[dict[str, Any]], metric: str
) -> dict[str, Any] | None:
    by_example: dict[int, list[float]] = {}
    for row in records:
        if metric not in row or not math.isfinite(float(row[metric])):
            continue
        by_example.setdefault(int(row["example"]), []).append(float(row[metric]))
    if not by_example:
        return None
    averaged = [
        {
            "example": example,
            "layer_mean": statistics.fmean(values),
        }
        for example, values in sorted(by_example.items())
    ]
    values = [row["layer_mean"] for row in averaged]
    result: dict[str, Any] = {
        "values": averaged,
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
    }
    if metric in _HIGHER_IS_BETTER:
        worst = min(averaged, key=lambda row: row["layer_mean"])
        result.update(
            {
                "p10": _percentile(values, 0.10),
                "p05": _percentile(values, 0.05),
                "minimum": worst["layer_mean"],
                "minimum_example": worst["example"],
            }
        )
    else:
        worst = max(averaged, key=lambda row: row["layer_mean"])
        result.update(
            {
                "p90": _percentile(values, 0.90),
                "p95": _percentile(values, 0.95),
                "maximum": worst["layer_mean"],
                "maximum_example": worst["example"],
            }
        )
    return result


def _schedule_aggregate(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in records:
        key = (
            row["policy"],
            row.get("query_head_aggregation", "not_applicable"),
            row["page_size"],
            row["landmarks_per_page"],
            row["exact_token_budget"],
            row["recent_exact_window"],
        )
        grouped.setdefault(key, []).append(row)
    result = []
    for key, group in sorted(grouped.items()):
        policy, aggregation, page_size, landmarks, budget, recent = key
        metrics = {
            metric: summary
            for metric in _QUALITY_METRICS
            if (summary := _metric_summary(group, metric)) is not None
        }
        window_metrics = {
            metric: summary
            for metric in _QUALITY_METRICS
            if (summary := _window_metric_summary(group, metric)) is not None
        }
        result.append(
            {
                "policy": policy,
                "query_head_aggregation": aggregation,
                "page_size": page_size,
                "landmarks_per_page": landmarks,
                "exact_token_budget": budget,
                "recent_exact_window": recent,
                "observations": len(group),
                "layers": sorted({int(row["layer"]) for row in group}),
                "examples": sorted({int(row["example"]) for row in group}),
                "metrics": metrics,
                "window_layer_mean_metrics": window_metrics,
                "decoded_output_energy_relative_l2": (
                    math.sqrt(
                        sum(
                            float(row["c1_decoded_output_error_l2"]) ** 2
                            for row in group
                            if "c1_decoded_output_error_l2" in row
                        )
                        / max(
                            sum(
                                float(row["c1_decoded_output_reference_l2"]) ** 2
                                for row in group
                                if "c1_decoded_output_reference_l2" in row
                            ),
                            1.0e-300,
                        )
                    )
                    if any("c1_decoded_output_error_l2" in row for row in group)
                    else None
                ),
            }
        )
    return result


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# C1 K-only Reverse ShadowKV oracle",
        "",
        "All `teacher_*` selectors consult full exact K and are not serving "
        "policies. ",
        "Logical byte/FLOP counts are not measured CPU-offload speedups.",
        "",
        "Each observation is one held-out window at one layer. The tail "
        "location therefore identifies both the layer and the window.",
        "",
        "| policy | head aggregation | landmarks/page | page | sparse-layer budget | recent | obs. | "
        "mass mean | mass min | decoded rel-L2 mean | decoded rel-L2 p95 | "
        "decoded rel-L2 max (layer/example) | energy rel-L2 |",
        "|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|---:|",
    ]
    for row in payload["schedule_aggregate"]:
        mass = row["metrics"].get("exact_attention_mass_selected", {})
        decoded = row["metrics"].get("c1_decoded_output_relative_l2", {})
        location = (
            "-"
            if not decoded
            else f"{decoded['maximum_layer']}/{decoded['maximum_example']}"
        )
        energy_relative_l2 = row.get("decoded_output_energy_relative_l2")
        energy_text = (
            f"{energy_relative_l2:.6e}"
            if energy_relative_l2 is not None
            else "nan"
        )
        lines.append(
            f"| {row['policy']} | "
            f"{row['query_head_aggregation']} | "
            f"{row['landmarks_per_page']} | {row['page_size']} | "
            f"{row['exact_token_budget']} | {row['recent_exact_window']} | "
            f"{row['observations']} | {mass.get('mean', math.nan):.6f} | "
            f"{mass.get('minimum', math.nan):.6f} | "
            f"{decoded.get('mean', math.nan):.6e} | "
            f"{decoded.get('p95', math.nan):.6e} | "
            f"{decoded.get('maximum', math.nan):.6e} ({location}) | "
            f"{energy_text} |"
        )
    return "\n".join(lines) + "\n"


def _c1_export(
    path: Path | None,
) -> tuple[Path | None, dict[str, Any] | None, Path | None]:
    if path is None:
        return None, None, None
    resolved = path.expanduser().resolve()
    if resolved.is_dir():
        candidates = (
            resolved / "manifest.json",
            resolved / "results.json",
            resolved / "result.json",
        )
        manifest_path = next((item for item in candidates if item.exists()), None)
        if manifest_path is None:
            raise FileNotFoundError(f"no C1 manifest/results JSON found in {resolved}")
        directory = resolved
    else:
        manifest_path = resolved
        directory = resolved.parent
    return (
        directory,
        json.loads(manifest_path.read_text(encoding="utf-8")),
        manifest_path,
    )


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> None:
    capture_path = args.capture.expanduser().resolve()
    device = torch.device(args.device)
    c1_dir, c1_result, c1_manifest_path = _c1_export(args.c1_export)
    if capture_path.is_dir():
        capture_manifest_path = capture_path / "manifest.json"
        capture_manifest = json.loads(
            capture_manifest_path.read_text(encoding="utf-8")
        )
        capture = None
        available_layers = sorted(int(layer) for layer in capture_manifest["artifacts"])
        capture_digest_path = capture_manifest_path
        captured_c1_hash = capture_manifest.get("c1_export", {}).get(
            "results_sha256"
        )
        if (
            captured_c1_hash is not None
            and c1_manifest_path is not None
            and captured_c1_hash != _sha256(c1_manifest_path)
        ):
            raise ValueError("capture and replay use different C1 exports")
    else:
        capture_manifest = None
        capture = load_file(str(capture_path), device=str(device))
        available_layers = _capture_layers(capture)
        capture_digest_path = capture_path

    layers = available_layers if args.layers == "all" else _parse_ints(args.layers)
    if not layers or any(layer not in available_layers for layer in layers):
        raise ValueError(
            f"requested layers are not captured; available={available_layers}"
        )
    page_sizes = _parse_ints(args.page_sizes)
    budgets = _parse_ints(args.exact_token_budgets)
    recent_windows = _parse_ints(args.recent_exact_windows)
    selectors = _parse_strings(args.selectors)
    query_head_aggregations = _parse_strings(
        getattr(args, "query_head_aggregations", "logsumexp_head")
    )
    if not query_head_aggregations or any(
        aggregation not in {"max_head", "logsumexp_head"}
        for aggregation in query_head_aggregations
    ):
        raise ValueError(
            "query-head aggregations must be max_head and/or logsumexp_head"
        )
    landmarks_per_page = _parse_ints(args.landmarks_per_page)
    supported_selectors = {
        "teacher_exact",
        "teacher_mass",
        "teacher_output",
        "teacher_influence",
        "mean_landmark",
        "centroid_radius",
        "quest_minmax",
        "quest_k",
        "quest_c1_latent",
        "quest_c1_output",
    }
    if any(selector not in supported_selectors for selector in selectors):
        raise ValueError(f"selectors must be drawn from {sorted(supported_selectors)}")
    full_exact_layers = set(_parse_ints(args.full_exact_layers))
    if any(layer not in layers for layer in full_exact_layers):
        raise ValueError("full-exact layers must be a subset of requested layers")
    policy_configs = []
    for selector in selectors:
        counts = (
            landmarks_per_page[:1]
            if selector
            in (
                "teacher_exact",
                "teacher_mass",
                "teacher_output",
                "teacher_influence",
                "quest_minmax",
                "quest_k",
                "quest_c1_latent",
                "quest_c1_output",
            )
            else landmarks_per_page
        )
        aggregations = (
            query_head_aggregations
            if selector in ("quest_k", "quest_c1_latent", "quest_c1_output")
            else ["max_head" if selector == "quest_minmax" else "not_applicable"]
        )
        for aggregation in aggregations:
            for count in counts:
                for budget in budgets:
                    for recent in recent_windows:
                        if budget == 0 and recent == 0:
                            continue
                        policy_configs.append(
                            (selector, aggregation, count, budget, recent)
                        )
    records: list[dict[str, Any]] = []

    for layer in layers:
        query, exact_key, c1_value, mask, decoder = _load_capture_layer(
            capture_path,
            capture,
            capture_manifest,
            layer=layer,
            device=device,
        )
        example_start = int(args.example_start)
        example_stop = (
            len(query) if args.examples == 0 else example_start + int(args.examples)
        )
        if example_start < 0 or example_stop > len(query) or example_start >= example_stop:
            raise ValueError("requested replay examples are outside the capture")
        query = query[example_start:example_stop]
        exact_key = exact_key[example_start:example_stop]
        c1_value = c1_value[example_start:example_stop]
        if mask is not None:
            mask = mask[example_start:example_stop]
        if decoder is None:
            decoder = _load_c1_decoder(
                c1_dir, c1_result, layer=layer, device=device
            )
        needs_c1_metadata = any(
            selector in ("quest_c1_latent", "quest_c1_output")
            for selector in selectors
        )
        decoder_gram = (
            build_c1_decoder_gram(decoder) if needs_c1_metadata else None
        )
        sequence = int(exact_key.shape[2])

        for local_example in range(len(query)):
            example = example_start + local_example
            example_query = query[local_example : local_example + 1]
            example_key = exact_key[local_example : local_example + 1]
            example_value = c1_value[local_example : local_example + 1]
            example_mask = (
                None if mask is None else mask[local_example : local_example + 1]
            )
            for page_size in page_sizes:
                landmark_bank = {
                    count: build_post_rope_k_landmarks(
                        example_key,
                        page_size=page_size,
                        landmarks_per_page=count,
                        attention_mask=example_mask,
                        landmark_dtype=args.landmark_dtype,
                    )
                    for count in landmarks_per_page
                }
                baseline_landmarks = landmark_bank[landmarks_per_page[0]]
                c1_page_metadata = (
                    build_c1_page_metadata(
                        example_value,
                        decoder,
                        page_size=page_size,
                        attention_mask=example_mask,
                        decoder_gram=decoder_gram,
                    )
                    if needs_c1_metadata
                    else None
                )
                baseline_config = ReverseShadowConfig(
                    page_size=page_size,
                    exact_token_budget=sequence,
                    landmarks_per_page=landmarks_per_page[0],
                    selector="teacher_exact",
                    landmark_dtype=args.landmark_dtype,
                )
                full_exact = c1_k_reverse_shadow_attention(
                    example_query,
                    baseline_landmarks,
                    example_value,
                    baseline_config,
                    example_key,
                    example_mask,
                    layer_idx=layer,
                )
                if layer in full_exact_layers:
                    quality = reverse_shadow_quality_statistics(
                        full_exact,
                        full_exact,
                        query=example_query,
                        exact_key=example_key,
                        config=baseline_config,
                        attention_mask=example_mask,
                        decoder=decoder,
                        attention_top_k=args.attention_top_k,
                    )
                    for policy, aggregation, count, budget, recent in policy_configs:
                        record = {
                            "layer": layer,
                            "example": example,
                            "source": "all_physical_kv_heads",
                            "policy": policy,
                            "query_head_aggregation": aggregation,
                            "selector": "full_exact_resident",
                            "key_placement": "gpu_resident_full",
                            "landmarks_per_page": count,
                            "page_size": page_size,
                            "exact_token_budget": budget,
                            "layer_exact_token_budget": sequence,
                            "recent_exact_window": recent,
                            **full_exact.statistics,
                            **quality,
                            "cpu_exact_key_bytes_fetched": 0.0,
                            "oracle_seconds": 0.0,
                        }
                        records.append(record)
                    print(
                        f"[C1 Reverse ShadowKV] layer={layer} example={example} "
                        "selector=full_exact_resident",
                        flush=True,
                    )
                    continue

                for selector, aggregation, count, budget, recent in policy_configs:
                    landmarks = landmark_bank[count]
                    config = ReverseShadowConfig(
                        page_size=page_size,
                        exact_token_budget=budget,
                        recent_exact_window=recent,
                        landmarks_per_page=count,
                        selector=selector,
                        landmark_dtype=args.landmark_dtype,
                        query_head_aggregation=(
                            aggregation
                            if aggregation != "not_applicable"
                            else "max_head"
                        ),
                    )
                    candidate, elapsed = _timed(
                        lambda: c1_k_reverse_shadow_attention(
                            example_query,
                            landmarks,
                            example_value,
                            config,
                            example_key,
                            example_mask,
                            c1_page_metadata=c1_page_metadata,
                            decoder=decoder,
                            layer_idx=layer,
                        ),
                        device=device,
                    )
                    quality = reverse_shadow_quality_statistics(
                        full_exact,
                        candidate,
                        query=example_query,
                        exact_key=example_key,
                        config=config,
                        attention_mask=example_mask,
                        decoder=decoder,
                        attention_top_k=args.attention_top_k,
                    )
                    record = {
                        "layer": layer,
                        "example": example,
                        "source": "all_physical_kv_heads",
                        "policy": selector,
                        "selector": selector,
                        "query_head_aggregation": aggregation,
                        "key_placement": "cpu_exact_key",
                        "landmarks_per_page": count,
                        "page_size": page_size,
                        "exact_token_budget": budget,
                        "layer_exact_token_budget": budget,
                        "recent_exact_window": recent,
                        **candidate.statistics,
                        **quality,
                        "cpu_exact_key_bytes_fetched": candidate.statistics[
                            "oracle_page_store_key_bytes_read"
                        ],
                        "oracle_seconds": elapsed,
                    }
                    records.append(record)
                    print(
                        f"[C1 Reverse ShadowKV] layer={layer} example={example} "
                        f"selector={selector} landmarks={count} "
                        f"aggregation={aggregation} "
                        f"page={page_size} budget={budget} recent={recent} "
                        f"mass={quality['exact_attention_mass_selected']:.4f} "
                        f"latent={quality['c1_latent_relative_l2']:.3e}",
                        flush=True,
                    )

    configuration = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    payload = {
        "format": FORMAT,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "notes": {
            "teacher_exact": "exact maximum-QK page oracle; not deployable",
            "teacher_mass": "exact aggregate-attention-mass page oracle; not deployable",
            "teacher_output": (
                "exact decoded page-contribution norm oracle; not deployable"
            ),
            "teacher_influence": (
                "exact decoded leave-one-page-out influence including softmax "
                "renormalization; not deployable"
            ),
            "centroid_radius": "Cauchy upper bound around the stored centroid",
            "quest_minmax": "QUEST coordinate-wise Min/Max page upper bound",
            "quest_k": "QUEST Key bound with explicit GQA head aggregation",
            "quest_c1_latent": "QUEST Key bound weighted by page-max C1 latent norm",
            "quest_c1_output": (
                "QUEST Key bound weighted by page-max decoded C1 output norm"
            ),
            "full_exact_resident": (
                "frozen schedule layer with full exact K resident on GPU; the "
                "sparse-layer budget does not apply"
            ),
            "per_example": (
                "quality metrics are emitted separately for every held-out window"
            ),
            "timing": "Python all-GPU correctness reference; not CPU-offload latency",
        },
        "capture": {
            "file": str(capture_path),
            "manifest_or_file_sha256": _sha256(capture_digest_path),
        },
        "c1_export": (
            None
            if c1_manifest_path is None
            else {
                "manifest": str(c1_manifest_path),
                "sha256": _sha256(c1_manifest_path),
            }
        ),
        "configuration": configuration,
        "records": records,
        "schedule_aggregate": _schedule_aggregate(records),
    }
    _atomic_text(
        args.output_json.expanduser().resolve(),
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )
    _atomic_text(args.output_markdown.expanduser().resolve(), _markdown(payload))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--c1-export", type=Path)
    parser.add_argument("--layers", default="all")
    parser.add_argument(
        "--full-exact-layers",
        default="",
        help="requested layers whose full exact K remains GPU resident",
    )
    parser.add_argument("--example-start", type=int, default=0)
    parser.add_argument("--examples", type=int, default=0)
    parser.add_argument(
        "--selectors",
        default=(
            "teacher_exact,teacher_mass,mean_landmark,centroid_radius,quest_minmax"
        ),
    )
    parser.add_argument("--landmarks-per-page", default="1,4")
    parser.add_argument(
        "--query-head-aggregations",
        default="logsumexp_head",
        help="comma-separated max_head/logsumexp_head modes for quest_* selectors",
    )
    parser.add_argument("--page-sizes", default="16,32,64")
    parser.add_argument("--exact-token-budgets", default="64,128,256,512,1024")
    parser.add_argument("--recent-exact-windows", default="0,256")
    parser.add_argument(
        "--landmark-dtype",
        choices=("float32", "float16", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument("--attention-top-k", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    return parser


if __name__ == "__main__":
    evaluate(_parser().parse_args())
