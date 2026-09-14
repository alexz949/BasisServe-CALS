#!/usr/bin/env python3
"""Evaluate the C1-V80 pre-K base-rank routing curve on fresh captures."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import sys
import time
from typing import Any

from safetensors.torch import load_file
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basisserve.core.c1_v_conditional_k_router import (  # noqa: E402
    AffineReducedRankMap,
    affine_predictive_spectrum,
    fit_affine_reduced_rank_map,
)
from evaluation.analyze_qwen3_8b_v80_pre_k_spectra import Moments  # noqa: E402
from evaluation.eval_qwen3_8b_v80_base16_page32_sink_diagnostics import (  # noqa: E402
    _fixed_budget_page_mask,
    _group_page_scores,
    _token_mask,
)
from evaluation.eval_qwen3_8b_v80_conditional_residual_router import (  # noqa: E402
    _apply_base,
    _load_direct,
    _parse_ints,
    _post_rope_rows,
    _rotary_embeddings,
    _stack_base_map,
    _value_codes,
)


FORMAT = "basisserve.qwen3_8b.v80_base_rank_routing.v1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--spectra-root", required=True)
    parser.add_argument("--fresh-direct-dir", required=True)
    parser.add_argument("--c1-v80", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--layers", default=",".join(map(str, range(36))))
    parser.add_argument("--ranks", default="4,8,16,24,32,48,64,80")
    parser.add_argument("--fresh-documents", type=int, default=4)
    parser.add_argument("--page-size", type=int, default=32)
    parser.add_argument("--pinned-prefix-pages", type=int, default=1)
    parser.add_argument("--physical-token-budget", type=int, default=2048)
    parser.add_argument("--work-device", default="cuda")
    parser.add_argument("--torch-num-threads", type=int, default=2)
    return parser.parse_args()


def _write_text(path: Path, contents: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(contents, encoding="utf-8")
    os.replace(temporary, path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    _write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _load_pre_rope_moments(root: Path, layer: int) -> Moments:
    path = next(root.glob(f"shard_*/layer_{layer:03d}_fit_moments.safetensors"))
    tensors = load_file(str(path), device="cpu")
    prefix = "fit_pre_rope"
    return Moments(
        row_count=int(tensors[f"{prefix}_row_count"].item()),
        input_sum=tensors[f"{prefix}_input_sum"],
        target_sum=tensors[f"{prefix}_target_sum"],
        input_gram=tensors[f"{prefix}_input_gram"],
        input_target_gram=tensors[f"{prefix}_input_target_gram"],
        target_gram=tensors[f"{prefix}_target_gram"],
    )


def _fit_maps(
    moments: Moments,
    ranks: tuple[int, ...],
) -> dict[int, tuple[AffineReducedRankMap, ...]]:
    groups = int(moments.input_sum.shape[0])
    return {
        rank: tuple(
            fit_affine_reduced_rank_map(
                row_count=moments.row_count,
                input_sum=moments.input_sum[group],
                target_sum=moments.target_sum[group],
                input_gram=moments.input_gram[group],
                input_target_gram=moments.input_target_gram[group],
                rank=rank,
            )
            for group in range(groups)
        )
        for rank in ranks
    }


def _energy_curve(
    moments: Moments,
    ranks: tuple[int, ...],
) -> dict[str, dict[str, float]]:
    group_spectra = [
        affine_predictive_spectrum(
            row_count=moments.row_count,
            input_sum=moments.input_sum[group],
            target_sum=moments.target_sum[group],
            input_gram=moments.input_gram[group],
            input_target_gram=moments.input_target_gram[group],
            target_gram=moments.target_gram[group],
        )
        for group in range(int(moments.input_sum.shape[0]))
    ]
    result = {}
    for rank in ranks:
        rows = []
        for spectrum in group_spectra:
            total = max(
                spectrum.target_centered_energy,
                torch.finfo(torch.float64).tiny,
            )
            predictable = max(
                spectrum.predictable_energy,
                torch.finfo(torch.float64).tiny,
            )
            captured = spectrum.captured_energy(rank)
            rows.append(
                {
                    "captured_total_fraction": captured / total,
                    "captured_predictable_fraction": captured / predictable,
                    "centered_relative_mse": spectrum.residual_energy(rank) / total,
                }
            )
        result[str(rank)] = {
            key: sum(float(row[key]) for row in rows) / len(rows)
            for key in rows[0]
        }
    return result


def _new_routing_accumulator() -> dict[str, Any]:
    return {
        "attention_mass": [],
        "non_sink_attention_mass": [],
        "non_sink_page_recall": [],
        "score_error": 0.0,
        "score_energy": 0.0,
    }


def _finish_routing(accumulator: dict[str, Any]) -> dict[str, float | int]:
    attention = torch.tensor(accumulator["attention_mass"], dtype=torch.float64)
    non_sink = torch.tensor(
        accumulator["non_sink_attention_mass"],
        dtype=torch.float64,
    )
    page_recall = torch.tensor(
        accumulator["non_sink_page_recall"],
        dtype=torch.float64,
    )
    return {
        "query_sample_count": int(attention.numel()),
        "page_sample_count": int(page_recall.numel()),
        "attention_mass_recall_mean": float(attention.mean()),
        "attention_mass_recall_minimum": float(attention.min()),
        "attention_mass_recall_p05": float(torch.quantile(attention, 0.05)),
        "non_sink_attention_mass_recall_mean": float(non_sink.mean()),
        "non_sink_physical_page_recall_mean": float(page_recall.mean()),
        "paired_query_score_nmse": (
            accumulator["score_error"] / accumulator["score_energy"]
        ),
    }


def _evaluate_layer(
    queries_raw: torch.Tensor,
    rows_raw: torch.Tensor,
    *,
    value_encoder: torch.Tensor,
    base_maps: dict[int, tuple[AffineReducedRankMap, ...]],
    cos: torch.Tensor,
    sin: torch.Tensor,
    page_size: int,
    page_budget: int,
    pinned_prefix_pages: int,
    device: torch.device,
) -> tuple[dict[str, dict[str, float | int]], dict[str, float | int]]:
    documents, query_heads, head_dim = map(int, queries_raw.shape)
    _, tokens, groups, _ = map(int, rows_raw.shape)
    heads_per_group = query_heads // groups
    scaling = head_dim**-0.5
    first_non_sink_token = pinned_prefix_pages * page_size
    encoder = value_encoder.to(device=device, dtype=torch.float32)
    factors = {
        rank: _stack_base_map(maps, device=device)
        for rank, maps in base_maps.items()
    }
    accumulators = {str(rank): _new_routing_accumulator() for rank in base_maps}
    oracle = _new_routing_accumulator()

    for document in range(documents):
        print(f"    fresh document={document + 1}/{documents}", flush=True)
        current = rows_raw[document].to(device=device, dtype=torch.float32)
        queries = queries_raw[document].to(device=device, dtype=torch.float32)
        dense_value = current[..., :head_dim]
        exact_key = current[..., head_dim:]
        codes = _value_codes(dense_value, encoder)
        exact_scores = torch.empty(
            query_heads,
            tokens,
            device=device,
            dtype=torch.float32,
        )
        for group in range(groups):
            first = group * heads_per_group
            stop = first + heads_per_group
            exact_scores[first:stop] = scaling * (
                queries[first:stop] @ exact_key[:, group].mT
            )
        probabilities = torch.softmax(exact_scores, dim=-1)
        exact_group_scores = _group_page_scores(
            exact_scores,
            num_kv_heads=groups,
            page_size=page_size,
            excluded_prefix_pages=pinned_prefix_pages,
        )
        oracle_pages = _fixed_budget_page_mask(
            exact_group_scores,
            page_budget=page_budget,
            forced_prefix_pages=pinned_prefix_pages,
        )
        oracle_tokens = _token_mask(
            oracle_pages,
            page_size=page_size,
            tokens=tokens,
        )
        oracle_query_mask = oracle_tokens.repeat_interleave(heads_per_group, dim=0)
        oracle_mass = (probabilities * oracle_query_mask).sum(dim=-1)
        oracle_non_sink_probabilities = probabilities[:, first_non_sink_token:]
        oracle_non_sink_mass = (
            oracle_non_sink_probabilities
            * oracle_query_mask[:, first_non_sink_token:]
        ).sum(dim=-1) / oracle_non_sink_probabilities.sum(dim=-1).clamp_min(
            torch.finfo(torch.float32).tiny
        )
        oracle["attention_mass"].extend(oracle_mass.tolist())
        oracle["non_sink_attention_mass"].extend(oracle_non_sink_mass.tolist())
        oracle["non_sink_page_recall"].extend(
            torch.ones(groups, device=device).tolist()
        )
        oracle["score_energy"] += float(exact_scores.square().sum())

        for rank, base_factor in factors.items():
            predicted_pre = _apply_base(codes, base_factor)
            predicted_key = _post_rope_rows(predicted_pre, cos, sin)
            proxy_scores = torch.empty_like(exact_scores)
            for group in range(groups):
                first = group * heads_per_group
                stop = first + heads_per_group
                proxy_scores[first:stop] = scaling * (
                    queries[first:stop] @ predicted_key[:, group].mT
                )
            proxy_group_scores = _group_page_scores(
                proxy_scores,
                num_kv_heads=groups,
                page_size=page_size,
                excluded_prefix_pages=pinned_prefix_pages,
            )
            selected_pages = _fixed_budget_page_mask(
                proxy_group_scores,
                page_budget=page_budget,
                forced_prefix_pages=pinned_prefix_pages,
            )
            selected_tokens = _token_mask(
                selected_pages,
                page_size=page_size,
                tokens=tokens,
            )
            query_mask = selected_tokens.repeat_interleave(heads_per_group, dim=0)
            selected_mass = (probabilities * query_mask).sum(dim=-1)
            non_sink_probabilities = probabilities[:, first_non_sink_token:]
            non_sink_mass = (
                non_sink_probabilities * query_mask[:, first_non_sink_token:]
            ).sum(dim=-1) / non_sink_probabilities.sum(dim=-1).clamp_min(
                torch.finfo(torch.float32).tiny
            )
            routed_selected = selected_pages[:, pinned_prefix_pages:]
            routed_oracle = oracle_pages[:, pinned_prefix_pages:]
            page_recall = (routed_selected & routed_oracle).sum(dim=-1) / (
                routed_oracle.sum(dim=-1).clamp_min(1)
            )
            accumulator = accumulators[str(rank)]
            accumulator["attention_mass"].extend(selected_mass.tolist())
            accumulator["non_sink_attention_mass"].extend(non_sink_mass.tolist())
            accumulator["non_sink_page_recall"].extend(page_recall.tolist())
            accumulator["score_error"] += float(
                (proxy_scores - exact_scores).square().sum()
            )
            accumulator["score_energy"] += float(exact_scores.square().sum())
            del predicted_pre, predicted_key, proxy_scores

        del current, queries, dense_value, exact_key, codes, exact_scores, probabilities

    return (
        {rank: _finish_routing(accumulator) for rank, accumulator in accumulators.items()},
        _finish_routing(oracle),
    )


def _weighted_mean(rows: list[dict[str, Any]], key: str, count_key: str) -> float:
    total = sum(int(row[count_key]) for row in rows)
    return sum(float(row[key]) * int(row[count_key]) for row in rows) / total


def aggregate_layers(
    layer_records: list[dict[str, Any]],
    ranks: tuple[int, ...],
) -> dict[str, Any]:
    result = {}
    for rank in ranks:
        key = str(rank)
        energies = [record["energy"][key] for record in layer_records]
        routing = [record["routing"][key] for record in layer_records]
        result[key] = {
            "rank": rank,
            "captured_total_fraction": sum(
                float(row["captured_total_fraction"]) for row in energies
            )
            / len(energies),
            "captured_predictable_fraction": sum(
                float(row["captured_predictable_fraction"]) for row in energies
            )
            / len(energies),
            "centered_relative_mse": sum(
                float(row["centered_relative_mse"]) for row in energies
            )
            / len(energies),
            "attention_mass_recall_mean": _weighted_mean(
                routing,
                "attention_mass_recall_mean",
                "query_sample_count",
            ),
            "non_sink_attention_mass_recall_mean": _weighted_mean(
                routing,
                "non_sink_attention_mass_recall_mean",
                "query_sample_count",
            ),
            "non_sink_physical_page_recall_mean": _weighted_mean(
                routing,
                "non_sink_physical_page_recall_mean",
                "page_sample_count",
            ),
            "paired_query_score_nmse": sum(
                float(row["paired_query_score_nmse"]) for row in routing
            )
            / len(routing),
        }
    oracle_rows = [record["exact_oracle"] for record in layer_records]
    return {
        "ranks": result,
        "exact_oracle": {
            "attention_mass_recall_mean": _weighted_mean(
                oracle_rows,
                "attention_mass_recall_mean",
                "query_sample_count",
            ),
            "non_sink_attention_mass_recall_mean": _weighted_mean(
                oracle_rows,
                "non_sink_attention_mass_recall_mean",
                "query_sample_count",
            ),
        },
    }


def markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Qwen3-8B C1-V80 Base-Rank K-Routing Sweep",
        "",
        "| Base rank | Captured K energy | Captured predictable K | Attention mass recall | Non-sink mass recall | Non-sink page recall | Score NMSE |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["aggregate"]["ranks"].values():
        lines.append(
            f"| {row['rank']} | {row['captured_total_fraction']:.6f} | "
            f"{row['captured_predictable_fraction']:.6f} | "
            f"{row['attention_mass_recall_mean']:.6f} | "
            f"{row['non_sink_attention_mass_recall_mean']:.6f} | "
            f"{row['non_sink_physical_page_recall_mean']:.6f} | "
            f"{row['paired_query_score_nmse']:.6f} |"
        )
    oracle = payload["aggregate"]["exact_oracle"]
    lines.extend(
        [
            "",
            f"Exact Page32/B2048 oracle attention mass: {oracle['attention_mass_recall_mean']:.6f}.",
            f"Exact non-sink attention mass: {oracle['non_sink_attention_mass_recall_mean']:.6f}.",
            "",
        ]
    )
    return "\n".join(lines)


@torch.inference_mode()
def main() -> None:
    args = _parse_args()
    torch.set_num_threads(args.torch_num_threads)
    torch.backends.cuda.matmul.allow_tf32 = True
    started = time.monotonic()
    model_root = Path(args.model).expanduser().resolve()
    spectra_root = Path(args.spectra_root).expanduser().resolve()
    fresh_root = Path(args.fresh_direct_dir).expanduser().resolve()
    c1_root = Path(args.c1_v80).expanduser().resolve()
    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    layers = _parse_ints(args.layers)
    ranks = _parse_ints(args.ranks)
    device = torch.device(args.work_device)
    sequence = 32768
    page_budget = args.physical_token_budget // args.page_size
    cos, sin = _rotary_embeddings(model_root, sequence=sequence, device=device)
    fresh_manifest = json.loads(
        (fresh_root / "manifest.json").read_text(encoding="utf-8")
    )
    c1_manifest = json.loads((c1_root / "results.json").read_text(encoding="utf-8"))
    layer_records = []
    result_path = output_root / "result.json"
    summary_path = output_root / "summary.md"

    for ordinal, layer in enumerate(layers, start=1):
        layer_started = time.monotonic()
        print(f"[base-rank routing] layer={layer} ({ordinal}/{len(layers)})", flush=True)
        moments = _load_pre_rope_moments(spectra_root, layer)
        maps = _fit_maps(moments, ranks)
        fresh_queries, fresh_rows = _load_direct(fresh_root, fresh_manifest, layer)
        fresh_queries = fresh_queries[: args.fresh_documents]
        fresh_rows = fresh_rows[: args.fresh_documents]
        artifact = c1_manifest["artifacts"][str(layer)]["file"]
        c1_tensors = load_file(str(c1_root / artifact), device="cpu")
        routing, oracle = _evaluate_layer(
            fresh_queries,
            fresh_rows,
            value_encoder=c1_tensors["value_coordinate_encoders"],
            base_maps=maps,
            cos=cos,
            sin=sin,
            page_size=args.page_size,
            page_budget=page_budget,
            pinned_prefix_pages=args.pinned_prefix_pages,
            device=device,
        )
        layer_records.append(
            {
                "layer": layer,
                "energy": _energy_curve(moments, ranks),
                "routing": routing,
                "exact_oracle": oracle,
                "seconds": time.monotonic() - layer_started,
            }
        )
        partial = {
            "format": FORMAT,
            "status": "running",
            "command": shlex.join(sys.argv),
            "layers": layer_records,
            "aggregate": aggregate_layers(layer_records, ranks),
            "elapsed_seconds": time.monotonic() - started,
        }
        _write_json(result_path, partial)
        _write_text(summary_path, markdown(partial))
        if device.type == "cuda":
            torch.cuda.empty_cache()

    result = {
        "format": FORMAT,
        "status": "complete",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(sys.argv),
        "protocol": {
            "model": str(model_root),
            "spectra_root": str(spectra_root),
            "fresh_capture": str(fresh_root),
            "fixed_payload": str(c1_root),
            "fit_documents": 64,
            "fresh_documents": args.fresh_documents,
            "sequence_length": sequence,
            "target": "pre-RoPE K followed by exact RoPE",
            "ranks": list(ranks),
            "page_size": args.page_size,
            "pinned_prefix_pages": args.pinned_prefix_pages,
            "physical_token_budget_per_kv_group": args.physical_token_budget,
            "routed_pages_per_kv_group": page_budget - args.pinned_prefix_pages,
            "selection": (
                "pin Page0, rank non-sink pages by per-head normalized page mass, "
                "take the GQA-group maximum, and select one fixed physical page set"
            ),
        },
        "layers": layer_records,
        "aggregate": aggregate_layers(layer_records, ranks),
        "elapsed_seconds": time.monotonic() - started,
        "environment": {
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "python_executable": sys.executable,
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_devices": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ],
        },
    }
    _write_json(result_path, result)
    _write_text(summary_path, markdown(result))
    print(f"[base-rank routing] wrote {result_path} and {summary_path}", flush=True)


if __name__ == "__main__":
    main()
