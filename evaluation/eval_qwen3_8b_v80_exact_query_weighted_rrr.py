#!/usr/bin/env python3
"""Fit pre-RoPE Base16 with exact per-query/per-position QK regression."""

from __future__ import annotations

from datetime import datetime, timezone
import argparse
import json
import os
from pathlib import Path
import shlex
import sys
import time
from typing import Any

from safetensors.torch import load_file, save_file
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.eval_qwen3_8b_v80_conditional_residual_router import (  # noqa: E402
    _discover_capture,
    _fit_base_maps,
    _load_direct,
    _parse_ints,
    _post_rope_rows,
    _rotary_embeddings,
    _value_codes,
)
from evaluation.eval_qwen3_8b_v80_fisher_base import (  # noqa: E402
    _check_query_alignment,
    _discover_query_statistics,
    _load_query_observations,
)
from evaluation.eval_qwen3_8b_v80_pre_rope_fisher_base import (  # noqa: E402
    _evaluate_fresh,
    _gqa_scores,
    _initial_maps,
    _train_base,
    _write_json,
    _write_text,
)


FORMAT = "basisserve.qwen3_8b.v80_exact_query_weighted_rrr.v2"
METRIC_NAME = "exact_query_weighted_qk"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--calibration-root", required=True)
    parser.add_argument("--query-statistics-root", required=True)
    parser.add_argument("--fresh-direct-dir", required=True)
    parser.add_argument("--c1-v80", required=True)
    parser.add_argument("--mse-initialization-root", required=True)
    parser.add_argument("--reference-root", required=True)
    parser.add_argument("--fisher-reference-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--layers", default="0,13,33")
    parser.add_argument("--base-rank", type=int, default=16)
    parser.add_argument("--page-size", type=int, default=32)
    parser.add_argument("--pinned-prefix-pages", type=int, default=1)
    parser.add_argument("--physical-token-budget", type=int, default=2048)
    parser.add_argument("--fit-documents", type=int, default=64)
    parser.add_argument("--validation-documents", type=int, default=16)
    parser.add_argument("--fresh-documents", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--documents-per-step", type=int, default=4)
    parser.add_argument("--factor-learning-rate", type=float, default=2e-3)
    parser.add_argument("--bias-learning-rate", type=float, default=5e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--early-stopping-patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--work-device", default="cuda")
    parser.add_argument("--torch-num-threads", type=int, default=2)
    return parser.parse_args()


def _query_weighted_document_loss(
    queries: torch.Tensor,
    rows: torch.Tensor,
    *,
    query_positions: torch.Tensor,
    value_encoder: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    bias: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    page_size: int,
    pinned_prefix_pages: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    head_dim = int(rows.shape[-1]) // 2
    first_token = int(page_size) * int(pinned_prefix_pages)
    codes = _value_codes(rows[..., :head_dim], value_encoder)
    base_codes = torch.einsum("tgv,gvr->tgr", codes, left)
    predicted_pre_key = torch.einsum("tgr,grd->tgd", base_codes, right)
    predicted_pre_key = predicted_pre_key + bias.unsqueeze(0)
    predicted_key = _post_rope_rows(predicted_pre_key, cos, sin)
    exact_key = rows[..., head_dim:]
    total_loss = predicted_key.new_zeros(())
    total_energy = predicted_key.new_zeros(())

    for sample, query_position in enumerate(query_positions.tolist()):
        causal_stop = int(query_position) + 1
        current_queries = queries[sample]
        predicted_scores = _gqa_scores(
            current_queries,
            predicted_key[first_token:causal_stop],
        )
        with torch.no_grad():
            exact_scores = _gqa_scores(
                current_queries,
                exact_key[first_token:causal_stop],
            )
        total_loss = total_loss + 0.5 * (
            predicted_scores - exact_scores
        ).square().sum()
        total_energy = total_energy + 0.5 * exact_scores.square().sum()
    return total_loss, total_energy


def _markdown(result: dict[str, Any]) -> str:
    layer = result["layers"][0]
    base_rank = result["protocol"]["base_rank"]
    validation_key = f"validation_{METRIC_NAME}_nmse"
    fit_key = f"fit_{METRIC_NAME}_nmse"
    lines = [
        f"# Qwen3-8B C1-V80 Exact Query-Weighted RRR Rank {base_rank}",
        "",
        f"Layer: {layer['layer']}",
        "",
        f"Best epoch: {layer['base_fit']['best_epoch']}",
        "",
        "| Epoch | Fit exact-QK NMSE | Validation exact-QK NMSE |",
        "|---:|---:|---:|",
    ]
    for row in layer["base_fit"]["history"]:
        fit = row.get(fit_key)
        fit_cell = "—" if fit is None else f"{fit:.6f}"
        lines.append(
            f"| {row['epoch']} | {fit_cell} | {row[validation_key]:.6f} |"
        )
    lines.extend(
        (
            "",
            "| Base | Selected mass | P01 mass | Page recall | Output rel-MSE |",
            "|:---|---:|---:|---:|---:|",
        )
    )
    for name, metric in layer["references"].items():
        lines.append(
            f"| {name} | {metric['attention_mass_recall_mean']:.6f} | "
            f"{metric['attention_mass_recall_p01']:.6f} | "
            f"{metric['non_sink_physical_page_recall_mean']:.6f} | "
            f"{metric['exact_refined_output_relative_mse']:.6f} |"
        )
    metric = layer["fresh"]["r0"]
    lines.append(
        f"| Exact query-weighted pre-RoPE rank{base_rank} | "
        f"{metric['attention_mass_recall_mean']:.6f} | "
        f"{metric['attention_mass_recall_p01']:.6f} | "
        f"{metric['non_sink_physical_page_recall_mean']:.6f} | "
        f"{metric['exact_refined_output_relative_mse']:.6f} |"
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = _parse_args()
    torch.set_num_threads(args.torch_num_threads)
    torch.backends.cuda.matmul.allow_tf32 = True
    started = time.monotonic()
    model_root = Path(args.model).expanduser().resolve()
    calibration_root = Path(args.calibration_root).expanduser().resolve()
    query_statistics_root = Path(args.query_statistics_root).expanduser().resolve()
    fresh_root = Path(args.fresh_direct_dir).expanduser().resolve()
    c1_root = Path(args.c1_v80).expanduser().resolve()
    initialization_root = Path(args.mse_initialization_root).expanduser().resolve()
    reference_root = Path(args.reference_root).expanduser().resolve()
    fisher_reference_root = Path(args.fisher_reference_root).expanduser().resolve()
    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.work_device)
    layers = _parse_ints(args.layers)
    assert len(layers) == 1
    layer = layers[0]
    cos, sin = _rotary_embeddings(model_root, sequence=32768, device=device)

    fit_root, fit_manifest = _discover_capture(
        calibration_root,
        split="fit",
        layer=layer,
    )
    validation_root, validation_manifest = _discover_capture(
        calibration_root,
        split="validation",
        layer=layer,
    )
    _, fit_rows = _load_direct(fit_root, fit_manifest, layer)
    _, validation_rows = _load_direct(validation_root, validation_manifest, layer)
    fresh_manifest = json.loads((fresh_root / "manifest.json").read_text(encoding="utf-8"))
    fresh_queries, fresh_rows = _load_direct(fresh_root, fresh_manifest, layer)
    fit_query_root, fit_query_manifest = _discover_query_statistics(
        query_statistics_root,
        split="fit",
        layer=layer,
    )
    validation_query_root, validation_query_manifest = _discover_query_statistics(
        query_statistics_root,
        split="validation",
        layer=layer,
    )
    _check_query_alignment(fit_manifest, fit_query_manifest)
    _check_query_alignment(validation_manifest, validation_query_manifest)
    fit_queries, fit_positions = _load_query_observations(
        fit_query_root,
        fit_query_manifest,
        layer=layer,
    )
    validation_queries, validation_positions = _load_query_observations(
        validation_query_root,
        validation_query_manifest,
        layer=layer,
    )
    assert torch.equal(fit_positions, validation_positions)
    fit_queries = fit_queries[: args.fit_documents]
    fit_rows = fit_rows[: args.fit_documents]
    validation_queries = validation_queries[: args.validation_documents]
    validation_rows = validation_rows[: args.validation_documents]
    fresh_queries = fresh_queries[: args.fresh_documents]
    fresh_rows = fresh_rows[: args.fresh_documents]

    c1_manifest = json.loads((c1_root / "results.json").read_text(encoding="utf-8"))
    c1_record = c1_manifest["artifacts"][str(layer)]
    c1_tensors = load_file(str(c1_root / c1_record["file"]), device="cpu")
    value_encoder = c1_tensors["value_coordinate_encoders"]
    output_decoder = c1_tensors["head_output_decoders"]
    if args.base_rank == 16:
        initial_maps = _initial_maps(initialization_root, layer)
    else:
        initial_maps = _fit_base_maps(
            fit_rows,
            value_encoder=value_encoder,
            base_ranks=(args.base_rank,),
            cos=cos,
            sin=sin,
            device=device,
        )[args.base_rank]

    initial_fresh = _evaluate_fresh(
        fresh_queries,
        fresh_rows,
        value_encoder=value_encoder,
        output_decoder=output_decoder,
        base_maps=initial_maps,
        residual_factors={},
        residual_ranks=(0,),
        cos=cos,
        sin=sin,
        page_size=args.page_size,
        page_budget=args.physical_token_budget // args.page_size,
        pinned_prefix_pages=args.pinned_prefix_pages,
        device=device,
    )

    fitted_maps, history, best_epoch = _train_base(
        fit_queries,
        fit_rows,
        validation_queries,
        validation_rows,
        query_positions=fit_positions,
        value_encoder=value_encoder,
        initial_maps=initial_maps,
        cos=cos,
        sin=sin,
        page_size=args.page_size,
        pinned_prefix_pages=args.pinned_prefix_pages,
        epochs=args.epochs,
        documents_per_step=args.documents_per_step,
        factor_learning_rate=args.factor_learning_rate,
        bias_learning_rate=args.bias_learning_rate,
        gradient_clip=args.gradient_clip,
        patience=args.early_stopping_patience,
        seed=args.seed,
        device=device,
        document_loss=_query_weighted_document_loss,
        metric_name=METRIC_NAME,
    )
    fresh = _evaluate_fresh(
        fresh_queries,
        fresh_rows,
        value_encoder=value_encoder,
        output_decoder=output_decoder,
        base_maps=fitted_maps,
        residual_factors={},
        residual_ranks=(0,),
        cos=cos,
        sin=sin,
        page_size=args.page_size,
        page_budget=args.physical_token_budget // args.page_size,
        pinned_prefix_pages=args.pinned_prefix_pages,
        device=device,
    )
    reference = json.loads(
        (reference_root / f"layer_{layer}" / "result.json").read_text(
            encoding="utf-8"
        )
    )["layers"][0]["fresh"]
    fisher_reference = json.loads(
        (fisher_reference_root / f"layer_{layer}" / "result.json").read_text(
            encoding="utf-8"
        )
    )["layers"][0]["fresh"]["r0"]
    references = {
        "MSE pre-RoPE rank16": reference["mse_r0"],
        "Mean-Q RRR pre-RoPE rank16": reference["q_rrr_r0"],
        "Page-Fisher pre-RoPE rank16": fisher_reference,
    }
    references[f"MSE pre-RoPE rank{args.base_rank}"] = initial_fresh["r0"]

    left = torch.stack([item.left for item in fitted_maps])
    right = torch.stack([item.right for item in fitted_maps])
    bias = torch.stack([item.bias for item in fitted_maps])
    artifact = f"layer_{layer:03d}.safetensors"
    temporary = output_root / f"{artifact}.tmp"
    save_file(
        {"base_left": left, "base_right": right, "base_bias": bias},
        str(temporary),
        metadata={"format": FORMAT},
    )
    os.replace(temporary, output_root / artifact)

    result = {
        "format": FORMAT,
        "status": "complete",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(sys.argv),
        "protocol": {
            "model": str(model_root),
            "fixed_payload": str(c1_root),
            "initialization": (
                str(initialization_root)
                if args.base_rank == 16
                else "fresh MSE-RRR fit on the selected calibration rows"
            ),
            "fit_documents": int(fit_rows.shape[0]),
            "validation_documents": int(validation_rows.shape[0]),
            "fresh_documents": int(fresh_rows.shape[0]),
            "queries_per_document": int(fit_queries.shape[1]),
            "query_positions": fit_positions.tolist(),
            "base_parameterization": (
                f"C1-V80 -> rank{args.base_rank} pre-RoPE K -> exact RoPE"
            ),
            "base_rank": args.base_rank,
            "base_objective": (
                "sum of raw QK score squared errors over every captured causal "
                "query, query head, and non-sink key position"
            ),
            "page_size": args.page_size,
            "pinned_prefix_pages": args.pinned_prefix_pages,
            "physical_token_budget_per_kv_group": args.physical_token_budget,
            "epochs": args.epochs,
            "documents_per_step": args.documents_per_step,
            "factor_learning_rate": args.factor_learning_rate,
            "bias_learning_rate": args.bias_learning_rate,
            "gradient_clip": args.gradient_clip,
            "early_stopping_patience": args.early_stopping_patience,
        },
        "layers": [
            {
                "layer": layer,
                "artifact": artifact,
                "base_fit": {"best_epoch": best_epoch, "history": history},
                "fresh": fresh,
                "references": references,
            }
        ],
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
    _write_json(output_root / "result.json", result)
    _write_text(output_root / "summary.md", _markdown(result))
    print(
        f"[Exact query-weighted RRR] wrote {output_root / 'result.json'} and "
        f"{output_root / 'summary.md'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
