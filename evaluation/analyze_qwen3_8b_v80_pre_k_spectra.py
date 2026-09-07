#!/usr/bin/env python3
"""Analyze C1-V80 to pre/post-RoPE Key predictive spectra."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
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

from basisserve.core.c1_v_conditional_k_router import (  # noqa: E402
    AffinePredictiveSpectrum,
    AffineReducedRankMap,
    affine_predictive_spectrum,
    fit_affine_reduced_rank_map,
)
from evaluation.eval_qwen3_8b_v80_conditional_residual_router import (  # noqa: E402
    _discover_capture,
    _load_direct,
    _post_rope_rows,
    _pre_rope_rows,
    _rotary_embeddings,
    _value_codes,
)


FORMAT = "basisserve.qwen3_8b.v80_pre_k_predictive_spectra.v1"
CONVENTIONS = ("pre_rope", "post_rope")


@dataclass
class Moments:
    row_count: int
    input_sum: torch.Tensor
    target_sum: torch.Tensor
    input_gram: torch.Tensor
    input_target_gram: torch.Tensor
    target_gram: torch.Tensor

    def add_(self, other: Moments) -> None:
        self.row_count += other.row_count
        self.input_sum.add_(other.input_sum)
        self.target_sum.add_(other.target_sum)
        self.input_gram.add_(other.input_gram)
        self.input_target_gram.add_(other.input_target_gram)
        self.target_gram.add_(other.target_gram)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--calibration-root", required=True)
    parser.add_argument("--c1-v80", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--layers", default="0,17,35")
    parser.add_argument("--ranks", default="0,4,8,16,24,32,48,64,80")
    parser.add_argument(
        "--position-boundaries",
        default="0,2048,8192,16384,32768",
    )
    parser.add_argument("--fit-documents", type=int, default=0)
    parser.add_argument("--validation-documents", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--work-device", default="cuda")
    parser.add_argument("--torch-num-threads", type=int, default=2)
    return parser.parse_args()


def _parse_ints(specification: str) -> tuple[int, ...]:
    return tuple(sorted({int(item) for item in specification.split(",") if item}))


def _atomic_text(path: Path, contents: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(contents, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _atomic_safetensors(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    save_file(tensors, str(temporary), metadata={"format": FORMAT})
    os.replace(temporary, path)


def _moment_tensors(
    moments: dict[str, dict[str, Moments]],
) -> dict[str, torch.Tensor]:
    tensors = {}
    for convention in CONVENTIONS:
        moment = moments[convention]["all"]
        prefix = f"fit_{convention}"
        tensors[f"{prefix}_row_count"] = torch.tensor(
            moment.row_count,
            dtype=torch.int64,
        )
        for name in (
            "input_sum",
            "target_sum",
            "input_gram",
            "input_target_gram",
            "target_gram",
        ):
            tensors[f"{prefix}_{name}"] = getattr(moment, name).contiguous()
    return tensors


def _empty_moments(groups: int, input_dim: int, target_dim: int) -> Moments:
    return Moments(
        row_count=0,
        input_sum=torch.zeros(groups, input_dim, dtype=torch.float64),
        target_sum=torch.zeros(groups, target_dim, dtype=torch.float64),
        input_gram=torch.zeros(
            groups,
            input_dim,
            input_dim,
            dtype=torch.float64,
        ),
        input_target_gram=torch.zeros(
            groups,
            input_dim,
            target_dim,
            dtype=torch.float64,
        ),
        target_gram=torch.zeros(
            groups,
            target_dim,
            target_dim,
            dtype=torch.float64,
        ),
    )


def _moment_from_rows(codes: torch.Tensor, targets: torch.Tensor) -> Moments:
    group_codes = codes.permute(1, 0, 2)
    group_targets = targets.permute(1, 0, 2)
    return Moments(
        row_count=int(codes.shape[0]),
        input_sum=group_codes.sum(dim=1).double().cpu(),
        target_sum=group_targets.sum(dim=1).double().cpu(),
        input_gram=torch.bmm(group_codes.mT, group_codes).double().cpu(),
        input_target_gram=torch.bmm(
            group_codes.mT,
            group_targets,
        )
        .double()
        .cpu(),
        target_gram=torch.bmm(group_targets.mT, group_targets).double().cpu(),
    )


def _bucket_ranges(boundaries: tuple[int, ...]) -> dict[str, tuple[int, int]]:
    return {
        f"{start}_{stop}": (start, stop)
        for start, stop in zip(boundaries[:-1], boundaries[1:])
    }


def _collect_moments(
    rows: torch.Tensor,
    *,
    value_encoder: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    buckets: dict[str, tuple[int, int]],
    device: torch.device,
    label: str,
) -> dict[str, dict[str, Moments]]:
    documents, tokens, groups, joint_dim = map(int, rows.shape)
    head_dim = joint_dim // 2
    value_rank = int(value_encoder.shape[-1])
    result = {
        convention: {
            name: _empty_moments(groups, value_rank, head_dim) for name in buckets
        }
        for convention in CONVENTIONS
    }
    encoder = value_encoder.to(device=device, dtype=torch.float32)
    analyzed_tokens = max(stop for _, stop in buckets.values())
    for document in range(documents):
        print(
            f"    {label} moments document={document + 1}/{documents}",
            flush=True,
        )
        current = rows[document, :analyzed_tokens].to(
            device=device,
            dtype=torch.float32,
        )
        dense_value = current[..., :head_dim]
        exact_post = current[..., head_dim:]
        exact_pre = _pre_rope_rows(exact_post, cos, sin)
        codes = _value_codes(dense_value, encoder)
        targets = {"pre_rope": exact_pre, "post_rope": exact_post}
        for name, (start, stop) in buckets.items():
            assert 0 <= start < stop <= tokens
            for convention, target in targets.items():
                result[convention][name].add_(
                    _moment_from_rows(codes[start:stop], target[start:stop])
                )
        del current, dense_value, exact_post, exact_pre, codes
    for convention in CONVENTIONS:
        combined = _empty_moments(groups, value_rank, head_dim)
        for item in result[convention].values():
            combined.add_(item)
        result[convention]["all"] = combined
    return result


def _group_moments(moment: Moments, group: int) -> dict[str, Any]:
    return {
        "row_count": moment.row_count,
        "input_sum": moment.input_sum[group],
        "target_sum": moment.target_sum[group],
        "input_gram": moment.input_gram[group],
        "input_target_gram": moment.input_target_gram[group],
        "target_gram": moment.target_gram[group],
    }


def _rank_curve(
    spectrum: AffinePredictiveSpectrum,
    ranks: tuple[int, ...],
) -> dict[str, dict[str, float]]:
    predictable = max(spectrum.predictable_energy, torch.finfo(torch.float64).tiny)
    target = max(
        spectrum.target_centered_energy,
        torch.finfo(torch.float64).tiny,
    )
    return {
        str(rank): {
            "captured_energy": spectrum.captured_energy(rank),
            "captured_total_fraction": spectrum.captured_energy(rank) / target,
            "captured_predictable_fraction": (
                spectrum.captured_energy(rank) / predictable
            ),
            "centered_relative_mse": spectrum.residual_energy(rank) / target,
        }
        for rank in ranks
    }


def _spectrum_record(
    moment: Moments,
    *,
    ranks: tuple[int, ...],
) -> dict[str, Any]:
    groups = []
    for group in range(int(moment.input_sum.shape[0])):
        spectrum = affine_predictive_spectrum(**_group_moments(moment, group))
        target = max(
            spectrum.target_centered_energy,
            torch.finfo(torch.float64).tiny,
        )
        groups.append(
            {
                "group": group,
                "target_centered_energy": spectrum.target_centered_energy,
                "predictable_energy": spectrum.predictable_energy,
                "predictable_total_fraction": spectrum.predictable_energy / target,
                "unrestricted_centered_relative_mse": (
                    spectrum.unrestricted_residual_energy / target
                ),
                "canonical_correlations": (spectrum.canonical_correlations.tolist()),
                "predictive_energy_fractions": (
                    spectrum.predictive_singular_values.square() / target
                ).tolist(),
                "ranks": _rank_curve(spectrum, ranks),
            }
        )
    aggregate_ranks = {}
    for rank in ranks:
        rows = [record["ranks"][str(rank)] for record in groups]
        aggregate_ranks[str(rank)] = {
            key: sum(float(row[key]) for row in rows) / len(rows) for key in rows[0]
        }
    return {
        "row_count_per_group": moment.row_count,
        "groups": groups,
        "aggregate": {
            "predictable_total_fraction_mean": sum(
                float(record["predictable_total_fraction"]) for record in groups
            )
            / len(groups),
            "predictable_total_fraction_minimum": min(
                float(record["predictable_total_fraction"]) for record in groups
            ),
            "unrestricted_centered_relative_mse_mean": sum(
                float(record["unrestricted_centered_relative_mse"]) for record in groups
            )
            / len(groups),
            "ranks": aggregate_ranks,
        },
    }


def _fit_maps(
    moments: dict[str, dict[str, Moments]],
    *,
    ranks: tuple[int, ...],
) -> dict[str, dict[int, tuple[AffineReducedRankMap, ...]]]:
    maps = {}
    for convention in CONVENTIONS:
        moment = moments[convention]["all"]
        maps[convention] = {
            rank: tuple(
                fit_affine_reduced_rank_map(
                    **{
                        key: value
                        for key, value in _group_moments(moment, group).items()
                        if key != "target_gram"
                    },
                    rank=rank,
                )
                for group in range(int(moment.input_sum.shape[0]))
            )
            for rank in ranks
            if rank > 0
        }
    return maps


def _empty_evaluation(groups: int) -> dict[str, torch.Tensor]:
    return {
        name: torch.zeros(groups, dtype=torch.float64)
        for name in (
            "key_error",
            "key_raw_energy",
            "prediction_energy",
            "key_prediction_dot",
            "score_error",
            "score_energy",
        )
    }


def _evaluate_maps(
    rows: torch.Tensor,
    queries_raw: torch.Tensor,
    *,
    value_encoder: torch.Tensor,
    maps: dict[str, dict[int, tuple[AffineReducedRankMap, ...]]],
    cos: torch.Tensor,
    sin: torch.Tensor,
    buckets: dict[str, tuple[int, int]],
    device: torch.device,
) -> dict[str, dict[str, dict[str, Any]]]:
    documents, _, groups, joint_dim = map(int, rows.shape)
    query_heads = int(queries_raw.shape[1])
    head_dim = joint_dim // 2
    heads_per_group = query_heads // groups
    scaling = head_dim**-0.5
    encoder = value_encoder.to(device=device, dtype=torch.float32)
    factors = {
        convention: {
            rank: (
                torch.stack([item.left for item in group_maps]).to(
                    device=device,
                    dtype=torch.float32,
                ),
                torch.stack([item.right for item in group_maps]).to(
                    device=device,
                    dtype=torch.float32,
                ),
                torch.stack([item.bias for item in group_maps]).to(
                    device=device,
                    dtype=torch.float32,
                ),
            )
            for rank, group_maps in rank_maps.items()
        }
        for convention, rank_maps in maps.items()
    }
    accumulators = {
        convention: {
            str(rank): {name: _empty_evaluation(groups) for name in (*buckets, "all")}
            for rank in rank_maps
        }
        for convention, rank_maps in maps.items()
    }
    analyzed_tokens = max(stop for _, stop in buckets.values())
    all_buckets = {**buckets, "all": (0, analyzed_tokens)}
    for document in range(documents):
        print(
            f"    held-out evaluation document={document + 1}/{documents}",
            flush=True,
        )
        current = rows[document, :analyzed_tokens].to(
            device=device,
            dtype=torch.float32,
        )
        queries = queries_raw[document].to(device=device, dtype=torch.float32)
        dense_value = current[..., :head_dim]
        exact_post = current[..., head_dim:]
        exact_pre = _pre_rope_rows(exact_post, cos, sin)
        codes = _value_codes(dense_value, encoder)
        for convention, rank_factors in factors.items():
            exact_target = exact_pre if convention == "pre_rope" else exact_post
            for rank, (left, right, bias) in rank_factors.items():
                prediction = torch.einsum("tgi,gir,grd->tgd", codes, left, right) + bias
                predicted_post = (
                    _post_rope_rows(prediction, cos, sin)
                    if convention == "pre_rope"
                    else prediction
                )
                for name, (start, stop) in all_buckets.items():
                    target_slice = exact_target[start:stop]
                    prediction_slice = prediction[start:stop]
                    post_slice = exact_post[start:stop]
                    predicted_post_slice = predicted_post[start:stop]
                    key_delta = prediction_slice - target_slice
                    accumulator = accumulators[convention][str(rank)][name]
                    accumulator["key_error"] += (
                        key_delta.square().sum(dim=(0, 2)).double().cpu()
                    )
                    accumulator["key_raw_energy"] += (
                        target_slice.square().sum(dim=(0, 2)).double().cpu()
                    )
                    accumulator["prediction_energy"] += (
                        prediction_slice.square().sum(dim=(0, 2)).double().cpu()
                    )
                    accumulator["key_prediction_dot"] += (
                        prediction_slice.mul(target_slice)
                        .sum(dim=(0, 2))
                        .double()
                        .cpu()
                    )
                    for group in range(groups):
                        first = group * heads_per_group
                        stop_head = first + heads_per_group
                        exact_scores = scaling * (
                            queries[first:stop_head] @ post_slice[:, group].mT
                        )
                        predicted_scores = scaling * (
                            queries[first:stop_head] @ predicted_post_slice[:, group].mT
                        )
                        accumulator["score_error"][group] += float(
                            (predicted_scores - exact_scores).square().sum()
                        )
                        accumulator["score_energy"][group] += float(
                            exact_scores.square().sum()
                        )
                del prediction, predicted_post
        del current, queries, dense_value, exact_post, exact_pre, codes
    return accumulators


def _finish_evaluation(
    accumulators: dict[str, dict[str, dict[str, Any]]],
    validation_moments: dict[str, dict[str, Moments]],
) -> dict[str, Any]:
    result = {}
    tiny = torch.finfo(torch.float64).tiny
    for convention, rank_records in accumulators.items():
        result[convention] = {}
        for rank, bucket_records in rank_records.items():
            result[convention][rank] = {}
            for bucket, accumulator in bucket_records.items():
                moment = validation_moments[convention][bucket]
                centered_energy = torch.diagonal(
                    moment.target_gram,
                    dim1=-2,
                    dim2=-1,
                ).sum(dim=-1) - moment.target_sum.square().sum(dim=-1) / float(
                    moment.row_count
                )
                cosine = accumulator["key_prediction_dot"] / torch.sqrt(
                    accumulator["key_raw_energy"].clamp_min(tiny)
                    * accumulator["prediction_energy"].clamp_min(tiny)
                )
                group_rows = []
                for group in range(int(accumulator["key_error"].numel())):
                    group_rows.append(
                        {
                            "group": group,
                            "key_centered_relative_mse": float(
                                accumulator["key_error"][group]
                                / centered_energy[group].clamp_min(tiny)
                            ),
                            "key_raw_relative_mse": float(
                                accumulator["key_error"][group]
                                / accumulator["key_raw_energy"][group].clamp_min(tiny)
                            ),
                            "post_key_cosine": float(cosine[group]),
                            "paired_final_query_score_nmse": float(
                                accumulator["score_error"][group]
                                / accumulator["score_energy"][group].clamp_min(tiny)
                            ),
                        }
                    )
                result[convention][rank][bucket] = {
                    "groups": group_rows,
                    "aggregate": {
                        key: sum(float(row[key]) for row in group_rows)
                        / len(group_rows)
                        for key in group_rows[0]
                        if key != "group"
                    },
                }
    return result


def _markdown(payload: dict[str, Any]) -> str:
    protocol = payload["protocol"]
    lines = [
        "# Qwen3-8B C1-V80 to Key Predictive Spectra",
        "",
        "| Layer | Convention | Rank | Fit captured K energy | Held-out K MSE | Held-out score NMSE |",
        "|---:|:---|---:|---:|---:|---:|",
    ]
    ranks = payload["protocol"]["ranks"]
    displayed_rank = 16 if 16 in ranks else max(ranks)
    for layer in payload["layers"]:
        for convention in CONVENTIONS:
            fit = layer["fit_spectra"][convention]["all"]["aggregate"]
            heldout = layer["heldout_prediction"][convention][str(displayed_rank)][
                "all"
            ]["aggregate"]
            captured = fit["ranks"][str(displayed_rank)]["captured_total_fraction"]
            lines.append(
                f"| {layer['layer']} | {convention} | {displayed_rank} | "
                f"{captured:.6f} | "
                f"{heldout['key_centered_relative_mse']:.6f} | "
                f"{heldout['paired_final_query_score_nmse']:.6f} |"
            )
    lines.extend(
        [
            "",
            f"Fit spectra use {protocol['fit_documents']} independent C4 windows over positions 0--{protocol['position_boundaries'][-1]}. Held-out prediction uses {protocol['validation_documents']} independent C4 windows and their paired final-token Queries.",
            "",
        ]
    )
    return "\n".join(lines)


@torch.inference_mode()
def main() -> None:
    args = _parse_args()
    torch.set_num_threads(args.torch_num_threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    started = time.monotonic()
    model_root = Path(args.model).expanduser().resolve()
    calibration_root = Path(args.calibration_root).expanduser().resolve()
    c1_root = Path(args.c1_v80).expanduser().resolve()
    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    layers = _parse_ints(args.layers)
    assert 0 <= args.shard_index < args.num_shards
    layers = layers[args.shard_index :: args.num_shards]
    ranks = _parse_ints(args.ranks)
    boundaries = _parse_ints(args.position_boundaries)
    assert boundaries[0] == 0
    buckets = _bucket_ranges(boundaries)
    sequence = boundaries[-1]
    device = torch.device(args.work_device)
    cos, sin = _rotary_embeddings(
        model_root,
        sequence=sequence,
        device=device,
    )
    c1_manifest = json.loads((c1_root / "results.json").read_text(encoding="utf-8"))
    layer_records = []
    result_path = output_root / "result.json"
    summary_path = output_root / "summary.md"
    for ordinal, layer in enumerate(layers, start=1):
        layer_started = time.monotonic()
        print(f"[V80 spectra] layer={layer} ({ordinal}/{len(layers)})", flush=True)
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
        fit_queries, fit_rows = _load_direct(fit_root, fit_manifest, layer)
        validation_queries, validation_rows = _load_direct(
            validation_root,
            validation_manifest,
            layer,
        )
        if args.fit_documents:
            fit_queries = fit_queries[: args.fit_documents]
            fit_rows = fit_rows[: args.fit_documents]
        if args.validation_documents:
            validation_queries = validation_queries[: args.validation_documents]
            validation_rows = validation_rows[: args.validation_documents]
        c1_artifact = c1_manifest["artifacts"][str(layer)]
        c1_tensors = load_file(str(c1_root / c1_artifact["file"]), device="cpu")
        value_encoder = c1_tensors["value_coordinate_encoders"]
        fit_moments = _collect_moments(
            fit_rows,
            value_encoder=value_encoder,
            cos=cos,
            sin=sin,
            buckets=buckets,
            device=device,
            label="fit",
        )
        validation_moments = _collect_moments(
            validation_rows,
            value_encoder=value_encoder,
            cos=cos,
            sin=sin,
            buckets=buckets,
            device=device,
            label="validation",
        )
        maps = _fit_maps(fit_moments, ranks=ranks)
        evaluation = _evaluate_maps(
            validation_rows,
            validation_queries,
            value_encoder=value_encoder,
            maps=maps,
            cos=cos,
            sin=sin,
            buckets=buckets,
            device=device,
        )
        layer_records.append(
            {
                "layer": layer,
                "fit_moments": f"layer_{layer:03d}_fit_moments.safetensors",
                "fit_spectra": {
                    convention: {
                        bucket: _spectrum_record(moment, ranks=ranks)
                        for bucket, moment in convention_moments.items()
                    }
                    for convention, convention_moments in fit_moments.items()
                },
                "validation_spectra": {
                    convention: {
                        bucket: _spectrum_record(moment, ranks=ranks)
                        for bucket, moment in convention_moments.items()
                    }
                    for convention, convention_moments in validation_moments.items()
                },
                "heldout_prediction": _finish_evaluation(
                    evaluation,
                    validation_moments,
                ),
                "seconds": time.monotonic() - layer_started,
            }
        )
        _atomic_safetensors(
            output_root / f"layer_{layer:03d}_fit_moments.safetensors",
            _moment_tensors(fit_moments),
        )
        partial = {
            "format": FORMAT,
            "status": "running",
            "command": shlex.join(sys.argv),
            "protocol": {
                "ranks": list(ranks),
                "position_boundaries": list(boundaries),
                "fit_documents": args.fit_documents if args.fit_documents else 64,
                "validation_documents": (
                    args.validation_documents if args.validation_documents else 16
                ),
            },
            "layers": layer_records,
            "elapsed_seconds": time.monotonic() - started,
        }
        _atomic_json(result_path, partial)
        _atomic_text(summary_path, _markdown(partial))
        del fit_moments, validation_moments, maps, evaluation, c1_tensors
        del fit_queries, fit_rows, validation_queries, validation_rows
        if device.type == "cuda":
            torch.cuda.empty_cache()
    result = {
        "format": FORMAT,
        "status": "complete",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(sys.argv),
        "protocol": {
            "model": str(model_root),
            "c1_v80": str(c1_root),
            "fit_documents": args.fit_documents if args.fit_documents else 64,
            "validation_documents": (
                args.validation_documents if args.validation_documents else 16
            ),
            "analyzed_positions": [0, sequence],
            "shard_index": args.shard_index,
            "num_shards": args.num_shards,
            "ranks": list(ranks),
            "position_boundaries": list(boundaries),
            "target_conventions": list(CONVENTIONS),
            "score_metric": "paired held-out final-token Query raw-score NMSE",
        },
        "layers": layer_records,
        "elapsed_seconds": time.monotonic() - started,
        "environment": {
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_devices": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ],
        },
    }
    _atomic_json(result_path, result)
    _atomic_text(summary_path, _markdown(result))
    print(f"[V80 spectra] wrote {result_path} and {summary_path}", flush=True)


if __name__ == "__main__":
    main()
