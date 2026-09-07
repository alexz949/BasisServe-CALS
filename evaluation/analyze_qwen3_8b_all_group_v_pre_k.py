#!/usr/bin/env python3
"""Measure local/all-group raw-V and C1-V affine pre-K ceilings."""

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

from safetensors.torch import load_file
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.analyze_qwen3_8b_v80_pre_k_spectra import (  # noqa: E402
    _atomic_json,
    _atomic_text,
    _parse_ints,
)
from evaluation.eval_qwen3_8b_v80_conditional_residual_router import (  # noqa: E402
    _discover_capture,
    _load_direct,
    _pre_rope_rows,
    _rotary_embeddings,
    _value_codes,
)


FORMAT = "basisserve.qwen3_8b.all_group_v_pre_k_ceiling.v1"


@dataclass
class AllGroupMoments:
    row_count: int
    raw_sum: torch.Tensor
    c1_sum: torch.Tensor
    target_sum: torch.Tensor
    raw_gram: torch.Tensor
    c1_gram: torch.Tensor
    raw_target_gram: torch.Tensor
    c1_target_gram: torch.Tensor
    target_gram: torch.Tensor

    def add_(self, other: AllGroupMoments) -> None:
        self.row_count += other.row_count
        for name in (
            "raw_sum",
            "c1_sum",
            "target_sum",
            "raw_gram",
            "c1_gram",
            "raw_target_gram",
            "c1_target_gram",
            "target_gram",
        ):
            getattr(self, name).add_(getattr(other, name))


@dataclass(frozen=True)
class AffineAllGroupMap:
    weight: torch.Tensor
    bias: torch.Tensor


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--calibration-root", required=True)
    parser.add_argument("--c1-v80", required=True)
    parser.add_argument("--local-ceilings-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--layers", default=",".join(str(i) for i in range(36)))
    parser.add_argument("--sequence-length", type=int, default=32768)
    parser.add_argument("--fit-documents", type=int, default=64)
    parser.add_argument("--validation-documents", type=int, default=16)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--work-device", default="cuda")
    parser.add_argument("--torch-num-threads", type=int, default=2)
    return parser.parse_args()


def _empty_moments(
    groups: int,
    head_dim: int,
    value_rank: int,
) -> AllGroupMoments:
    raw_dim = groups * head_dim
    c1_dim = groups * value_rank
    return AllGroupMoments(
        row_count=0,
        raw_sum=torch.zeros(raw_dim, dtype=torch.float64),
        c1_sum=torch.zeros(c1_dim, dtype=torch.float64),
        target_sum=torch.zeros(groups, head_dim, dtype=torch.float64),
        raw_gram=torch.zeros(raw_dim, raw_dim, dtype=torch.float64),
        c1_gram=torch.zeros(c1_dim, c1_dim, dtype=torch.float64),
        raw_target_gram=torch.zeros(
            raw_dim,
            groups,
            head_dim,
            dtype=torch.float64,
        ),
        c1_target_gram=torch.zeros(
            c1_dim,
            groups,
            head_dim,
            dtype=torch.float64,
        ),
        target_gram=torch.zeros(
            groups,
            head_dim,
            head_dim,
            dtype=torch.float64,
        ),
    )


def _moments_from_rows(
    dense_value: torch.Tensor,
    c1_codes: torch.Tensor,
    target: torch.Tensor,
) -> AllGroupMoments:
    tokens, groups, head_dim = map(int, dense_value.shape)
    value_rank = int(c1_codes.shape[-1])
    raw = dense_value.reshape(tokens, groups * head_dim)
    c1 = c1_codes.reshape(tokens, groups * value_rank)
    target_flat = target.reshape(tokens, groups * head_dim)
    return AllGroupMoments(
        row_count=tokens,
        raw_sum=raw.sum(dim=0).double().cpu(),
        c1_sum=c1.sum(dim=0).double().cpu(),
        target_sum=target.sum(dim=0).double().cpu(),
        raw_gram=(raw.mT @ raw).double().cpu(),
        c1_gram=(c1.mT @ c1).double().cpu(),
        raw_target_gram=(raw.mT @ target_flat)
        .reshape(groups * head_dim, groups, head_dim)
        .double()
        .cpu(),
        c1_target_gram=(c1.mT @ target_flat)
        .reshape(groups * value_rank, groups, head_dim)
        .double()
        .cpu(),
        target_gram=torch.bmm(
            target.permute(1, 2, 0),
            target.permute(1, 0, 2),
        )
        .double()
        .cpu(),
    )


def _collect_moments(
    rows: torch.Tensor,
    *,
    value_encoder: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    sequence_length: int,
    device: torch.device,
    label: str,
) -> AllGroupMoments:
    documents, tokens, groups, joint_dim = map(int, rows.shape)
    head_dim = joint_dim // 2
    value_rank = int(value_encoder.shape[-1])
    assert sequence_length <= tokens
    moments = _empty_moments(groups, head_dim, value_rank)
    encoder = value_encoder.to(device=device, dtype=torch.float32)
    for document in range(documents):
        print(
            f"    {label} all-group moments document={document + 1}/{documents}",
            flush=True,
        )
        current = rows[document, :sequence_length].to(
            device=device,
            dtype=torch.float32,
        )
        dense_value = current[..., :head_dim]
        target = _pre_rope_rows(current[..., head_dim:], cos, sin)
        c1_codes = _value_codes(dense_value, encoder)
        moments.add_(_moments_from_rows(dense_value, c1_codes, target))
        del current, dense_value, target, c1_codes
    return moments


def _target_centered_energy(moments: AllGroupMoments) -> torch.Tensor:
    return torch.diagonal(
        moments.target_gram,
        dim1=-2,
        dim2=-1,
    ).sum(dim=-1) - moments.target_sum.square().sum(dim=-1) / float(
        moments.row_count
    )


def _fit_map(
    moments: AllGroupMoments,
    *,
    representation: str,
) -> tuple[AffineAllGroupMap, dict[str, Any]]:
    input_sum = getattr(moments, f"{representation}_sum")
    input_gram = getattr(moments, f"{representation}_gram")
    input_target_gram = getattr(
        moments,
        f"{representation}_target_gram",
    )
    count = float(moments.row_count)
    mean_input = input_sum / count
    mean_target = moments.target_sum / count
    covariance = input_gram - count * torch.outer(mean_input, mean_input)
    covariance = 0.5 * (covariance + covariance.mT)
    cross = input_target_gram - count * torch.einsum(
        "i,gd->igd",
        mean_input,
        mean_target,
    )
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    cutoff = (
        torch.finfo(covariance.dtype).eps
        * covariance.shape[0]
        * eigenvalues.abs().amax()
    )
    inverse_values = torch.where(
        eigenvalues > cutoff,
        eigenvalues.clamp_min(torch.finfo(eigenvalues.dtype).tiny).reciprocal(),
        torch.zeros_like(eigenvalues),
    )
    inverse = (eigenvectors * inverse_values.unsqueeze(0)) @ eigenvectors.mT
    weight = torch.einsum("ij,jgd->igd", inverse, cross)
    bias = mean_target - torch.einsum("i,igd->gd", mean_input, weight)
    explained = torch.einsum("igd,igd->g", cross, weight)
    target_energy = _target_centered_energy(moments)
    residual = target_energy - explained
    group_rows = [
        {
            "group": group,
            "target_centered_energy": float(target_energy[group]),
            "explained_energy": float(explained[group]),
            "predictable_total_fraction": float(
                explained[group] / target_energy[group]
            ),
            "conditional_relative_trace": float(
                residual[group] / target_energy[group]
            ),
        }
        for group in range(int(target_energy.numel()))
    ]
    record = {
        "input_dim": int(input_sum.numel()),
        "numerical_input_rank": int((eigenvalues > cutoff).sum()),
        "groups": group_rows,
        "aggregate": {
            key: sum(float(row[key]) for row in group_rows) / len(group_rows)
            for key in (
                "predictable_total_fraction",
                "conditional_relative_trace",
            )
        },
    }
    return AffineAllGroupMap(weight=weight, bias=bias), record


def _empty_evaluation(groups: int) -> dict[str, torch.Tensor]:
    return {
        name: torch.zeros(groups, dtype=torch.float64)
        for name in (
            "key_error",
            "key_energy",
            "prediction_energy",
            "key_prediction_dot",
        )
    }


@torch.inference_mode()
def _evaluate_maps(
    rows: torch.Tensor,
    maps: dict[str, AffineAllGroupMap],
    validation_moments: AllGroupMoments,
    *,
    value_encoder: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    sequence_length: int,
    device: torch.device,
) -> dict[str, Any]:
    documents, _, groups, joint_dim = map(int, rows.shape)
    head_dim = joint_dim // 2
    value_rank = int(value_encoder.shape[-1])
    encoder = value_encoder.to(device=device, dtype=torch.float32)
    factors = {
        name: (
            item.weight.to(device=device, dtype=torch.float32),
            item.bias.to(device=device, dtype=torch.float32),
        )
        for name, item in maps.items()
    }
    accumulators = {name: _empty_evaluation(groups) for name in maps}

    for document in range(documents):
        print(
            f"    held-out all-group prediction document={document + 1}/{documents}",
            flush=True,
        )
        current = rows[document, :sequence_length].to(
            device=device,
            dtype=torch.float32,
        )
        dense_value = current[..., :head_dim]
        target = _pre_rope_rows(current[..., head_dim:], cos, sin)
        c1_codes = _value_codes(dense_value, encoder)
        inputs = {
            "raw": dense_value.reshape(sequence_length, groups * head_dim),
            "c1": c1_codes.reshape(sequence_length, groups * value_rank),
        }
        for name, input_rows in inputs.items():
            weight, bias = factors[name]
            prediction = torch.einsum("ti,igd->tgd", input_rows, weight) + bias
            delta = prediction - target
            accumulator = accumulators[name]
            accumulator["key_error"] += (
                delta.square().sum(dim=(0, 2)).double().cpu()
            )
            accumulator["key_energy"] += (
                target.square().sum(dim=(0, 2)).double().cpu()
            )
            accumulator["prediction_energy"] += (
                prediction.square().sum(dim=(0, 2)).double().cpu()
            )
            accumulator["key_prediction_dot"] += (
                prediction.mul(target).sum(dim=(0, 2)).double().cpu()
            )
            del prediction, delta
        del current, dense_value, target, c1_codes, inputs

    centered_energy = _target_centered_energy(validation_moments)
    tiny = torch.finfo(torch.float64).tiny
    result = {}
    for name, accumulator in accumulators.items():
        cosine = accumulator["key_prediction_dot"] / torch.sqrt(
            accumulator["key_energy"].clamp_min(tiny)
            * accumulator["prediction_energy"].clamp_min(tiny)
        )
        group_rows = [
            {
                "group": group,
                "key_centered_relative_mse": float(
                    accumulator["key_error"][group]
                    / centered_energy[group].clamp_min(tiny)
                ),
                "key_raw_relative_mse": float(
                    accumulator["key_error"][group]
                    / accumulator["key_energy"][group].clamp_min(tiny)
                ),
                "pre_key_cosine": float(cosine[group]),
            }
            for group in range(groups)
        ]
        result[name] = {
            "groups": group_rows,
            "aggregate": {
                key: sum(float(row[key]) for row in group_rows) / len(group_rows)
                for key in group_rows[0]
                if key != "group"
            },
        }
    return result


def _load_local_records(root: Path) -> dict[int, dict[str, Any]]:
    records = {}
    for result_path in sorted(root.glob("shard_*/result.json")):
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        for record in payload["layers"]:
            records[int(record["layer"])] = record
    return records


def _compact_local_record(record: dict[str, Any]) -> dict[str, Any]:
    raw = record["raw_v128"]
    c1 = record["c1_v80"]
    return {
        "local_raw_v128": {
            "fit_predictable_total_fraction": raw["fit_spectra"]["aggregate"][
                "predictable_total_fraction_mean"
            ],
            "heldout": raw["heldout"],
        },
        "local_c1_v80": {
            "fit_predictable_total_fraction": c1[
                "fit_predictable_total_fraction_mean"
            ],
            "heldout": c1["heldout"],
        },
    }


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Qwen3-8B Local/All-Group V-to-Pre-K Ceilings",
        "",
        "| Layer | Local C1-V80 | Local raw V128 | All C1-V640 | All raw V1024 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for layer in payload["layers"]:
        lines.append(
            f"| {layer['layer']} | "
            f"{layer['local_c1_v80']['fit_predictable_total_fraction']:.6f} | "
            f"{layer['local_raw_v128']['fit_predictable_total_fraction']:.6f} | "
            f"{layer['all_c1_v640']['fit']['aggregate']['predictable_total_fraction']:.6f} | "
            f"{layer['all_raw_v1024']['fit']['aggregate']['predictable_total_fraction']:.6f} |"
        )
    lines.extend(
        (
            "",
            "All four values are centered affine explained fractions for the same pre-RoPE K targets and activation windows.",
            "",
        )
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
    local_root = Path(args.local_ceilings_root).expanduser().resolve()
    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.work_device)
    layers = _parse_ints(args.layers)
    assert 0 <= args.shard_index < args.num_shards
    layers = layers[args.shard_index :: args.num_shards]
    local_records = _load_local_records(local_root)
    assert all(layer in local_records for layer in layers)
    c1_manifest = json.loads((c1_root / "results.json").read_text(encoding="utf-8"))
    cos, sin = _rotary_embeddings(
        model_root,
        sequence=args.sequence_length,
        device=device,
    )
    layer_records = []

    for ordinal, layer in enumerate(layers, start=1):
        layer_started = time.monotonic()
        print(
            f"[All-group V ceiling] layer={layer} ({ordinal}/{len(layers)})",
            flush=True,
        )
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
        _, validation_rows = _load_direct(
            validation_root,
            validation_manifest,
            layer,
        )
        fit_rows = fit_rows[: args.fit_documents]
        validation_rows = validation_rows[: args.validation_documents]
        c1_artifact = c1_manifest["artifacts"][str(layer)]
        c1_tensors = load_file(str(c1_root / c1_artifact["file"]), device="cpu")
        value_encoder = c1_tensors["value_coordinate_encoders"]
        fit_moments = _collect_moments(
            fit_rows,
            value_encoder=value_encoder,
            cos=cos,
            sin=sin,
            sequence_length=args.sequence_length,
            device=device,
            label="fit",
        )
        validation_moments = _collect_moments(
            validation_rows,
            value_encoder=value_encoder,
            cos=cos,
            sin=sin,
            sequence_length=args.sequence_length,
            device=device,
            label="validation",
        )
        raw_map, raw_fit = _fit_map(fit_moments, representation="raw")
        c1_map, c1_fit = _fit_map(fit_moments, representation="c1")
        heldout = _evaluate_maps(
            validation_rows,
            {"raw": raw_map, "c1": c1_map},
            validation_moments,
            value_encoder=value_encoder,
            cos=cos,
            sin=sin,
            sequence_length=args.sequence_length,
            device=device,
        )
        record = {
            "layer": layer,
            **_compact_local_record(local_records[layer]),
            "all_c1_v640": {"fit": c1_fit, "heldout": heldout["c1"]},
            "all_raw_v1024": {"fit": raw_fit, "heldout": heldout["raw"]},
            "seconds": time.monotonic() - layer_started,
        }
        layer_records.append(record)
        partial = {
            "format": FORMAT,
            "status": "running",
            "command": shlex.join(sys.argv),
            "protocol": {
                "fit_documents": args.fit_documents,
                "validation_documents": args.validation_documents,
                "sequence_length": args.sequence_length,
                "shard_index": args.shard_index,
                "num_shards": args.num_shards,
            },
            "layers": layer_records,
            "elapsed_seconds": time.monotonic() - started,
        }
        _atomic_json(output_root / "result.json", partial)
        _atomic_text(output_root / "summary.md", _markdown(partial))
        del fit_rows, validation_rows, c1_tensors, value_encoder
        del fit_moments, validation_moments, raw_map, c1_map, heldout
        if device.type == "cuda":
            torch.cuda.empty_cache()

    result = {
        "format": FORMAT,
        "status": "complete",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(sys.argv),
        "protocol": {
            "model": str(model_root),
            "calibration_root": str(calibration_root),
            "c1_v80": str(c1_root),
            "local_ceilings_root": str(local_root),
            "fit_documents": args.fit_documents,
            "validation_documents": args.validation_documents,
            "sequence_length": args.sequence_length,
            "target": "per-group pre-RoPE K128",
            "all_c1_input_dim": 640,
            "all_raw_v_input_dim": 1024,
            "shard_index": args.shard_index,
            "num_shards": args.num_shards,
        },
        "layers": layer_records,
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
    _atomic_json(output_root / "result.json", result)
    _atomic_text(output_root / "summary.md", _markdown(result))
    print(
        f"[All-group V ceiling] wrote {output_root / 'result.json'} and "
        f"{output_root / 'summary.md'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
