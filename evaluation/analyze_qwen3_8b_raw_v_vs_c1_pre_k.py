#!/usr/bin/env python3
"""Compare unrestricted raw-V128 and C1-V80 prediction of pre-RoPE K."""

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

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basisserve.core.c1_v_conditional_k_router import (  # noqa: E402
    AffineReducedRankMap,
    fit_affine_reduced_rank_map,
)
from evaluation.analyze_qwen3_8b_v80_pre_k_spectra import (  # noqa: E402
    Moments,
    _atomic_json,
    _atomic_text,
    _empty_moments,
    _group_moments,
    _moment_from_rows,
    _parse_ints,
    _spectrum_record,
)
from evaluation.eval_qwen3_8b_v80_conditional_residual_router import (  # noqa: E402
    _discover_capture,
    _load_direct,
    _pre_rope_rows,
    _rotary_embeddings,
)


FORMAT = "basisserve.qwen3_8b.raw_v_vs_c1_pre_k.v1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--calibration-root", required=True)
    parser.add_argument("--c1-spectra-root", required=True)
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


def _collect_raw_v_moments(
    rows: torch.Tensor,
    *,
    cos: torch.Tensor,
    sin: torch.Tensor,
    sequence_length: int,
    device: torch.device,
    label: str,
) -> Moments:
    documents, tokens, groups, joint_dim = map(int, rows.shape)
    head_dim = joint_dim // 2
    assert sequence_length <= tokens
    moments = _empty_moments(groups, head_dim, head_dim)
    for document in range(documents):
        print(
            f"    {label} raw-V moments document={document + 1}/{documents}",
            flush=True,
        )
        current = rows[document, :sequence_length].to(
            device=device,
            dtype=torch.float32,
        )
        dense_value = current[..., :head_dim]
        exact_pre_key = _pre_rope_rows(current[..., head_dim:], cos, sin)
        moments.add_(_moment_from_rows(dense_value, exact_pre_key))
        del current, dense_value, exact_pre_key
    return moments


def _fit_unrestricted_maps(
    moments: Moments,
) -> tuple[AffineReducedRankMap, ...]:
    groups, input_dim = map(int, moments.input_sum.shape)
    target_dim = int(moments.target_sum.shape[-1])
    rank = min(input_dim, target_dim)
    return tuple(
        fit_affine_reduced_rank_map(
            **{
                key: value
                for key, value in _group_moments(moments, group).items()
                if key != "target_gram"
            },
            rank=rank,
        )
        for group in range(groups)
    )


@torch.inference_mode()
def _evaluate_unrestricted_maps(
    rows: torch.Tensor,
    maps: tuple[AffineReducedRankMap, ...],
    validation_moments: Moments,
    *,
    cos: torch.Tensor,
    sin: torch.Tensor,
    sequence_length: int,
    device: torch.device,
) -> dict[str, Any]:
    documents, _, groups, joint_dim = map(int, rows.shape)
    head_dim = joint_dim // 2
    left = torch.stack([item.left for item in maps]).to(
        device=device,
        dtype=torch.float32,
    )
    right = torch.stack([item.right for item in maps]).to(
        device=device,
        dtype=torch.float32,
    )
    bias = torch.stack([item.bias for item in maps]).to(
        device=device,
        dtype=torch.float32,
    )
    key_error = torch.zeros(groups, dtype=torch.float64)
    key_energy = torch.zeros(groups, dtype=torch.float64)
    prediction_energy = torch.zeros(groups, dtype=torch.float64)
    key_prediction_dot = torch.zeros(groups, dtype=torch.float64)

    for document in range(documents):
        print(
            f"    held-out raw-V prediction document={document + 1}/{documents}",
            flush=True,
        )
        current = rows[document, :sequence_length].to(
            device=device,
            dtype=torch.float32,
        )
        dense_value = current[..., :head_dim]
        exact_pre_key = _pre_rope_rows(current[..., head_dim:], cos, sin)
        prediction = torch.einsum(
            "tgi,gir,grd->tgd",
            dense_value,
            left,
            right,
        ) + bias
        delta = prediction - exact_pre_key
        key_error += delta.square().sum(dim=(0, 2)).double().cpu()
        key_energy += exact_pre_key.square().sum(dim=(0, 2)).double().cpu()
        prediction_energy += prediction.square().sum(dim=(0, 2)).double().cpu()
        key_prediction_dot += (
            prediction.mul(exact_pre_key).sum(dim=(0, 2)).double().cpu()
        )
        del current, dense_value, exact_pre_key, prediction, delta

    centered_energy = torch.diagonal(
        validation_moments.target_gram,
        dim1=-2,
        dim2=-1,
    ).sum(dim=-1) - validation_moments.target_sum.square().sum(dim=-1) / float(
        validation_moments.row_count
    )
    tiny = torch.finfo(torch.float64).tiny
    cosine = key_prediction_dot / torch.sqrt(
        key_energy.clamp_min(tiny) * prediction_energy.clamp_min(tiny)
    )
    group_rows = [
        {
            "group": group,
            "key_centered_relative_mse": float(
                key_error[group] / centered_energy[group].clamp_min(tiny)
            ),
            "key_raw_relative_mse": float(
                key_error[group] / key_energy[group].clamp_min(tiny)
            ),
            "pre_key_cosine": float(cosine[group]),
        }
        for group in range(groups)
    ]
    return {
        "groups": group_rows,
        "aggregate": {
            key: sum(float(row[key]) for row in group_rows) / len(group_rows)
            for key in group_rows[0]
            if key != "group"
        },
    }


def _load_c1_records(root: Path) -> dict[int, dict[str, Any]]:
    records = {}
    for result_path in sorted(root.glob("shard_*/result.json")):
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        for record in payload["layers"]:
            records[int(record["layer"])] = record
    return records


def _c1_unrestricted_record(record: dict[str, Any]) -> dict[str, Any]:
    fit = record["fit_spectra"]["pre_rope"]["all"]
    heldout_by_rank = record["heldout_prediction"]["pre_rope"]
    unrestricted_rank = max(int(rank) for rank in heldout_by_rank)
    heldout = heldout_by_rank[str(unrestricted_rank)]["all"]
    return {
        "input_dim": unrestricted_rank,
        "fit_predictable_total_fraction_mean": fit["aggregate"][
            "predictable_total_fraction_mean"
        ],
        "fit_unrestricted_centered_relative_mse_mean": fit["aggregate"][
            "unrestricted_centered_relative_mse_mean"
        ],
        "fit_groups": [
            {
                "group": group["group"],
                "predictable_total_fraction": group[
                    "predictable_total_fraction"
                ],
                "unrestricted_centered_relative_mse": group[
                    "unrestricted_centered_relative_mse"
                ],
            }
            for group in fit["groups"]
        ],
        "heldout": heldout,
    }


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Qwen3-8B Raw-V128 versus C1-V80 Pre-K Predictability",
        "",
        "| Layer | Raw-V fit predictable K | C1-V80 fit predictable K | Raw-V held-out centered MSE | C1-V80 held-out centered MSE |",
        "|---:|---:|---:|---:|---:|",
    ]
    for layer in payload["layers"]:
        raw = layer["raw_v128"]
        c1 = layer["c1_v80"]
        lines.append(
            f"| {layer['layer']} | "
            f"{raw['fit_spectra']['aggregate']['predictable_total_fraction_mean']:.6f} | "
            f"{c1['fit_predictable_total_fraction_mean']:.6f} | "
            f"{raw['heldout']['aggregate']['key_centered_relative_mse']:.6f} | "
            f"{c1['heldout']['aggregate']['key_centered_relative_mse']:.6f} |"
        )
    lines.extend(
        (
            "",
            "Raw-V128 and C1-V80 use the same activation windows and pre-RoPE K targets. The fit predictable fraction is the unrestricted affine explained fraction of centered K energy. Held-out MSE evaluates the fit-split affine map on separate windows.",
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
    c1_spectra_root = Path(args.c1_spectra_root).expanduser().resolve()
    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.work_device)
    layers = _parse_ints(args.layers)
    assert 0 <= args.shard_index < args.num_shards
    layers = layers[args.shard_index :: args.num_shards]
    c1_records = _load_c1_records(c1_spectra_root)
    assert all(layer in c1_records for layer in layers)
    cos, sin = _rotary_embeddings(
        model_root,
        sequence=args.sequence_length,
        device=device,
    )
    layer_records = []

    for ordinal, layer in enumerate(layers, start=1):
        layer_started = time.monotonic()
        print(
            f"[Raw V versus C1] layer={layer} ({ordinal}/{len(layers)})",
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
        fit_moments = _collect_raw_v_moments(
            fit_rows,
            cos=cos,
            sin=sin,
            sequence_length=args.sequence_length,
            device=device,
            label="fit",
        )
        validation_moments = _collect_raw_v_moments(
            validation_rows,
            cos=cos,
            sin=sin,
            sequence_length=args.sequence_length,
            device=device,
            label="validation",
        )
        maps = _fit_unrestricted_maps(fit_moments)
        heldout = _evaluate_unrestricted_maps(
            validation_rows,
            maps,
            validation_moments,
            cos=cos,
            sin=sin,
            sequence_length=args.sequence_length,
            device=device,
        )
        raw_spectra = _spectrum_record(fit_moments, ranks=(128,))
        layer_records.append(
            {
                "layer": layer,
                "raw_v128": {
                    "input_dim": 128,
                    "fit_spectra": raw_spectra,
                    "heldout": heldout,
                },
                "c1_v80": _c1_unrestricted_record(c1_records[layer]),
                "seconds": time.monotonic() - layer_started,
            }
        )
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
        del fit_moments, validation_moments, maps, heldout, raw_spectra
        del fit_rows, validation_rows
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
            "c1_spectra_root": str(c1_spectra_root),
            "fit_documents": args.fit_documents,
            "validation_documents": args.validation_documents,
            "sequence_length": args.sequence_length,
            "raw_v_input_dim": 128,
            "c1_v_input_dim": 80,
            "target": "pre-RoPE K128",
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
        f"[Raw V versus C1] wrote {output_root / 'result.json'} and "
        f"{output_root / 'summary.md'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
