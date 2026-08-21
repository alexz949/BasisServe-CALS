#!/usr/bin/env python3
"""Measure post-state Qwen3.5 GDN SiLU effective sparsity offline."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import sys
import time
from typing import Any, Mapping, Sequence

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basisserve.calibration.mlp_snapshots import SPLITS
from basisserve.core.qwen35_gdn_foldability import (
    Qwen35GDNFoldabilityGeometry,
    SNAPSHOT_FORMAT,
    validate_snapshot_matrices,
)
from basisserve.diagnostics.qwen35_gated_attention_sparsity import (
    SCORE_METHODS,
    RuntimeActivations,
    reconstruct_gdn_silu_runtime,
)
from evaluation.analyze_qwen35_gated_attention_sparsity import (
    AUDIT_SCOPE,
    MAIN_SCOPE,
    _csv_ints,
    _document_balanced_audit_indices,
    _evaluate_analytic_random,
    _evaluate_greedy_audit,
    _evaluate_ranked_method,
    _file_sha256,
    _gate_statistics,
    _git_commit,
    _installed_version,
    _json_text,
    _methods,
    _model_id_and_revision,
    _plot,
    _random_expectation_validation,
    _ratios,
    _records_by_layer,
    _require_sha256,
    _summary_csv,
    _thresholds,
    _validate_document_provenance,
    _write_text,
)


FORMAT = "basisserve.qwen35.gdn_silu_effective_sparsity.v1"
ROW_FORMAT = "basisserve.qwen35.gdn_silu_effective_sparsity.row.v1"
FREQUENCY_FORMAT = "basisserve.qwen35.gdn_silu_selection_frequency.v1"
DEFAULT_METHODS = ("random", *SCORE_METHODS, "random_expected")
DEFAULT_KEEP_RATIOS = (0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0)
DEFAULT_GREEDY_RATIOS = (0.1, 0.2, 0.3)
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-path")
    parser.add_argument("--layers", default="representative")
    parser.add_argument("--num-tokens", type=int, default=8192)
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument(
        "--keep-ratios", default=",".join(map(str, DEFAULT_KEEP_RATIOS))
    )
    parser.add_argument("--random-seeds", default="0,1,2")
    parser.add_argument("--audit-num-tokens", type=int, default=128)
    parser.add_argument("--audit-seed", type=int, default=20260808)
    parser.add_argument(
        "--greedy-keep-ratios", default=",".join(map(str, DEFAULT_GREEDY_RATIOS))
    )
    parser.add_argument("--skip-greedy", action="store_true")
    parser.add_argument("--strict-greedy", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--gate-thresholds", default="1e-4,1e-3,1e-2,5e-2,1e-1")
    parser.add_argument(
        "--activation-thresholds", default="1e-4,1e-3,1e-2,5e-2,1e-1"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--torch-num-threads", type=int, default=4)
    return parser.parse_args()


def _geometry_from_manifest(
    manifest: Mapping[str, Any],
) -> Qwen35GDNFoldabilityGeometry:
    source = manifest["geometry"]
    geometry = Qwen35GDNFoldabilityGeometry(
        hidden_size=int(source["hidden_size"]),
        num_value_heads=int(source["num_value_heads"]),
        value_head_dim=int(source["value_head_dim"]),
        rms_norm_eps=float(source["rms_norm_eps"]),
        layer_types=tuple(map(str, source["layer_types"])),
    )
    geometry.validate()
    if int(source["wire_width"]) != geometry.wire_width:
        raise ValueError("snapshot wire_width is inconsistent with GDN geometry")
    if tuple(map(int, source["gdn_layers"])) != geometry.gdn_layers:
        raise ValueError("snapshot GDN indices disagree with layer_types")
    return geometry


def _representative_layers(
    geometry: Qwen35GDNFoldabilityGeometry,
) -> tuple[int, ...]:
    """Match the GDN layer immediately before each full-attention sentinel."""

    full = tuple(
        index
        for index, layer_type in enumerate(geometry.layer_types)
        if layer_type == "full_attention"
    )
    if not full:
        raise ValueError("model has no full-attention sentinels")
    sentinel = tuple(dict.fromkeys((full[0], full[(len(full) - 1) // 2], full[-1])))
    selected = tuple(layer - 1 for layer in sentinel if layer - 1 in geometry.gdn_layers)
    if len(selected) != len(sentinel):
        raise ValueError("a representative full-attention layer has no preceding GDN layer")
    return selected


def _parse_layers(
    raw: str,
    *,
    geometry: Qwen35GDNFoldabilityGeometry,
    snapshot_layers: set[int],
) -> tuple[int, ...]:
    normalized = raw.strip().lower()
    if normalized == "representative":
        selected = _representative_layers(geometry)
    elif normalized == "all":
        selected = tuple(sorted(snapshot_layers))
    else:
        selected = tuple(sorted(set(_csv_ints(raw))))
    non_gdn = sorted(set(selected) - set(geometry.gdn_layers))
    missing = sorted(set(selected) - snapshot_layers)
    if not selected or non_gdn or missing:
        raise ValueError(
            f"invalid GDN layer selection: non-GDN={non_gdn}, missing snapshots={missing}"
        )
    return selected


def _validate_manifest(
    manifest: Mapping[str, Any], *, manifest_path: Path
) -> tuple[Qwen35GDNFoldabilityGeometry, dict[int, Mapping[str, Any]], dict[str, Any]]:
    if (
        manifest.get("format") != SNAPSHOT_FORMAT
        or int(manifest.get("schema_version", -1)) != 1
    ):
        raise ValueError("unsupported GDN foldability snapshot manifest")
    geometry = _geometry_from_manifest(manifest)
    layer_records = _records_by_layer(manifest["layers"])
    selected_layers = tuple(map(int, manifest["selected_layers"]))
    if set(selected_layers) != set(layer_records) or len(selected_layers) != len(
        layer_records
    ):
        raise ValueError("selected_layers and GDN layer records differ")
    if not set(layer_records).issubset(geometry.gdn_layers):
        raise ValueError("snapshot manifest contains a non-GDN layer")
    collection = manifest["collection"]
    if (
        collection.get("model_dtype") != "bfloat16"
        or collection.get("storage_dtype") != "bfloat16"
    ):
        raise ValueError("GDN oracle requires BF16 collection and snapshot storage")
    if set(collection.get("signals", ())) != {"raw_core", "gate_preactivation"}:
        raise ValueError("GDN snapshot signals differ from raw_core/gate_preactivation")
    if collection.get("gate_activation") != "silu_float32":
        raise ValueError("GDN snapshot gate is not the checkpoint FP32 SiLU path")
    if (
        int(collection["sequence_length"]) <= 0
        or int(collection["tokens_per_sequence"]) <= 0
    ):
        raise ValueError("GDN snapshot sequence/token geometry is invalid")
    document = _validate_document_provenance(manifest)
    provenance = {
        "snapshot_manifest": str(manifest_path),
        "snapshot_manifest_sha256": _file_sha256(manifest_path),
        "collection_seed": int(collection["seed"]),
        "dataset_row_order_seed": int(manifest["dataset"]["row_order_seed"]),
        "window_crop_seed": int(manifest["dataset"]["window_crop_seed"]),
        "sequence_length": int(collection["sequence_length"]),
        "tokens_per_sequence": int(collection["tokens_per_sequence"]),
        "split_samples": {
            key: int(value) for key, value in collection["split_samples"].items()
        },
        "storage_dtype": str(collection["storage_dtype"]),
        "gate_activation": str(collection["gate_activation"]),
        **document,
    }
    return geometry, layer_records, provenance


def _validate_model(
    manifest: Mapping[str, Any],
    model_path: Path,
    snapshot_geometry: Qwen35GDNFoldabilityGeometry,
) -> tuple[dict[str, Any], dict[str, str]]:
    manifest_model = Path(str(manifest["model"])).expanduser().resolve()
    if model_path.resolve() != manifest_model:
        raise ValueError("model path differs from the GDN snapshot checkpoint")
    config_path = model_path / "config.json"
    index_path = model_path / "model.safetensors.index.json"
    if not config_path.is_file() or not index_path.is_file():
        raise FileNotFoundError("model config or safetensors index is missing")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("model_type") != "qwen3_5":
        raise ValueError("checkpoint is not a Qwen3.5 model")
    config_geometry = Qwen35GDNFoldabilityGeometry.from_config(config)
    if config_geometry != snapshot_geometry:
        raise ValueError("model config geometry differs from GDN snapshot geometry")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = dict(index["weight_map"])
    if not weight_map or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in weight_map.items()
    ):
        raise ValueError("model safetensors weight_map is malformed")
    model_id, revision = _model_id_and_revision(model_path, config)
    if "qwen3.5-9b" not in model_id.lower():
        raise ValueError(f"checkpoint is not identified as Qwen3.5-9B: {model_id!r}")
    return {
        "model_id": model_id,
        "model_revision": revision,
        "model_path": str(model_path),
        "resolved_model_path": str(model_path.resolve()),
        "config_sha256": _file_sha256(config_path),
        "safetensors_index_sha256": _file_sha256(index_path),
        "checkpoint_transformers_version": config.get("transformers_version"),
    }, weight_map


def _o_proj_specification(
    weight_map: Mapping[str, str], layer_index: int
) -> tuple[str, str]:
    suffix = f".layers.{layer_index}.linear_attn.out_proj.weight"
    matches = [(name, shard) for name, shard in weight_map.items() if name.endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(
            f"expected one GDN out_proj weight for layer {layer_index}, found {len(matches)}"
        )
    bias_suffix = f".layers.{layer_index}.linear_attn.out_proj.bias"
    if any(name.endswith(bias_suffix) for name in weight_map):
        raise ValueError("effective-sparsity oracle requires bias-free GDN out_proj")
    tensor_name, shard_name = matches[0]
    component = Path(shard_name)
    if component.is_absolute() or ".." in component.parts:
        raise ValueError("unsafe model shard path in weight_map")
    return tensor_name, shard_name


def _load_o_proj(
    model_path: Path,
    weight_map: Mapping[str, str],
    layer_index: int,
    geometry: Qwen35GDNFoldabilityGeometry,
) -> tuple[torch.Tensor, dict[str, Any]]:
    tensor_name, shard_name = _o_proj_specification(weight_map, layer_index)
    shard_path = model_path / shard_name
    if not shard_path.is_file():
        raise FileNotFoundError(shard_path)
    shard_sha256 = _file_sha256(shard_path)
    resolved_name = shard_path.resolve().name
    if SHA256_PATTERN.fullmatch(resolved_name) is not None and resolved_name != shard_sha256:
        raise ValueError(f"Hugging Face blob hash mismatch: {shard_path}")
    with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
        if tensor_name not in handle.keys():
            raise KeyError(f"{tensor_name} is absent from {shard_path}")
        weight = handle.get_tensor(tensor_name)
    expected = (geometry.hidden_size, geometry.wire_width)
    if tuple(weight.shape) != expected or weight.dtype != torch.bfloat16:
        raise ValueError(
            f"layer {layer_index} GDN out_proj must be BF16 with shape {expected}, "
            f"got {weight.dtype} {tuple(weight.shape)}"
        )
    if not bool(torch.isfinite(weight).all()):
        raise ValueError(f"layer {layer_index} GDN out_proj contains non-finite values")
    return weight.contiguous(), {
        "tensor_name": tensor_name,
        "shard": shard_name,
        "shard_sha256": shard_sha256,
    }


def _validate_snapshot_files(
    snapshot_dir: Path,
    manifest: Mapping[str, Any],
    layer_records: Mapping[int, Mapping[str, Any]],
    layers: Sequence[int],
    *,
    num_tokens: int,
    geometry: Qwen35GDNFoldabilityGeometry,
) -> dict[int, dict[str, Any]]:
    tolerances = manifest["collection"]["consistency_tolerances"]
    diagnostic_tolerances = {
        "postgate_reconstruction_relative_mse": float(
            tolerances["postgate_reconstruction_relative_mse"]
        ),
        "norm_to_out_proj_wire_relative_mse": float(
            tolerances["norm_to_out_proj_wire_relative_mse"]
        ),
        "direct_out_proj_relative_mse": float(
            tolerances["direct_out_proj_relative_mse"]
        ),
    }
    validated: dict[int, dict[str, Any]] = {}
    for layer_index in layers:
        layer = layer_records[layer_index]
        if set(layer["splits"]) != set(SPLITS):
            raise ValueError(f"layer {layer_index} does not contain exactly {SPLITS}")
        diagnostics = layer["diagnostics"]["validation"]
        for name, tolerance in diagnostic_tolerances.items():
            value = float(diagnostics[name])
            if not math.isfinite(value) or value < 0.0 or value > tolerance:
                raise ValueError(
                    f"layer {layer_index} invalid GDN capture diagnostic {name}={value!r}"
                )
        if int(diagnostics["calls"]) <= 0:
            raise ValueError(f"layer {layer_index} GDN capture has no validation calls")
        record = layer["splits"]["validation"]
        if int(record["rows"]) != num_tokens:
            raise ValueError(
                f"layer {layer_index} validation rows={record['rows']} "
                f"but this screen requires {num_tokens}"
            )
        component = Path(str(record["file"]))
        if component.is_absolute() or ".." in component.parts:
            raise ValueError("unsafe GDN snapshot filename")
        source = snapshot_dir / component
        if not source.is_file():
            raise FileNotFoundError(source)
        expected_hash = _require_sha256(
            record["file_sha256"], name=f"layer {layer_index} validation hash"
        )
        observed_hash = _file_sha256(source)
        if observed_hash != expected_hash:
            raise ValueError(f"GDN snapshot hash mismatch: {source}")
        tensors = load_file(source, device="cpu")
        if set(tensors) != {"raw_core", "gate_preactivation", "norm_weight"}:
            raise ValueError(f"unexpected tensors in {source}: {sorted(tensors)}")
        raw_core = tensors["raw_core"]
        gate = tensors["gate_preactivation"]
        norm_weight = tensors["norm_weight"]
        validate_snapshot_matrices(raw_core, gate, geometry)
        if raw_core.dtype != torch.bfloat16 or gate.dtype != torch.bfloat16:
            raise TypeError(f"{source} activation tensors are not BF16")
        if norm_weight.dtype != torch.float32 or tuple(norm_weight.shape) != (
            geometry.value_head_dim,
        ):
            raise TypeError(f"{source} norm_weight is not FP32 head_dim")
        if int(raw_core.shape[0]) != num_tokens:
            raise ValueError(f"{source} tensor row count differs from manifest")
        validated[layer_index] = {
            "path": str(source),
            "file": str(component),
            "file_sha256": observed_hash,
            "rows": num_tokens,
            "capture_diagnostics": diagnostics,
        }
    return validated


def _index_runtime(runtime: RuntimeActivations, indices: torch.Tensor) -> RuntimeActivations:
    return RuntimeActivations(
        h_pre_gate=runtime.h_pre_gate.index_select(0, indices),
        gate_logits=runtime.gate_logits.index_select(0, indices),
        a_post_sigmoid=runtime.a_post_sigmoid.index_select(0, indices),
        c_post_gate=runtime.c_post_gate.index_select(0, indices),
    )


def _markdown(rows: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any]) -> str:
    lines = [
        "# Qwen3.5 GDN Post-State SiLU Effective Sparsity Oracle",
        "",
        (
            "Training-free, validation-only analysis of the dense GDN output wire. "
            "The recurrent state and state update remain exact and unmodified."
        ),
        "",
        (
            "Each activation follows the checkpoint order: per-head RMSNorm, FP32 "
            "SiLU gate multiplication, BF16 wire, then FP32 diagnostic `out_proj`."
        ),
        "",
        (
            "The decisive metric is held-out `linear_attn.out_proj` output relative "
            "MSE. This diagnoses effective, not literal, sparsity."
        ),
        "",
        f"Main scope: **{MAIN_SCOPE}** ({manifest['num_tokens']} validation tokens).",
        "",
        (
            f"Audit scope: **{AUDIT_SCOPE}** ({manifest['audit']['num_tokens']} "
            "document-balanced tokens); greedy rows are privileged audit references."
        ),
        "",
        (
            "A sparse candidate wire alone does **not** shorten the row-parallel "
            "`out_proj` AllReduce."
        ),
        "",
    ]
    for layer in manifest["layers"]:
        main_rows = [
            row
            for row in rows
            if row["layer"] == layer and row["scope"] == MAIN_SCOPE
        ]
        gate_stats = main_rows[0]["gate_statistics"] if main_rows else {}
        lines.extend(
            [
                f"## GDN layer {layer}",
                "",
                (
                    f"Mean `|SiLU(g)|`: {float(gate_stats.get('mean_a', float('nan'))):.6g}; "
                    f"median: {float(gate_stats.get('gate_abs_quantiles', {}).get('q50', float('nan'))):.6g}."
                ),
                "",
                "| Method | Seed | Keep ratio | Relative MSE | P95 token error | Retained input energy |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in main_rows:
            metrics = row["metrics"]
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row["method"]),
                        "-" if row["seed"] is None else str(row["seed"]),
                        f"{row['realized_ratio']:.6f}",
                        f"{metrics['relative_mse']:.6g}",
                        f"{metrics['per_token_relative_squared_error']['p95']:.6g}",
                        f"{metrics['retained_input_energy']:.6g}",
                    ]
                )
                + " |"
            )
        audit_rows = [
            row
            for row in rows
            if row["layer"] == layer and row["scope"] == AUDIT_SCOPE
        ]
        lines.extend(
            [
                "",
                "### Shared document-balanced audit",
                "",
                "| Method | Seed | Keep ratio | Relative MSE | P95 token error |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for row in audit_rows:
            metrics = row["metrics"]
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row["method"]),
                        "-" if row["seed"] is None else str(row["seed"]),
                        f"{row['realized_ratio']:.6f}",
                        f"{metrics['relative_mse']:.6g}",
                        f"{metrics['per_token_relative_squared_error']['p95']:.6g}",
                    ]
                )
                + " |"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.torch_num_threads)
    if args.num_tokens != 8192:
        raise ValueError("this GDN validation main screen requires --num-tokens 8192")
    if args.chunk_size <= 0 or args.audit_num_tokens <= 0 or args.torch_num_threads <= 0:
        raise ValueError("chunk, audit-token, and thread counts must be positive")
    methods = _methods(args.methods)
    ratios = _ratios(args.keep_ratios)
    greedy_ratios = _ratios(args.greedy_keep_ratios)
    random_seeds = _csv_ints(args.random_seeds)
    gate_thresholds = _thresholds(args.gate_thresholds)
    activation_thresholds = _thresholds(args.activation_thresholds)
    if "random" in methods and len(random_seeds) < 3:
        raise ValueError("random baseline requires at least three seeds")

    snapshot_dir = Path(args.snapshots).expanduser().resolve()
    manifest_path = snapshot_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    snapshot_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    geometry, layer_records, snapshot_provenance = _validate_manifest(
        snapshot_manifest, manifest_path=manifest_path
    )
    model_path = Path(args.model_path or snapshot_manifest["model"]).expanduser().resolve()
    model_provenance, weight_map = _validate_model(
        snapshot_manifest, model_path, geometry
    )
    layers = _parse_layers(
        args.layers, geometry=geometry, snapshot_layers=set(layer_records)
    )
    expected_representative = _representative_layers(geometry)
    if args.layers.strip().lower() == "representative" and layers != expected_representative:
        raise RuntimeError("GDN representative discovery did not select layers 2/14/30")
    validated_snapshots = _validate_snapshot_files(
        snapshot_dir,
        snapshot_manifest,
        layer_records,
        layers,
        num_tokens=args.num_tokens,
        geometry=geometry,
    )
    audit_indices = _document_balanced_audit_indices(
        snapshot_manifest,
        num_rows=args.num_tokens,
        audit_num_tokens=args.audit_num_tokens,
        seed=args.audit_seed,
    )
    audit_indices_sha256 = hashlib.sha256(audit_indices.numpy().tobytes()).hexdigest()
    validation_document_count = len(
        snapshot_manifest["collection"]["split_metadata"]["records"]["validation"]
    )
    audit_document_counts = torch.bincount(
        audit_indices // int(snapshot_manifest["collection"]["tokens_per_sequence"]),
        minlength=validation_document_count,
    )

    output_dir = Path(args.output_dir).expanduser().resolve()
    partial_dir = output_dir.with_name(output_dir.name + ".partial")
    if output_dir.exists() or partial_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite output or incomplete staging directory: {output_dir}"
        )
    partial_dir.mkdir(parents=True, exist_ok=False)

    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        torch.cuda.set_device(device.index or 0)
        torch.empty((), device=device)
    git_commit = _git_commit()
    context = {
        **model_provenance,
        "dataset": snapshot_manifest["dataset"],
        "validation_document_hash": snapshot_provenance["validation_document_hash"],
        "audit_indices_sha256": audit_indices_sha256,
        "git_commit": git_commit,
        "row_format": ROW_FORMAT,
    }
    rows: list[dict[str, Any]] = []
    frequencies: dict[str, torch.Tensor] = {}
    layer_runtime: dict[str, Any] = {}
    weight_sources: dict[str, Any] = {}
    greedy_inputs: dict[int, dict[str, Any]] = {}
    greedy_failures: list[dict[str, Any]] = []
    greedy_completed_layers: list[int] = []
    timestamp_started = datetime.now(timezone.utc).isoformat()
    started_all = time.perf_counter()

    with torch.inference_mode():
        for layer_index in layers:
            print(f"[GDNSiLUSparsity] loading layer={layer_index}", flush=True)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            layer_started = time.perf_counter()
            tensors = load_file(validated_snapshots[layer_index]["path"], device="cpu")
            raw_core = tensors["raw_core"].to(device=device)
            gate_preactivation = tensors["gate_preactivation"].to(device=device)
            norm_weight = tensors["norm_weight"].to(device=device)
            runtime = reconstruct_gdn_silu_runtime(
                raw_core,
                gate_preactivation,
                norm_weight,
                num_value_heads=geometry.num_value_heads,
                value_head_dim=geometry.value_head_dim,
                rms_norm_eps=geometry.rms_norm_eps,
            )
            weight_cpu, weight_source = _load_o_proj(
                model_path, weight_map, layer_index, geometry
            )
            weight = weight_cpu.to(device=device, dtype=torch.float32)
            teacher = runtime.c_post_gate.float() @ weight.transpose(0, 1)
            if not bool(torch.isfinite(teacher).all()):
                raise FloatingPointError(
                    f"GDN layer {layer_index} teacher contains non-finite values"
                )
            main_gate_statistics = _gate_statistics(
                runtime,
                gate_thresholds=gate_thresholds,
                activation_thresholds=activation_thresholds,
            )
            audit_device_indices = audit_indices.to(device)
            audit_runtime = _index_runtime(runtime, audit_device_indices)
            audit_gate_statistics = _gate_statistics(
                audit_runtime,
                gate_thresholds=gate_thresholds,
                activation_thresholds=activation_thresholds,
            )
            for method in methods:
                print(
                    f"[GDNSiLUSparsity] layer={layer_index} method={method}",
                    flush=True,
                )
                if method == "random":
                    for seed in random_seeds:
                        rows.extend(
                            _evaluate_ranked_method(
                                runtime=runtime,
                                weight=weight,
                                teacher=teacher,
                                audit_indices=audit_indices,
                                method=method,
                                seed=seed,
                                ratios=ratios,
                                chunk_size=args.chunk_size,
                                layer_index=layer_index,
                                context=context,
                                main_gate_statistics=main_gate_statistics,
                                audit_gate_statistics=audit_gate_statistics,
                                frequencies=frequencies,
                            )
                        )
                elif method == "random_expected":
                    rows.extend(
                        _evaluate_analytic_random(
                            runtime=runtime,
                            weight=weight,
                            audit_indices=audit_indices,
                            ratios=ratios,
                            layer_index=layer_index,
                            context=context,
                            main_gate_statistics=main_gate_statistics,
                            audit_gate_statistics=audit_gate_statistics,
                            frequencies=frequencies,
                        )
                    )
                else:
                    rows.extend(
                        _evaluate_ranked_method(
                            runtime=runtime,
                            weight=weight,
                            teacher=teacher,
                            audit_indices=audit_indices,
                            method=method,
                            seed=None,
                            ratios=ratios,
                            chunk_size=args.chunk_size,
                            layer_index=layer_index,
                            context=context,
                            main_gate_statistics=main_gate_statistics,
                            audit_gate_statistics=audit_gate_statistics,
                            frequencies=frequencies,
                        )
                    )
            if not args.skip_greedy:
                greedy_inputs[layer_index] = {
                    "c": audit_runtime.c_post_gate.detach().cpu().contiguous(),
                    "a": audit_runtime.a_post_sigmoid.detach().cpu().contiguous(),
                    "weight": weight_cpu.contiguous(),
                    "gate_statistics": audit_gate_statistics,
                }
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            layer_runtime[str(layer_index)] = {
                "main_seconds": time.perf_counter() - layer_started,
                "main_peak_cuda_allocated_bytes": (
                    int(torch.cuda.max_memory_allocated(device))
                    if device.type == "cuda"
                    else 0
                ),
            }
            weight_sources[str(layer_index)] = weight_source
            _write_text(
                partial_dir / "nongreedy_results.jsonl",
                "".join(
                    _json_text(row) + "\n"
                    for row in rows
                    if row["method"] != "greedy_residual_reference"
                ),
            )
            del (
                tensors,
                raw_core,
                gate_preactivation,
                norm_weight,
                runtime,
                audit_runtime,
                weight_cpu,
                weight,
                teacher,
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()

        if not args.skip_greedy:
            for layer_index in layers:
                payload = greedy_inputs[layer_index]
                print(
                    f"[GDNSiLUSparsity] layer={layer_index} method=greedy_residual_reference",
                    flush=True,
                )
                greedy_started = time.perf_counter()
                greedy_c = greedy_a = greedy_weight = None
                greedy_frequency_rows: dict[str, torch.Tensor] = {}
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                try:
                    greedy_c = payload["c"].to(device=device)
                    greedy_a = payload["a"].to(device=device)
                    greedy_weight = payload["weight"].to(
                        device=device, dtype=torch.float32
                    )
                    greedy_rows = _evaluate_greedy_audit(
                        c=greedy_c,
                        a=greedy_a,
                        weight=greedy_weight,
                        ratios=greedy_ratios,
                        layer_index=layer_index,
                        context=context,
                        audit_gate_statistics=payload["gate_statistics"],
                        frequencies=greedy_frequency_rows,
                    )
                    rows.extend(greedy_rows)
                    frequencies.update(greedy_frequency_rows)
                    greedy_completed_layers.append(layer_index)
                except Exception as error:
                    failure = {
                        "layer": layer_index,
                        "exception_type": type(error).__name__,
                        "message": str(error),
                    }
                    greedy_failures.append(failure)
                    print(f"[GDNSiLUSparsity][GreedyWarning] {_json_text(failure)}", flush=True)
                    if args.strict_greedy:
                        raise
                finally:
                    layer_runtime[str(layer_index)]["greedy_seconds"] = (
                        time.perf_counter() - greedy_started
                    )
                    layer_runtime[str(layer_index)][
                        "greedy_peak_cuda_allocated_bytes"
                    ] = (
                        int(torch.cuda.max_memory_allocated(device))
                        if device.type == "cuda"
                        else 0
                    )
                    del greedy_c, greedy_a, greedy_weight
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

    nongreedy_path = partial_dir / "nongreedy_results.jsonl"
    results_path = partial_dir / "results.jsonl"
    _write_text(results_path, "".join(_json_text(row) + "\n" for row in rows))
    summary_csv_path = partial_dir / "summary.csv"
    _write_text(summary_csv_path, _summary_csv(rows))
    frequency_path = partial_dir / "channel_selection_frequency.safetensors"
    save_file(
        frequencies,
        frequency_path,
        metadata={
            "format": FREQUENCY_FORMAT,
            "schema_version": "1",
            "semantics": "per-token GDN selection frequency; random_expected stores expectation",
        },
    )
    plot_path = partial_dir / "sparsity_curve.svg"
    _plot(
        rows,
        layers,
        plot_path,
        scope=MAIN_SCOPE,
        title="Qwen3.5 GDN post-state SiLU effective sparsity (8192-token main scope)",
        layer_title_prefix="GDN layer",
    )
    audit_plot_path = partial_dir / "audit_sparsity_curve.svg"
    _plot(
        rows,
        layers,
        audit_plot_path,
        scope=AUDIT_SCOPE,
        title="Qwen3.5 GDN post-state SiLU effective sparsity (shared 128-token audit)",
        layer_title_prefix="GDN layer",
    )

    result_manifest: dict[str, Any] = {
        "format": FORMAT,
        "schema_version": 1,
        "timestamp_started": timestamp_started,
        "timestamp_completed": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "command": " ".join(shlex.quote(piece) for piece in sys.argv),
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
        "software": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": _installed_version("transformers"),
            "safetensors": _installed_version("safetensors"),
        },
        "model": model_provenance,
        "snapshots": snapshot_provenance,
        "validated_snapshot_files": validated_snapshots,
        "weight_sources": weight_sources,
        "dataset": snapshot_manifest["dataset"],
        "geometry": {
            "hidden_size": geometry.hidden_size,
            "wire_width": geometry.wire_width,
            "num_value_heads": geometry.num_value_heads,
            "value_head_dim": geometry.value_head_dim,
            "rms_norm_eps": geometry.rms_norm_eps,
            "gdn_layers": list(geometry.gdn_layers),
        },
        "layers": list(layers),
        "representative_layers_discovered": list(expected_representative),
        "split": "validation",
        "num_tokens": args.num_tokens,
        "methods": list(methods),
        "random_seeds": list(random_seeds),
        "keep_ratios": list(ratios),
        "greedy_keep_ratios": list(greedy_ratios),
        "greedy": {
            "requested": not args.skip_greedy,
            "strict_failure_policy": bool(args.strict_greedy),
            "runs_after_all_nongreedy_layers": True,
            "completed_layers": greedy_completed_layers,
            "failures": greedy_failures,
        },
        "audit": {
            "scope": AUDIT_SCOPE,
            "num_tokens": args.audit_num_tokens,
            "seed": args.audit_seed,
            "document_balanced": True,
            "indices": audit_indices.tolist(),
            "indices_sha256": audit_indices_sha256,
            "per_document_token_counts": audit_document_counts.tolist(),
            "minimum_tokens_per_document": int(audit_document_counts.min()),
            "maximum_tokens_per_document": int(audit_document_counts.max()),
            "shared_by_every_method": True,
        },
        "numerics": {
            "core_rmsnorm_input_dtype": "bfloat16",
            "rms_reduction_dtype": "float32",
            "normalized_core_rounding_dtype": "bfloat16",
            "gate_activation": "silu_float32",
            "post_gate_wire_dtype": "bfloat16",
            "post_gate_and_o_proj_work_dtype": "float32",
            "metric_reduction_dtype": "float64",
            "stable_score_ties": True,
        },
        "scope_guards": {
            "training_free": True,
            "fitting": False,
            "backward": False,
            "optimizer": None,
            "checkpoint_modified": False,
            "recurrent_state_modified": False,
            "state_update_modified": False,
            "sparsified_interface": "post_state_rmsnorm_silu_wire_before_linear_attn_out_proj",
            "greedy_label": "privileged_target_aware_fixed_k_reference_not_global_oracle",
            "audit_never_compared_as_full_validation": True,
            "does_not_reduce_row_parallel_out_proj_allreduce_by_itself": True,
        },
        "gate_thresholds": list(gate_thresholds),
        "activation_thresholds": list(activation_thresholds),
        "row_count": len(rows),
        "random_monte_carlo_validation": _random_expectation_validation(rows),
        "runtime_seconds": time.perf_counter() - started_all,
        "layer_runtime": layer_runtime,
        "artifacts": {},
    }
    summary_md_path = partial_dir / "summary.md"
    _write_text(summary_md_path, _markdown(rows, result_manifest))
    for artifact_path in (
        results_path,
        nongreedy_path,
        summary_csv_path,
        summary_md_path,
        plot_path,
        audit_plot_path,
        frequency_path,
    ):
        result_manifest["artifacts"][artifact_path.name] = {
            "size_bytes": artifact_path.stat().st_size,
            "sha256": _file_sha256(artifact_path),
        }
    _write_text(
        partial_dir / "manifest.json", _json_text(result_manifest, indent=2) + "\n"
    )
    os.replace(partial_dir, output_dir)
    print(f"[Saved] {output_dir}", flush=True)


if __name__ == "__main__":
    main()
