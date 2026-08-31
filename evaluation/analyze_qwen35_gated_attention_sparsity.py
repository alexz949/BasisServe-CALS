#!/usr/bin/env python3
"""Measure Qwen3.5 gated-attention token-dependent effective sparsity offline."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
from html import escape
from importlib.metadata import PackageNotFoundError, version as package_version
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basisserve.calibration.mlp_snapshots import SPLITS, records_sha256
from basisserve.core.qwen35_gated_wo_foldability import (
    Qwen35FullAttentionGeometry,
    SNAPSHOT_FORMAT,
    validate_snapshot_matrices,
)
from basisserve.diagnostics.qwen35_gated_attention_sparsity import (
    SCORE_METHODS,
    RuntimeActivations,
    analytic_random_expected_metrics,
    exact_linear_quantiles,
    greedy_residual_reference,
    output_metrics,
    random_expectation_statistics,
    random_rankings,
    reconstruct_bf16_runtime,
    retained_energy_metrics,
    retained_k,
    score_channels,
    selected_indices,
    selected_output,
    selection_frequency,
    selection_stability,
    stable_descending_ranking,
)


FORMAT = "basisserve.qwen35.gated_attention_effective_sparsity.v1"
ROW_FORMAT = "basisserve.qwen35.gated_attention_effective_sparsity.row.v1"
FREQUENCY_FORMAT = "basisserve.qwen35.gated_attention_selection_frequency.v1"
MAIN_SCOPE = "validation_main_8192"
AUDIT_SCOPE = "validation_document_balanced_audit"
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
    parser.add_argument(
        "--methods",
        default=",".join(DEFAULT_METHODS),
        help="Comma-separated score methods plus random and random_expected.",
    )
    parser.add_argument(
        "--keep-ratios",
        default=",".join(map(str, DEFAULT_KEEP_RATIOS)),
    )
    parser.add_argument("--random-seeds", default="0,1,2")
    parser.add_argument("--audit-num-tokens", type=int, default=128)
    parser.add_argument("--audit-seed", type=int, default=20260808)
    parser.add_argument(
        "--greedy-keep-ratios",
        default=",".join(map(str, DEFAULT_GREEDY_RATIOS)),
    )
    parser.add_argument(
        "--skip-greedy",
        action="store_true",
        help="Publish the main and matched-audit score sweeps without greedy.",
    )
    parser.add_argument(
        "--strict-greedy",
        action="store_true",
        help="Fail the whole command instead of recording a greedy audit failure.",
    )
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument(
        "--gate-thresholds",
        default="1e-4,1e-3,1e-2,5e-2,1e-1",
    )
    parser.add_argument(
        "--activation-thresholds",
        default="1e-4,1e-3,1e-2,5e-2,1e-1",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--torch-num-threads", type=int, default=4)
    return parser.parse_args()


def _csv_ints(raw: str) -> tuple[int, ...]:
    values = tuple(dict.fromkeys(int(piece.strip()) for piece in raw.split(",") if piece.strip()))
    if not values:
        raise ValueError("integer list is empty")
    return values


def _ratios(raw: str) -> tuple[float, ...]:
    values = tuple(sorted(set(float(piece.strip()) for piece in raw.split(",") if piece.strip())))
    if not values or any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
        raise ValueError("retained ratios must be finite values in [0, 1]")
    return values


def _thresholds(raw: str) -> tuple[float, ...]:
    values = tuple(sorted(set(float(piece.strip()) for piece in raw.split(",") if piece.strip())))
    if not values or any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("effective-zero thresholds must be finite and nonnegative")
    return values


def _methods(raw: str) -> tuple[str, ...]:
    values = tuple(dict.fromkeys(piece.strip() for piece in raw.split(",") if piece.strip()))
    allowed = set(DEFAULT_METHODS)
    unknown = sorted(set(values) - allowed)
    if not values or unknown:
        raise ValueError(f"unsupported sparsity methods: {unknown}")
    return values


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: Any, *, name: str) -> str:
    text = str(value)
    if SHA256_PATTERN.fullmatch(text) is None:
        raise ValueError(f"{name} is not a lowercase SHA-256 digest")
    return text


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _installed_version(distribution: str) -> str | None:
    try:
        return package_version(distribution)
    except PackageNotFoundError:
        return None


def _json_text(payload: Any, *, indent: int | None = None) -> str:
    return json.dumps(
        _jsonable(payload),
        allow_nan=False,
        indent=indent,
        sort_keys=indent is not None,
        separators=None if indent is not None else (",", ":"),
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return _jsonable(value.item())
        return [_jsonable(item) for item in value.detach().cpu().tolist()]
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"refusing to serialize non-finite float {value}")
    return value


def _write_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _geometry_from_manifest(manifest: Mapping[str, Any]) -> Qwen35FullAttentionGeometry:
    source = manifest["geometry"]
    geometry = Qwen35FullAttentionGeometry(
        hidden_size=int(source["hidden_size"]),
        num_query_heads=int(source["num_query_heads"]),
        num_kv_heads=int(source["num_kv_heads"]),
        head_dim=int(source["head_dim"]),
        layer_types=tuple(map(str, source["layer_types"])),
    )
    geometry.validate()
    if int(source["wire_width"]) != geometry.wire_width:
        raise ValueError("snapshot wire_width is inconsistent with head geometry")
    if int(source["query_heads_per_kv_group"]) != geometry.query_heads_per_kv_group:
        raise ValueError("snapshot query-head/KV-group ratio is inconsistent")
    if tuple(map(int, source["full_attention_layers"])) != geometry.full_attention_layers:
        raise ValueError("snapshot full-attention indices disagree with layer_types")
    return geometry


def _representative_layers(full_attention_layers: Sequence[int]) -> tuple[int, ...]:
    layers = tuple(map(int, full_attention_layers))
    if not layers:
        raise ValueError("model config contains no full-attention layers")
    if tuple(sorted(set(layers))) != layers:
        raise ValueError("full-attention layers must be unique and increasing")
    return tuple(dict.fromkeys((layers[0], layers[(len(layers) - 1) // 2], layers[-1])))


def _parse_layers(
    raw: str,
    *,
    full_attention_layers: Sequence[int],
    snapshot_layers: set[int],
) -> tuple[int, ...]:
    full = set(map(int, full_attention_layers))
    normalized = raw.strip().lower()
    if normalized == "representative":
        selected = _representative_layers(full_attention_layers)
    elif normalized == "all":
        selected = tuple(sorted(snapshot_layers))
    else:
        selected = tuple(sorted(set(_csv_ints(raw))))
    non_full = sorted(set(selected) - full)
    missing = sorted(set(selected) - snapshot_layers)
    if not selected or non_full or missing:
        raise ValueError(
            f"invalid layer selection: non-full-attention={non_full}, missing snapshots={missing}"
        )
    return selected


def _records_by_layer(records: Sequence[Mapping[str, Any]]) -> dict[int, Mapping[str, Any]]:
    result = {int(record["layer_index"]): record for record in records}
    if len(result) != len(records):
        raise ValueError("snapshot manifest contains duplicate layer records")
    return result


def _validate_document_provenance(manifest: Mapping[str, Any]) -> dict[str, Any]:
    collection = manifest["collection"]
    split_metadata = collection["split_metadata"]
    if not bool(collection.get("document_split_isolation")) or not bool(
        split_metadata.get("document_split_isolation")
    ):
        raise ValueError("snapshot collection does not guarantee document-isolated splits")
    records = split_metadata["records"]
    if set(records) != set(SPLITS):
        raise ValueError(f"document records must contain exactly {SPLITS}")
    split_samples = collection["split_samples"]
    if set(split_samples) != set(SPLITS):
        raise ValueError(f"split_samples must contain exactly {SPLITS}")
    if int(split_metadata["sequence_length"]) != int(collection["sequence_length"]):
        raise ValueError("document and collection sequence lengths differ")
    if int(split_metadata["tokens_per_sequence"]) != int(
        collection["tokens_per_sequence"]
    ):
        raise ValueError("document and collection retained-token counts differ")
    line_sets: dict[str, set[int]] = {}
    for split in SPLITS:
        split_records = records[split]
        if len(split_records) != int(split_samples[split]):
            raise ValueError(f"{split} document count differs from split_samples")
        expected = _require_sha256(
            split_metadata["record_sha256"][split],
            name=f"{split} document-record hash",
        )
        if records_sha256(split_records) != expected:
            raise ValueError(f"{split} document-record hash mismatch")
        line_sets[split] = {int(record["line_index"]) for record in split_records}
        if len(line_sets[split]) != len(split_records):
            raise ValueError(f"{split} repeats a document")
    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1 :]:
            overlap = line_sets[left] & line_sets[right]
            if overlap:
                raise ValueError(f"document leakage between {left} and {right}: {sorted(overlap)[:8]}")
    combined = records_sha256(
        [{"split": split, **dict(record)} for split in SPLITS for record in records[split]]
    )
    if combined != _require_sha256(
        split_metadata["combined_record_sha256"],
        name="combined document-record hash",
    ):
        raise ValueError("combined document-record hash mismatch")
    for family in ("sampled_position_sha256", "sampled_position_records_sha256"):
        for split in SPLITS:
            _require_sha256(collection[family][split], name=f"{family}.{split}")
    return {
        "document_split_isolation": True,
        "combined_document_hash": combined,
        "validation_document_hash": split_metadata["record_sha256"]["validation"],
        "validation_document_ids": sorted(line_sets["validation"]),
        "validation_document_count": len(line_sets["validation"]),
        "validation_sampled_position_sha256": collection["sampled_position_sha256"]["validation"],
        "validation_sampled_position_records_sha256": collection[
            "sampled_position_records_sha256"
        ]["validation"],
    }


def _model_id_and_revision(model_path: Path, config: Mapping[str, Any]) -> tuple[str, str | None]:
    resolved = model_path.resolve()
    revision: str | None = None
    model_id = str(model_path)
    parts = resolved.parts
    for index, part in enumerate(parts):
        if part.startswith("models--"):
            model_id = "/".join(part[len("models--") :].split("--"))
        if part == "snapshots" and index + 1 < len(parts):
            candidate = parts[index + 1]
            if re.fullmatch(r"[0-9a-f]{40}", candidate):
                revision = candidate
    config_revision = config.get("_commit_hash")
    if config_revision is not None:
        if revision is not None and str(config_revision) != revision:
            raise ValueError("config revision and resolved snapshot revision differ")
        revision = str(config_revision)
    return model_id, revision


def _validate_model(
    manifest: Mapping[str, Any],
    model_path: Path,
    snapshot_geometry: Qwen35FullAttentionGeometry,
) -> tuple[dict[str, Any], dict[str, str]]:
    manifest_model = Path(str(manifest["model"])).expanduser().resolve()
    if model_path.resolve() != manifest_model:
        raise ValueError("model path differs from the checkpoint used to collect snapshots")
    config_path = model_path / "config.json"
    index_path = model_path / "model.safetensors.index.json"
    if not config_path.is_file() or not index_path.is_file():
        raise FileNotFoundError("model config or safetensors index is missing")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("model_type") != "qwen3_5":
        raise ValueError("checkpoint is not a Qwen3.5 model")
    text_config = config.get("text_config", config)
    if not bool(text_config.get("attn_output_gate")):
        raise ValueError("checkpoint does not enable the full-attention output gate")
    if bool(text_config.get("attention_bias", False)):
        raise ValueError("effective-sparsity reconstruction requires bias-free attention")
    config_geometry = Qwen35FullAttentionGeometry.from_config(config)
    if config_geometry != snapshot_geometry:
        raise ValueError("model config geometry differs from snapshot geometry")
    if int(text_config.get("num_hidden_layers", -1)) != len(config_geometry.layer_types):
        raise ValueError("num_hidden_layers differs from layer_types length")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = dict(index["weight_map"])
    if not weight_map or any(not isinstance(key, str) or not isinstance(value, str) for key, value in weight_map.items()):
        raise ValueError("model safetensors weight_map is malformed")
    model_id, revision = _model_id_and_revision(model_path, config)
    if "qwen3.5-9b" not in model_id.lower():
        raise ValueError(f"checkpoint is not identified as Qwen3.5-9B: {model_id!r}")
    provenance = {
        "model_id": model_id,
        "model_revision": revision,
        "model_path": str(model_path),
        "resolved_model_path": str(model_path.resolve()),
        "config_sha256": _file_sha256(config_path),
        "safetensors_index_sha256": _file_sha256(index_path),
        "checkpoint_transformers_version": config.get("transformers_version"),
    }
    return provenance, weight_map


def _o_proj_specification(weight_map: Mapping[str, str], layer_index: int) -> tuple[str, str]:
    suffix = f".layers.{layer_index}.self_attn.o_proj.weight"
    matches = [(name, shard) for name, shard in weight_map.items() if name.endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f"expected one o_proj weight for layer {layer_index}, found {len(matches)}")
    bias_suffix = f".layers.{layer_index}.self_attn.o_proj.bias"
    if any(name.endswith(bias_suffix) for name in weight_map):
        raise ValueError("effective-sparsity oracle requires bias-free o_proj")
    tensor_name, shard_name = matches[0]
    shard_component = Path(shard_name)
    if shard_component.is_absolute() or ".." in shard_component.parts:
        raise ValueError("unsafe model shard path in weight_map")
    return tensor_name, shard_name


def _load_o_proj(
    model_path: Path,
    weight_map: Mapping[str, str],
    layer_index: int,
    geometry: Qwen35FullAttentionGeometry,
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
            f"layer {layer_index} o_proj must be BF16 with shape {expected}, got {weight.dtype} {tuple(weight.shape)}"
        )
    if not bool(torch.isfinite(weight).all()):
        raise ValueError(f"layer {layer_index} o_proj contains non-finite values")
    return weight.contiguous(), {
        "tensor_name": tensor_name,
        "shard": shard_name,
        "shard_sha256": shard_sha256,
    }


def _validate_manifest(
    manifest: Mapping[str, Any],
    *,
    manifest_path: Path,
) -> tuple[Qwen35FullAttentionGeometry, dict[int, Mapping[str, Any]], dict[str, Any]]:
    if manifest.get("format") != SNAPSHOT_FORMAT or int(manifest.get("schema_version", -1)) != 1:
        raise ValueError("unsupported gated-Wo snapshot manifest")
    geometry = _geometry_from_manifest(manifest)
    layer_records = _records_by_layer(manifest["layers"])
    selected_layers = tuple(map(int, manifest["selected_layers"]))
    if set(selected_layers) != set(layer_records) or len(selected_layers) != len(layer_records):
        raise ValueError("selected_layers and layer records differ")
    if not set(layer_records).issubset(geometry.full_attention_layers):
        raise ValueError("snapshot manifest contains a non-full-attention layer")
    collection = manifest["collection"]
    if collection.get("model_dtype") != "bfloat16" or collection.get("storage_dtype") != "bfloat16":
        raise ValueError("oracle requires BF16 collection and snapshot storage")
    if set(collection.get("signals", ())) != {"h_pre_gate", "gate_logits"}:
        raise ValueError("snapshot signals differ from h_pre_gate/gate_logits")
    if int(collection["sequence_length"]) <= 0 or int(collection["tokens_per_sequence"]) <= 0:
        raise ValueError("snapshot sequence/token geometry is invalid")
    document = _validate_document_provenance(manifest)
    provenance = {
        "snapshot_manifest": str(manifest_path),
        "snapshot_manifest_sha256": _file_sha256(manifest_path),
        "collection_seed": int(collection["seed"]),
        "dataset_row_order_seed": int(manifest["dataset"]["row_order_seed"]),
        "window_crop_seed": int(manifest["dataset"]["window_crop_seed"]),
        "sequence_length": int(collection["sequence_length"]),
        "tokens_per_sequence": int(collection["tokens_per_sequence"]),
        "split_samples": {key: int(value) for key, value in collection["split_samples"].items()},
        "storage_dtype": str(collection["storage_dtype"]),
        **document,
    }
    return geometry, layer_records, provenance


def _validate_snapshot_files(
    snapshot_dir: Path,
    layer_records: Mapping[int, Mapping[str, Any]],
    layers: Sequence[int],
    *,
    num_tokens: int,
    geometry: Qwen35FullAttentionGeometry,
) -> dict[int, dict[str, Any]]:
    validated: dict[int, dict[str, Any]] = {}
    for layer_index in layers:
        layer = layer_records[layer_index]
        if set(layer["splits"]) != set(SPLITS):
            raise ValueError(f"layer {layer_index} does not contain exactly {SPLITS}")
        diagnostics = layer["diagnostics"]["validation"]
        for diagnostic_name, tolerance in (
            ("postgate_reconstruction_relative_mse", 1.0e-12),
            ("postgate_reconstruction_max_abs", 1.0e-6),
            ("direct_o_proj_relative_mse", 1.0e-12),
            ("direct_o_proj_max_abs", 1.0e-6),
        ):
            value = float(diagnostics[diagnostic_name])
            if not math.isfinite(value) or value < 0.0 or value > tolerance:
                raise ValueError(
                    f"layer {layer_index} invalid capture diagnostic "
                    f"{diagnostic_name}={value!r}"
                )
        if int(diagnostics["calls"]) <= 0:
            raise ValueError(f"layer {layer_index} capture has no validation calls")
        record = layer["splits"]["validation"]
        if int(record["rows"]) != num_tokens:
            raise ValueError(
                f"layer {layer_index} validation rows={record['rows']} but this screen requires {num_tokens}"
            )
        component = Path(str(record["file"]))
        if component.is_absolute() or ".." in component.parts:
            raise ValueError("unsafe snapshot filename")
        source = snapshot_dir / component
        if not source.is_file():
            raise FileNotFoundError(source)
        expected_hash = _require_sha256(record["file_sha256"], name=f"layer {layer_index} validation hash")
        observed_hash = _file_sha256(source)
        if observed_hash != expected_hash:
            raise ValueError(f"snapshot hash mismatch: {source}")
        with safe_open(str(source), framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
            if keys != {"h_pre_gate", "gate_logits"}:
                raise ValueError(f"unexpected tensors in {source}: {sorted(keys)}")
            h = handle.get_tensor("h_pre_gate")
            gate = handle.get_tensor("gate_logits")
        validate_snapshot_matrices(h, gate, geometry)
        if h.dtype != torch.bfloat16 or gate.dtype != torch.bfloat16:
            raise TypeError(f"{source} is not BF16")
        if int(h.shape[0]) != num_tokens:
            raise ValueError(f"{source} tensor row count differs from manifest")
        validated[layer_index] = {
            "path": str(source),
            "file": str(component),
            "file_sha256": observed_hash,
            "rows": num_tokens,
        }
    return validated


def _document_balanced_audit_indices(
    manifest: Mapping[str, Any],
    *,
    num_rows: int,
    audit_num_tokens: int,
    seed: int,
) -> torch.Tensor:
    records = manifest["collection"]["split_metadata"]["records"]["validation"]
    tokens_per_document = int(manifest["collection"]["tokens_per_sequence"])
    if len(records) * tokens_per_document != num_rows:
        raise ValueError("validation rows do not form equal contiguous document blocks")
    if not 0 < audit_num_tokens <= num_rows:
        raise ValueError("audit token count lies outside validation rows")
    quotient, remainder = divmod(audit_num_tokens, len(records))
    if quotient > tokens_per_document or (quotient == tokens_per_document and remainder):
        raise ValueError("audit request exceeds per-document retained positions")
    document_order = torch.randperm(
        len(records), generator=torch.Generator().manual_seed(seed)
    ).tolist()
    extras = set(document_order[:remainder])
    selected: list[int] = []
    for document_index in range(len(records)):
        count = quotient + int(document_index in extras)
        if count == 0:
            continue
        document_seed = (
            int(seed)
            ^ (int(records[document_index]["line_index"]) * 0x9E3779B1)
            ^ document_index
        ) & 0x7FFFFFFFFFFFFFFF
        positions = torch.randperm(
            tokens_per_document,
            generator=torch.Generator().manual_seed(document_seed),
        )[:count]
        selected.extend((positions + document_index * tokens_per_document).tolist())
    result = torch.tensor(sorted(selected), dtype=torch.long)
    if result.numel() != audit_num_tokens or int(torch.unique(result).numel()) != audit_num_tokens:
        raise RuntimeError("document-balanced audit selection is not unique and exact-size")
    return result


def _gate_statistics(
    runtime: RuntimeActivations,
    *,
    gate_thresholds: Sequence[float],
    activation_thresholds: Sequence[float],
) -> dict[str, Any]:
    # CUDA quantile has a practical 2**24-element indexing limit in several
    # supported PyTorch builds; a full 8192x4096 layer is twice that size.
    # Move the BF16 values first, then convert on CPU for exact full-population
    # statistics rather than silently subsampling them.
    gate = runtime.a_post_sigmoid.detach().reshape(-1).cpu().float().abs()
    post = runtime.c_post_gate.detach().reshape(-1).cpu().float().abs()
    quantile_levels = torch.tensor(
        [0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0],
        dtype=torch.float64,
    )
    gate_quantiles = exact_linear_quantiles(gate, quantile_levels)
    post_quantiles = exact_linear_quantiles(post, quantile_levels)
    labels = ("q00", "q01", "q05", "q10", "q25", "q50", "q75", "q90", "q95", "q99", "q100")
    return {
        "elements": int(gate.numel()),
        "mean_a": float(gate.mean(dtype=torch.float64)),
        "std_a": float(gate.double().std(unbiased=False)),
        "gate_abs_quantiles": {label: float(value) for label, value in zip(labels, gate_quantiles)},
        "gated_activation_abs_quantiles": {
            label: float(value) for label, value in zip(labels, post_quantiles)
        },
        "fraction_gate_abs_below": {
            f"{threshold:.12g}": float((gate < threshold).double().mean())
            for threshold in gate_thresholds
        },
        "fraction_gated_activation_abs_below": {
            f"{threshold:.12g}": float((post < threshold).double().mean())
            for threshold in activation_thresholds
        },
        "terminology": "effective_sparsity_not_exact_sparsity",
    }


class _OutputAccumulator:
    def __init__(self) -> None:
        self.teacher_energy = 0.0
        self.prediction_energy = 0.0
        self.cross = 0.0
        self.residual_energy = 0.0
        self.token_relative: list[torch.Tensor] = []
        self.near_zero = 0

    def update(self, teacher: torch.Tensor, prediction: torch.Tensor) -> None:
        teacher64 = teacher.double()
        prediction64 = prediction.double()
        teacher_per = teacher64.square().sum(dim=1)
        prediction_per = prediction64.square().sum(dim=1)
        residual_per = (teacher64 - prediction64).square().sum(dim=1)
        self.teacher_energy += float(teacher_per.sum())
        self.prediction_energy += float(prediction_per.sum())
        self.cross += float((teacher64 * prediction64).sum())
        self.residual_energy += float(residual_per.sum())
        valid = teacher_per > 1.0e-24
        self.near_zero += int((~valid).sum())
        self.token_relative.append((residual_per[valid] / teacher_per[valid]).cpu())

    def finalize(self) -> dict[str, Any]:
        if self.teacher_energy <= 0.0:
            raise ValueError("aggregate teacher energy is zero")
        values = torch.cat(self.token_relative).double()
        quantiles = torch.quantile(values, torch.tensor([0.5, 0.9, 0.95, 0.99], dtype=torch.float64))
        relative = self.residual_energy / self.teacher_energy
        normalized_cross = self.cross / self.teacher_energy
        prediction_teacher = self.prediction_energy / self.teacher_energy
        identity = 1.0 + prediction_teacher - 2.0 * normalized_cross
        cosine_denominator = math.sqrt(self.teacher_energy * self.prediction_energy)
        per_token = {
            "mean": float(values.mean()),
            "median": float(quantiles[0]),
            "p90": float(quantiles[1]),
            "p95": float(quantiles[2]),
            "p99": float(quantiles[3]),
        }
        return {
            "relative_mse": relative,
            "normalized_cross": normalized_cross,
            "prediction_teacher_energy": prediction_teacher,
            "cosine_similarity": self.cross / cosine_denominator if cosine_denominator > 0.0 else 0.0,
            "relative_mse_from_identity": identity,
            "relative_mse_identity_absolute_error": abs(relative - identity),
            "per_token_relative_squared_error": per_token,
            "per_token_relative_error": dict(per_token),
            "near_zero_teacher_tokens": self.near_zero,
            "valid_per_token_count": int(values.numel()),
            "teacher_energy": self.teacher_energy,
            "prediction_energy": self.prediction_energy,
            "cross_inner_product": self.cross,
            "residual_energy": self.residual_energy,
        }


def _chunked_main_metrics(
    runtime: RuntimeActivations,
    weight: torch.Tensor,
    teacher: torch.Tensor,
    indices: torch.Tensor,
    *,
    chunk_size: int,
) -> tuple[dict[str, Any], float]:
    accumulator = _OutputAccumulator()
    selected_c_energy = 0.0
    selected_a_energy = 0.0
    total_c_energy = float(runtime.c_post_gate.double().square().sum())
    total_a_energy = float(runtime.a_post_sigmoid.double().square().sum())
    started = time.perf_counter()
    for start in range(0, int(indices.shape[0]), chunk_size):
        stop = min(start + chunk_size, int(indices.shape[0]))
        chunk_indices = indices[start:stop]
        c = runtime.c_post_gate[start:stop]
        a = runtime.a_post_sigmoid[start:stop]
        prediction = selected_output(c, weight, chunk_indices)
        accumulator.update(teacher[start:stop], prediction)
        if chunk_indices.shape[1]:
            selected_c_energy += float(c.gather(1, chunk_indices).double().square().sum())
            selected_a_energy += float(a.gather(1, chunk_indices).double().square().sum())
        del prediction
    if weight.device.type == "cuda":
        torch.cuda.synchronize(weight.device)
    elapsed = time.perf_counter() - started
    metrics = accumulator.finalize()
    metrics.update(
        {
            "retained_input_energy": selected_c_energy / max(total_c_energy, 1.0e-300),
            "retained_gate_energy": selected_a_energy / max(total_a_energy, 1.0e-300),
        }
    )
    return metrics, elapsed


def _frequency_key(
    *, layer: int, scope: str, method: str, selected_k: int, seed: int | None
) -> str:
    safe_scope = re.sub(r"[^a-zA-Z0-9]+", "_", scope).strip("_")
    safe_method = re.sub(r"[^a-zA-Z0-9]+", "_", method).strip("_")
    return f"layer_{layer:02d}__{safe_scope}__{safe_method}__k_{selected_k:04d}__seed_{seed if seed is not None else 'none'}"


def _base_row(
    *,
    context: Mapping[str, Any],
    layer_index: int,
    scope: str,
    num_tokens: int,
    method: str,
    keep_ratio: float,
    selected_k_value: int,
    width: int,
    seed: int | None,
    metrics: Mapping[str, Any],
    gate_statistics: Mapping[str, Any],
    runtime_seconds: float,
    frequency_key: str,
    stability: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {
        "format": str(context.get("row_format", ROW_FORMAT)),
        "schema_version": 1,
        "model_id": context["model_id"],
        "model_revision": context["model_revision"],
        "layer": layer_index,
        "layer_index": layer_index,
        "dataset": context["dataset"],
        "document_hash": context["validation_document_hash"],
        "split": "validation",
        "scope": scope,
        "num_tokens": num_tokens,
        "method": method,
        "keep_ratio": keep_ratio,
        "selected_k": selected_k_value,
        "realized_ratio": selected_k_value / width,
        "seed": seed,
        "metrics": dict(metrics),
        "gate_statistics": dict(gate_statistics),
        "selection": {
            "fixed_cardinality": True,
            "stable_tie_breaking": method not in {"random", "random_expected"},
            "nested_across_ratios": True,
            "selection_stability": stability,
            "frequency_artifact_key": frequency_key,
        },
        "runtime_seconds": runtime_seconds,
        "git_commit": context["git_commit"],
        "training_free": True,
    }


def _endpoint_checks(row: Mapping[str, Any], *, width: int) -> None:
    k = int(row["selected_k"])
    relative = float(row["metrics"]["relative_mse"])
    identity_error = float(row["metrics"]["relative_mse_identity_absolute_error"])
    if identity_error > 2.0e-10:
        raise RuntimeError(f"metric identity failed: {identity_error:.6g}")
    if k == 0 and abs(relative - 1.0) > 2.0e-10:
        raise RuntimeError(f"zero-retention endpoint is not unit relative MSE: {relative:.6g}")
    if k == width and relative > 2.0e-10:
        raise RuntimeError(f"full-retention endpoint is not exact: {relative:.6g}")


def _evaluate_ranked_method(
    *,
    runtime: RuntimeActivations,
    weight: torch.Tensor,
    teacher: torch.Tensor,
    audit_indices: torch.Tensor,
    method: str,
    seed: int | None,
    ratios: Sequence[float],
    chunk_size: int,
    layer_index: int,
    context: Mapping[str, Any],
    main_gate_statistics: Mapping[str, Any],
    audit_gate_statistics: Mapping[str, Any],
    frequencies: dict[str, torch.Tensor],
) -> list[dict[str, Any]]:
    num_tokens, width = map(int, runtime.c_post_gate.shape)
    ranking_started = time.perf_counter()
    if method == "random":
        assert seed is not None
        ranking = random_rankings(num_tokens, width, seed=seed, device=weight.device)
    else:
        ranking = stable_descending_ranking(score_channels(runtime, weight, method))
    if weight.device.type == "cuda":
        torch.cuda.synchronize(weight.device)
    ranking_seconds = time.perf_counter() - ranking_started
    audit_device_indices = audit_indices.to(weight.device)
    audit_runtime = RuntimeActivations(
        h_pre_gate=runtime.h_pre_gate.index_select(0, audit_device_indices),
        gate_logits=runtime.gate_logits.index_select(0, audit_device_indices),
        a_post_sigmoid=runtime.a_post_sigmoid.index_select(0, audit_device_indices),
        c_post_gate=runtime.c_post_gate.index_select(0, audit_device_indices),
    )
    audit_teacher = teacher.index_select(0, audit_device_indices)
    audit_ranking = ranking.index_select(0, audit_device_indices)
    rows: list[dict[str, Any]] = []
    for ratio in ratios:
        k = retained_k(ratio, width)
        chosen = selected_indices(ranking, k)
        main_metrics, evaluation_seconds = _chunked_main_metrics(
            runtime, weight, teacher, chosen, chunk_size=chunk_size
        )
        audit_chosen = selected_indices(audit_ranking, k)
        audit_started = time.perf_counter()
        audit_prediction = selected_output(audit_runtime.c_post_gate, weight, audit_chosen)
        audit_metrics = output_metrics(audit_teacher, audit_prediction)
        audit_metrics.update(
            retained_energy_metrics(
                audit_runtime.c_post_gate,
                audit_runtime.a_post_sigmoid,
                audit_chosen,
            )
        )
        if weight.device.type == "cuda":
            torch.cuda.synchronize(weight.device)
        audit_evaluation_seconds = time.perf_counter() - audit_started
        stability = selection_stability(audit_chosen, width)
        for scope, tokens, metrics, gate_stats, selected, scope_runtime in (
            (MAIN_SCOPE, num_tokens, main_metrics, main_gate_statistics, chosen, evaluation_seconds),
            (
                AUDIT_SCOPE,
                int(audit_indices.numel()),
                audit_metrics,
                audit_gate_statistics,
                audit_chosen,
                audit_evaluation_seconds,
            ),
        ):
            frequency_key = _frequency_key(
                layer=layer_index,
                scope=scope,
                method=method,
                selected_k=k,
                seed=seed,
            )
            frequencies[frequency_key] = selection_frequency(selected, width).float().cpu().contiguous()
            row = _base_row(
                context=context,
                layer_index=layer_index,
                scope=scope,
                num_tokens=tokens,
                method=method,
                keep_ratio=ratio,
                selected_k_value=k,
                width=width,
                seed=seed,
                metrics=metrics,
                gate_statistics=gate_stats,
                runtime_seconds=scope_runtime + ranking_seconds,
                frequency_key=frequency_key,
                stability=stability,
            )
            row["runtime_breakdown"] = {
                "ranking_seconds_shared_across_ratios": ranking_seconds,
                "scatter_gemm_and_metrics_seconds": scope_runtime,
            }
            row["selection"]["selection_stability_scope"] = AUDIT_SCOPE
            row["selection"]["selection_stability_indices_sha256"] = context[
                "audit_indices_sha256"
            ]
            _endpoint_checks(row, width=width)
            rows.append(row)
        del chosen, audit_chosen, audit_prediction
    return rows


def _evaluate_analytic_random(
    *,
    runtime: RuntimeActivations,
    weight: torch.Tensor,
    audit_indices: torch.Tensor,
    ratios: Sequence[float],
    layer_index: int,
    context: Mapping[str, Any],
    main_gate_statistics: Mapping[str, Any],
    audit_gate_statistics: Mapping[str, Any],
    frequencies: dict[str, torch.Tensor],
) -> list[dict[str, Any]]:
    num_tokens, width = map(int, runtime.c_post_gate.shape)
    audit_device_indices = audit_indices.to(weight.device)
    rows: list[dict[str, Any]] = []
    for scope, c, a, gate_stats in (
        (MAIN_SCOPE, runtime.c_post_gate, runtime.a_post_sigmoid, main_gate_statistics),
        (
            AUDIT_SCOPE,
            runtime.c_post_gate.index_select(0, audit_device_indices),
            runtime.a_post_sigmoid.index_select(0, audit_device_indices),
            audit_gate_statistics,
        ),
    ):
        statistics = random_expectation_statistics(c, weight)
        for ratio in ratios:
            k = retained_k(ratio, width)
            started = time.perf_counter()
            metrics = analytic_random_expected_metrics(
                c,
                weight,
                k,
                a_post_sigmoid=a,
                statistics=statistics,
            )
            if weight.device.type == "cuda":
                torch.cuda.synchronize(weight.device)
            elapsed = time.perf_counter() - started
            frequency_key = _frequency_key(
                layer=layer_index,
                scope=scope,
                method="random_expected",
                selected_k=k,
                seed=None,
            )
            frequencies[frequency_key] = torch.full(
                (width,), k / width, dtype=torch.float32
            )
            row = _base_row(
                context=context,
                layer_index=layer_index,
                scope=scope,
                num_tokens=int(c.shape[0]),
                method="random_expected",
                keep_ratio=ratio,
                selected_k_value=k,
                width=width,
                seed=None,
                metrics=metrics,
                gate_statistics=gate_stats,
                runtime_seconds=elapsed,
                frequency_key=frequency_key,
                stability=None,
            )
            row["selection"]["frequency_semantics"] = "expected_selection_probability"
            row["selection"]["selection_stability"] = {
                "definition": "not_applicable_to_first_moment_analytic_expectation"
            }
            _endpoint_checks(row, width=width)
            rows.append(row)
    return rows


def _evaluate_greedy_audit(
    *,
    c: torch.Tensor,
    a: torch.Tensor,
    weight: torch.Tensor,
    ratios: Sequence[float],
    layer_index: int,
    context: Mapping[str, Any],
    audit_gate_statistics: Mapping[str, Any],
    frequencies: dict[str, torch.Tensor],
) -> list[dict[str, Any]]:
    if c.shape != a.shape or c.device != weight.device or a.device != weight.device:
        raise ValueError("greedy audit tensors have incompatible shape or device")
    width = int(c.shape[1])
    checkpoints = tuple(retained_k(ratio, width) for ratio in ratios)
    started = time.perf_counter()
    result = greedy_residual_reference(c, weight, checkpoints)
    if weight.device.type == "cuda":
        torch.cuda.synchronize(weight.device)
    elapsed = time.perf_counter() - started
    rows: list[dict[str, Any]] = []
    for ratio, k in zip(ratios, checkpoints):
        checkpoint = result.checkpoints[k]
        # One checkpoint-level transfer avoids hundreds of tiny GPU
        # synchronizations while deriving per-token diagnostic positions.
        checkpoint_gains = result.selected_gains[:, :k].detach().cpu()
        negative_mask = checkpoint_gains < 0
        nonpositive_mask = checkpoint_gains <= 0
        first_negative_step_by_token: list[int | None] = []
        first_nonpositive_step_by_token: list[int | None] = []
        for token_negative, token_nonpositive in zip(
            negative_mask, nonpositive_mask, strict=True
        ):
            negative_locations = torch.nonzero(
                token_negative, as_tuple=False
            ).flatten()
            nonpositive_locations = torch.nonzero(
                token_nonpositive, as_tuple=False
            ).flatten()
            first_negative_step_by_token.append(
                None
                if negative_locations.numel() == 0
                else int(negative_locations[0]) + 1
            )
            first_nonpositive_step_by_token.append(
                None
                if nonpositive_locations.numel() == 0
                else int(nonpositive_locations[0]) + 1
            )
        greedy_diagnostics = {
            **dict(result.diagnostics),
            "diagnostic_selected_k": k,
            "first_negative_step_by_token": first_negative_step_by_token,
            "first_nonpositive_step_by_token": first_nonpositive_step_by_token,
            "tokens_with_negative_gain": sum(
                step is not None for step in first_negative_step_by_token
            ),
            "tokens_with_nonpositive_gain": sum(
                step is not None for step in first_nonpositive_step_by_token
            ),
            "negative_gain_count": int(negative_mask.sum()),
            "nonpositive_gain_count": int(nonpositive_mask.sum()),
            "minimum_selected_gain": (
                float(checkpoint_gains.min()) if checkpoint_gains.numel() else None
            ),
            "maximum_residual_increase": (
                float((-checkpoint_gains[negative_mask]).max())
                if bool(negative_mask.any())
                else 0.0
            ),
        }
        metrics = output_metrics(result.teacher, checkpoint.prediction)
        metrics.update(retained_energy_metrics(c, a, checkpoint.selected_indices))
        stability = selection_stability(checkpoint.selected_indices, width)
        frequency_key = _frequency_key(
            layer=layer_index,
            scope=AUDIT_SCOPE,
            method="greedy_residual_reference",
            selected_k=k,
            seed=None,
        )
        frequencies[frequency_key] = selection_frequency(
            checkpoint.selected_indices, width
        ).float().cpu().contiguous()
        row = _base_row(
            context=context,
            layer_index=layer_index,
            scope=AUDIT_SCOPE,
            num_tokens=int(c.shape[0]),
            method="greedy_residual_reference",
            keep_ratio=ratio,
            selected_k_value=k,
            width=width,
            seed=None,
            metrics=metrics,
            gate_statistics=audit_gate_statistics,
            runtime_seconds=elapsed,
            frequency_key=frequency_key,
            stability=stability,
        )
        row["greedy_diagnostics"] = greedy_diagnostics
        row["greedy_checkpoint"] = {
            "maximum_recurrence_absolute_error": checkpoint.maximum_recurrence_absolute_error,
            "maximum_recurrence_relative_error": checkpoint.maximum_recurrence_relative_error,
            "direct_residual_energy_sum": float(checkpoint.direct_residual_energy.sum()),
            "recurrence_residual_energy_sum": float(
                checkpoint.recurrence_residual_energy.sum()
            ),
        }
        rows.append(row)
    return rows


def _summary_csv(rows: Sequence[Mapping[str, Any]]) -> str:
    columns = (
        "scope",
        "layer",
        "method",
        "seed",
        "keep_ratio",
        "selected_k",
        "realized_ratio",
        "num_tokens",
        "relative_mse",
        "normalized_cross",
        "prediction_teacher_energy",
        "cosine_similarity",
        "per_token_error_mean",
        "per_token_error_median",
        "per_token_error_p90",
        "per_token_error_p95",
        "per_token_error_p99",
        "retained_input_energy",
        "retained_gate_energy",
        "mean_pairwise_jaccard",
        "runtime_seconds",
    )
    from io import StringIO

    stream = StringIO()
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        metrics = row["metrics"]
        token = metrics["per_token_relative_squared_error"]
        stability = row["selection"].get("selection_stability") or {}
        writer.writerow(
            {
                "scope": row["scope"],
                "layer": row["layer"],
                "method": row["method"],
                "seed": "" if row["seed"] is None else row["seed"],
                "keep_ratio": row["keep_ratio"],
                "selected_k": row["selected_k"],
                "realized_ratio": row["realized_ratio"],
                "num_tokens": row["num_tokens"],
                "relative_mse": metrics["relative_mse"],
                "normalized_cross": metrics["normalized_cross"],
                "prediction_teacher_energy": metrics["prediction_teacher_energy"],
                "cosine_similarity": metrics["cosine_similarity"],
                "per_token_error_mean": token["mean"],
                "per_token_error_median": token["median"],
                "per_token_error_p90": token["p90"],
                "per_token_error_p95": token["p95"],
                "per_token_error_p99": token["p99"],
                "retained_input_energy": metrics["retained_input_energy"],
                "retained_gate_energy": metrics["retained_gate_energy"],
                "mean_pairwise_jaccard": stability.get("mean_pairwise_jaccard"),
                "runtime_seconds": row["runtime_seconds"],
            }
        )
    return stream.getvalue()


def _random_expectation_validation(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Compare seed-averaged Monte Carlo rows with the analytic baseline."""

    records: list[dict[str, Any]] = []
    expected_rows = [row for row in rows if row["method"] == "random_expected"]
    metric_names = (
        "relative_mse",
        "normalized_cross",
        "prediction_teacher_energy",
    )
    for expected in expected_rows:
        matches = [
            row
            for row in rows
            if row["method"] == "random"
            and row["layer"] == expected["layer"]
            and row["scope"] == expected["scope"]
            and row["selected_k"] == expected["selected_k"]
        ]
        if not matches:
            continue
        record: dict[str, Any] = {
            "layer": expected["layer"],
            "scope": expected["scope"],
            "selected_k": expected["selected_k"],
            "realized_ratio": expected["realized_ratio"],
            "monte_carlo_seed_count": len(matches),
            "seeds": sorted(int(row["seed"]) for row in matches),
        }
        for name in metric_names:
            empirical = sum(float(row["metrics"][name]) for row in matches) / len(matches)
            analytic = float(expected["metrics"][name])
            record[f"monte_carlo_mean_{name}"] = empirical
            record[f"analytic_{name}"] = analytic
            record[f"monte_carlo_minus_analytic_{name}"] = empirical - analytic
        records.append(record)
    return records


def _markdown(rows: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any]) -> str:
    lines = [
        "# Qwen3.5 Gated-Attention Effective Sparsity Oracle",
        "",
        (
            "Training-free, validation-only output analysis. Gates and post-gate "
            "activations use the captured BF16 sigmoid/multiply path; `c` and "
            "`o_proj` weights are converted to FP32 for output products."
        ),
        "",
        (
            "The decisive metric is held-out `o_proj` output relative MSE. "
            "Retained activation or gate energy alone is not evidence of success."
        ),
        "",
        f"Main scope: **{MAIN_SCOPE}** ({manifest['num_tokens']} validation tokens).",
        "",
        (
            f"Audit scope: **{AUDIT_SCOPE}** ({manifest['audit']['num_tokens']} "
            "document-balanced tokens). Audit rows, including the privileged "
            "greedy residual reference, must never be compared as full-validation rows."
        ),
        "",
        (
            "Random seeds are Monte-Carlo checks of one analytic random "
            "baseline; they are not independent validation datasets."
        ),
        "",
        (
            "This oracle diagnoses a possible sparse interface only. By itself "
            "it does **not** shorten the row-parallel `o_proj` AllReduce."
        ),
        "",
    ]
    for layer in manifest["layers"]:
        lines.extend(
            [
                f"## Layer {layer}",
                "",
                "| Method | Seed | Keep ratio | Relative MSE | Normalized cross | Prediction/teacher energy | P95 token error | Retained input energy |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        main_rows = [row for row in rows if row["layer"] == layer and row["scope"] == MAIN_SCOPE]
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
                        f"{metrics['normalized_cross']:.6g}",
                        f"{metrics['prediction_teacher_energy']:.6g}",
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
                "### Shared document-balanced audit (all methods)",
                "",
                "Every row below uses the identical deterministic audit indices; none is a full-validation result.",
                "",
                "| Method | Seed | Keep ratio | Relative MSE | Normalized cross | Prediction/teacher energy | P95 token error | Retained input energy |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
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
                        f"{metrics['normalized_cross']:.6g}",
                        f"{metrics['prediction_teacher_energy']:.6g}",
                        f"{metrics['per_token_relative_squared_error']['p95']:.6g}",
                        f"{metrics['retained_input_energy']:.6g}",
                    ]
                )
                + " |"
            )
        greedy_rows = [
            row
            for row in rows
            if row["layer"] == layer
            and row["scope"] == AUDIT_SCOPE
            and row["method"] == "greedy_residual_reference"
        ]
        lines.extend(
            [
                "",
                "### Privileged greedy audit only",
                "",
                "These 128-token rows use the dense target and are a fixed-k greedy reference, not a global optimum or a full-validation result.",
                "",
                "| Keep ratio | Relative MSE | Negative gains | First nonpositive tokens | Min gain | Recurrence max abs error |",
                "|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in greedy_rows:
            diagnostics = row["greedy_diagnostics"]
            checkpoint = row["greedy_checkpoint"]
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"{row['realized_ratio']:.6f}",
                        f"{row['metrics']['relative_mse']:.6g}",
                        str(diagnostics["negative_gain_count"]),
                        str(diagnostics["tokens_with_nonpositive_gain"]),
                        str(diagnostics["minimum_selected_gain"]),
                        f"{checkpoint['maximum_recurrence_absolute_error']:.6g}",
                    ]
                )
                + " |"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _plot(
    rows: Sequence[Mapping[str, Any]],
    layers: Sequence[int],
    path: Path,
    *,
    scope: str,
    title: str,
    layer_title_prefix: str = "Full-attention layer",
) -> None:
    """Write a dependency-free SVG curve plot."""

    panel_width = 560
    height = 480
    width = panel_width * len(layers)
    plot_width = 350
    plot_height = 320
    top = 70
    palette = (
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    )
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:DejaVu Sans,Arial,sans-serif;fill:#222}.tick{font-size:11px}.label{font-size:12px}.title{font-size:15px;font-weight:600}.legend{font-size:10px}</style>',
        f'<text x="{width / 2:.1f}" y="24" text-anchor="middle" class="title">{escape(title)}</text>',
    ]
    for panel_index, layer in enumerate(layers):
        left = panel_index * panel_width + 62
        bottom = top + plot_height
        layer_rows = [row for row in rows if row["scope"] == scope and row["layer"] == layer]
        methods = tuple(dict.fromkeys(str(row["method"]) for row in layer_rows))
        grouped_by_method: dict[str, tuple[list[float], list[float]]] = {}
        maximum_y = 0.0
        for method in methods:
            method_rows = [row for row in layer_rows if row["method"] == method]
            grouped: dict[float, list[float]] = {}
            for row in method_rows:
                grouped.setdefault(float(row["realized_ratio"]), []).append(
                    float(row["metrics"]["relative_mse"])
                )
            xs = sorted(grouped)
            ys = [sum(grouped[x]) / len(grouped[x]) for x in xs]
            grouped_by_method[method] = (xs, ys)
            maximum_y = max(maximum_y, max(ys, default=0.0))
        y_limit = max(1.0, maximum_y * 1.05, 1.0e-12)
        for tick_index in range(5):
            x_value = tick_index / 4
            x_pixel = left + x_value * plot_width
            y_value = tick_index / 4 * y_limit
            y_pixel = bottom - tick_index / 4 * plot_height
            elements.extend(
                [
                    f'<line x1="{x_pixel:.2f}" y1="{top}" x2="{x_pixel:.2f}" y2="{bottom}" stroke="#ddd" stroke-width="1"/>',
                    f'<text x="{x_pixel:.2f}" y="{bottom + 18}" text-anchor="middle" class="tick">{x_value:.2g}</text>',
                    f'<line x1="{left}" y1="{y_pixel:.2f}" x2="{left + plot_width}" y2="{y_pixel:.2f}" stroke="#ddd" stroke-width="1"/>',
                    f'<text x="{left - 8}" y="{y_pixel + 4:.2f}" text-anchor="end" class="tick">{y_value:.3g}</text>',
                ]
            )
        elements.extend(
            [
                f'<rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" fill="none" stroke="#333" stroke-width="1"/>',
                f'<text x="{left + plot_width / 2:.2f}" y="{bottom + 42}" text-anchor="middle" class="label">Retained-channel ratio</text>',
                f'<text x="{left - 48}" y="{top + plot_height / 2:.2f}" text-anchor="middle" class="label" transform="rotate(-90 {left - 48} {top + plot_height / 2:.2f})">Output relative MSE</text>',
                f'<text x="{left + plot_width / 2:.2f}" y="{top - 14}" text-anchor="middle" class="title">{escape(layer_title_prefix)} {layer}</text>',
            ]
        )
        legend_x = left + plot_width + 18
        for method_index, (method, (xs, ys)) in enumerate(grouped_by_method.items()):
            color = palette[method_index % len(palette)]
            points = " ".join(
                f"{left + x * plot_width:.2f},{bottom - y / y_limit * plot_height:.2f}"
                for x, y in zip(xs, ys, strict=True)
            )
            if points:
                elements.append(
                    f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>'
                )
                for x, y in zip(xs, ys, strict=True):
                    elements.append(
                        f'<circle cx="{left + x * plot_width:.2f}" cy="{bottom - y / y_limit * plot_height:.2f}" r="2.8" fill="{color}"/>'
                    )
            legend_y = top + 10 + method_index * 18
            elements.extend(
                [
                    f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 18}" y2="{legend_y}" stroke="{color}" stroke-width="2"/>',
                    f'<text x="{legend_x + 23}" y="{legend_y + 4}" class="legend">{escape(method)}</text>',
                ]
            )
    elements.append("</svg>")
    _write_text(path, "\n".join(elements) + "\n")


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.torch_num_threads)
    if args.num_tokens != 8192:
        raise ValueError("this validation-only main screen requires exactly --num-tokens 8192")
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
        args.layers,
        full_attention_layers=geometry.full_attention_layers,
        snapshot_layers=set(layer_records),
    )
    expected_representative = _representative_layers(geometry.full_attention_layers)
    if args.layers.strip().lower() == "representative" and layers != expected_representative:
        raise RuntimeError("representative layer discovery did not select first/middle/last full attention")
    validated_snapshots = _validate_snapshot_files(
        snapshot_dir,
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
            print(f"[EffectiveSparsity] loading layer={layer_index}", flush=True)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            layer_started = time.perf_counter()
            tensors = load_file(validated_snapshots[layer_index]["path"], device="cpu")
            h = tensors["h_pre_gate"].to(device=device)
            gate_logits = tensors["gate_logits"].to(device=device)
            runtime = reconstruct_bf16_runtime(h, gate_logits)
            weight_cpu, weight_source = _load_o_proj(
                model_path, weight_map, layer_index, geometry
            )
            weight = weight_cpu.to(device=device, dtype=torch.float32)
            teacher = runtime.c_post_gate.float() @ weight.transpose(0, 1)
            if not bool(torch.isfinite(teacher).all()):
                raise FloatingPointError(f"layer {layer_index} teacher contains non-finite values")
            main_gate_statistics = _gate_statistics(
                runtime,
                gate_thresholds=gate_thresholds,
                activation_thresholds=activation_thresholds,
            )
            audit_device_indices = audit_indices.to(device)
            audit_runtime = RuntimeActivations(
                h_pre_gate=runtime.h_pre_gate.index_select(0, audit_device_indices),
                gate_logits=runtime.gate_logits.index_select(0, audit_device_indices),
                a_post_sigmoid=runtime.a_post_sigmoid.index_select(0, audit_device_indices),
                c_post_gate=runtime.c_post_gate.index_select(0, audit_device_indices),
            )
            audit_gate_statistics = _gate_statistics(
                audit_runtime,
                gate_thresholds=gate_thresholds,
                activation_thresholds=activation_thresholds,
            )
            for method in methods:
                print(f"[EffectiveSparsity] layer={layer_index} method={method}", flush=True)
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
                # Keep only the small matched audit and one BF16 weight matrix
                # on CPU.  Greedy runs after every layer's main sweep has been
                # completed and staged, so it cannot erase or delay later main
                # layers if its privileged reference fails.
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
                    int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
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
            del tensors, h, gate_logits, runtime, audit_runtime, weight_cpu, weight, teacher
            if device.type == "cuda":
                torch.cuda.empty_cache()

        if not args.skip_greedy:
            for layer_index in layers:
                payload = greedy_inputs[layer_index]
                print(
                    f"[EffectiveSparsity] layer={layer_index} "
                    "method=greedy_residual_reference",
                    flush=True,
                )
                greedy_started = time.perf_counter()
                greedy_c = None
                greedy_a = None
                greedy_weight = None
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
                    print(
                        "[EffectiveSparsity][GreedyWarning] "
                        + _json_text(failure),
                        flush=True,
                    )
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

    nongreedy_results_path = partial_dir / "nongreedy_results.jsonl"
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
            "semantics": "per-token selection frequency; random_expected stores expectation",
        },
    )
    plot_path = partial_dir / "sparsity_curve.svg"
    _plot(
        rows,
        layers,
        plot_path,
        scope=MAIN_SCOPE,
        title="Qwen3.5 gated-attention effective sparsity (8192-token main scope)",
    )
    audit_plot_path = partial_dir / "audit_sparsity_curve.svg"
    _plot(
        rows,
        layers,
        audit_plot_path,
        scope=AUDIT_SCOPE,
        title="Qwen3.5 gated-attention effective sparsity (shared 128-token audit)",
    )

    completed = datetime.now(timezone.utc).isoformat()
    result_manifest: dict[str, Any] = {
        "format": FORMAT,
        "schema_version": 1,
        "timestamp_started": timestamp_started,
        "timestamp_completed": completed,
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
            "num_query_heads": geometry.num_query_heads,
            "num_kv_heads": geometry.num_kv_heads,
            "head_dim": geometry.head_dim,
            "full_attention_layers": list(geometry.full_attention_layers),
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
            "nongreedy_results_staged_before_greedy": True,
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
            "gate_sigmoid_and_multiply_dtype": "bfloat16",
            "post_gate_and_o_proj_work_dtype": "float32",
            "metric_reduction_dtype": "float64",
            "stable_score_ties": True,
            "score_output_evaluation": "chunked_token_dependent_scatter_plus_GEMM",
        },
        "scope_guards": {
            "training_free": True,
            "fitting": False,
            "backward": False,
            "optimizer": None,
            "checkpoint_modified": False,
            "greedy_label": "privileged_target_aware_fixed_k_reference_not_global_oracle",
            "audit_never_compared_as_full_validation": True,
            "does_not_reduce_row_parallel_o_proj_allreduce_by_itself": True,
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
        nongreedy_results_path,
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
    _write_text(partial_dir / "manifest.json", _json_text(result_manifest, indent=2) + "\n")
    os.replace(partial_dir, output_dir)
    print(f"[Saved] {output_dir}", flush=True)


if __name__ == "__main__":
    main()
