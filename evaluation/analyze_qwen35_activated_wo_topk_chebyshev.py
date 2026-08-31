#!/usr/bin/env python3
"""Compare post-gate top-k and degree-8 gate folding at Qwen3.5 Wo inputs.

This training-free offline analysis reuses aligned document-isolated C4
snapshots.  It evaluates full-attention sigmoid gates and GDN SiLU gates with
the exact checkpoint BF16 gate/RMSNorm ordering.  Validation is never used to
select an interval, ridge, or polynomial coefficients.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import sys
import time
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor
from torch.nn import functional as F
from safetensors import safe_open
from safetensors.torch import load_file, save_file


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basisserve.analysis.activated_wo_topk_chebyshev import (  # noqa: E402
    GateFamily,
    GatePolynomialVariant,
    build_gate_wire_components,
    gate_chebyshev_polynomial,
    gate_component_wire_relative_mse,
    gate_latent_from_components,
    gate_normal_equations_from_components,
    gate_polynomial_candidate_activation,
    scalar_gate_diagnostics,
    solve_gate_ridge_coefficients,
    variants_for_family,
)
from basisserve.analysis.mlp_topk_chebyshev import (  # noqa: E402
    CanonicalWire,
    canonical_output_metrics,
)
from basisserve.calibration.mlp_snapshots import SPLITS  # noqa: E402
from basisserve.diagnostics.qwen35_gated_attention_sparsity import (  # noqa: E402
    reconstruct_bf16_runtime,
    reconstruct_gdn_silu_runtime,
)
from evaluation.analyze_qwen35_gated_attention_sparsity import (  # noqa: E402
    _file_sha256,
    _git_commit,
    _installed_version,
    _load_o_proj as load_full_o_proj,
    _validate_manifest as validate_full_manifest,
    _validate_model as validate_full_model,
)
from evaluation.analyze_qwen35_gdn_silu_sparsity import (  # noqa: E402
    _load_o_proj as load_gdn_o_proj,
    _validate_manifest as validate_gdn_manifest,
    _validate_model as validate_gdn_model,
)
from evaluation.analyze_qwen35_mlp_topk_chebyshev import (  # noqa: E402
    _atomic_json,
    _atomic_text,
    _estimate_radii,
    _topk_records_for_layer,
    _wire_bank,
)


FORMAT = "basisserve.qwen35.activated_wo_topk_chebyshev.v1"
ROW_FORMAT = "basisserve.qwen35.activated_wo_topk_chebyshev.row.v1"
DEFAULT_FULL_SNAPSHOTS = (
    REPO_ROOT / "results/qwen35_9b_gated_wo_foldability/"
    "c4_train128_dev32_val64_s2048_p128_seed20260807_l3_15_31"
)
DEFAULT_GDN_SNAPSHOTS = (
    REPO_ROOT / "results/qwen35_9b_gdn_foldability/"
    "c4_train128_dev32_val64_s2048_p128_seed20260807_l2_14_30"
)
DEFAULT_RATIOS = (0.1, 0.2, 0.3, 0.5, 0.75, 1.0)
DEFAULT_BLOCK_SIZES = (1, 8, 16, 32)
DEFAULT_DEGREES = (2, 4, 6, 8)
DEFAULT_QUANTILES = (0.999, 0.9999)
DEFAULT_RIDGES = (0.0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-snapshots", default=str(DEFAULT_FULL_SNAPSHOTS))
    parser.add_argument("--gdn-snapshots", default=str(DEFAULT_GDN_SNAPSHOTS))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-path")
    parser.add_argument("--full-layers", default="3,15,31")
    parser.add_argument("--gdn-layers", default="2,14,30")
    parser.add_argument("--train-rows", type=int, default=8192)
    parser.add_argument("--dev-rows", type=int, default=2048)
    parser.add_argument("--validation-rows", type=int, default=8192)
    parser.add_argument("--wire-ranks", default="1536,2048,4096")
    parser.add_argument("--topk-ratios", default=",".join(map(str, DEFAULT_RATIOS)))
    parser.add_argument("--topk-scopes", default="global_oracle,tp_local")
    parser.add_argument("--topk-scores", default="magnitude,decoder_weighted")
    parser.add_argument(
        "--topk-block-sizes",
        default=",".join(map(str, DEFAULT_BLOCK_SIZES)),
    )
    parser.add_argument("--cheb-degrees", default=",".join(map(str, DEFAULT_DEGREES)))
    parser.add_argument(
        "--cheb-interval-quantiles",
        default=",".join(map(str, DEFAULT_QUANTILES)),
    )
    parser.add_argument("--cheb-modes", default="global,clipped")
    parser.add_argument("--ridge-values", default=",".join(map(str, DEFAULT_RIDGES)))
    parser.add_argument("--primary-degree", type=int, default=8)
    parser.add_argument("--tp", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--metric-chunk-size", type=int, default=256)
    parser.add_argument("--quantile-max-elements", type=int, default=4_000_000)
    parser.add_argument("--pod-oversample", type=int, default=16)
    parser.add_argument("--pod-niter", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--polynomial-pass-threshold", type=float, default=0.01)
    parser.add_argument("--topk-pass-threshold", type=float, default=0.01)
    parser.add_argument("--topk-pass-maximum-keep", type=float, default=0.5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--torch-num-threads", type=int, default=4)
    return parser.parse_args()


def _csv_ints(raw: str) -> tuple[int, ...]:
    return tuple(int(piece.strip()) for piece in raw.split(",") if piece.strip())


def _csv_floats(raw: str) -> tuple[float, ...]:
    return tuple(float(piece.strip()) for piece in raw.split(",") if piece.strip())


def _csv_strings(raw: str) -> tuple[str, ...]:
    return tuple(piece.strip() for piece in raw.split(",") if piece.strip())


def _records_by_layer(manifest: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    records = {
        int(record["layer_index"]): record for record in manifest.get("layers", ())
    }
    if len(records) != len(manifest.get("layers", ())):
        raise ValueError("snapshot manifest contains duplicate layer records")
    return records


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_aligned_provenance(
    full_manifest: Mapping[str, Any],
    gdn_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    full_collection = full_manifest["collection"]
    gdn_collection = gdn_manifest["collection"]
    comparisons = {
        "collection_seed": (full_collection["seed"], gdn_collection["seed"]),
        "sequence_length": (
            full_collection["sequence_length"],
            gdn_collection["sequence_length"],
        ),
        "tokens_per_sequence": (
            full_collection["tokens_per_sequence"],
            gdn_collection["tokens_per_sequence"],
        ),
        "split_samples": (
            full_collection["split_samples"],
            gdn_collection["split_samples"],
        ),
        "sampled_position_sha256": (
            full_collection["sampled_position_sha256"],
            gdn_collection["sampled_position_sha256"],
        ),
        "sampled_position_records_sha256": (
            full_collection["sampled_position_records_sha256"],
            gdn_collection["sampled_position_records_sha256"],
        ),
        "combined_document_sha256": (
            full_collection["split_metadata"]["combined_record_sha256"],
            gdn_collection["split_metadata"]["combined_record_sha256"],
        ),
    }
    mismatched = [name for name, pair in comparisons.items() if pair[0] != pair[1]]
    if mismatched:
        raise ValueError(
            f"full-attention/GDN snapshot provenance differs: {mismatched}"
        )
    return {name: pair[0] for name, pair in comparisons.items()}


def _validate_split_file(
    snapshot_dir: Path,
    layer_record: Mapping[str, Any],
    *,
    layer: int,
    split: str,
    family: str,
    required_rows: int,
) -> dict[str, Any]:
    if split not in layer_record["splits"]:
        raise ValueError(f"layer {layer} has no {split} snapshot")
    record = layer_record["splits"][split]
    rows = int(record["rows"])
    if rows < required_rows:
        raise ValueError(
            f"layer {layer} {split} has {rows} rows, requires {required_rows}"
        )
    component = Path(str(record["file"]))
    if component.is_absolute() or ".." in component.parts:
        raise ValueError("unsafe snapshot path")
    path = snapshot_dir / component
    if not path.is_file():
        raise FileNotFoundError(path)
    observed_hash = _file_sha256(path)
    if observed_hash != str(record["file_sha256"]):
        raise ValueError(f"snapshot hash mismatch: {path}")
    expected_keys = (
        {"h_pre_gate", "gate_logits"}
        if family == "full_attention"
        else {"raw_core", "gate_preactivation", "norm_weight"}
    )
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        if keys != expected_keys:
            raise ValueError(f"unexpected tensors in {path}: {sorted(keys)}")
        activation_name = "h_pre_gate" if family == "full_attention" else "raw_core"
        activation = handle.get_tensor(activation_name)
        if activation.dtype != torch.bfloat16 or int(activation.shape[0]) != rows:
            raise TypeError(f"snapshot activation metadata differs in {path}")
    diagnostics = layer_record["diagnostics"][split]
    relative_names = (
        ("postgate_reconstruction_relative_mse", "direct_o_proj_relative_mse")
        if family == "full_attention"
        else (
            "postgate_reconstruction_relative_mse",
            "norm_to_out_proj_wire_relative_mse",
            "direct_out_proj_relative_mse",
        )
    )
    if any(float(diagnostics[name]) > 2.0e-5 for name in relative_names):
        raise ValueError(f"layer {layer} {split} capture consistency check failed")
    return {
        "path": str(path),
        "file_sha256": observed_hash,
        "source_rows": rows,
        "used_rows": int(required_rows),
        "selection": "deterministic_prefix",
    }


def _validate_snapshot_bank(
    snapshot_dir: Path,
    manifest: Mapping[str, Any],
    *,
    layers: Sequence[int],
    family: str,
    row_targets: Mapping[str, int],
) -> dict[int, dict[str, dict[str, Any]]]:
    records = _records_by_layer(manifest)
    missing = sorted(set(layers) - set(records))
    if missing:
        raise ValueError(f"missing {family} snapshots for layers {missing}")
    return {
        layer: {
            split: _validate_split_file(
                snapshot_dir,
                records[layer],
                layer=layer,
                split=split,
                family=family,
                required_rows=int(row_targets[split]),
            )
            for split in SPLITS
        }
        for layer in layers
    }


@torch.no_grad()
def _prepare_layer_snapshots(
    validated: Mapping[str, Mapping[str, Any]],
    *,
    family: str,
    geometry: Any,
    weight: Tensor,
    device: torch.device,
) -> dict[str, dict[str, Tensor]]:
    result: dict[str, dict[str, Tensor]] = {}
    weight_device = weight.to(device=device)
    for split in SPLITS:
        record = validated[split]
        rows = int(record["used_rows"])
        tensors = load_file(record["path"], device="cpu")
        if family == "full_attention":
            gate = tensors["gate_logits"][:rows].contiguous()
            pre = tensors["h_pre_gate"][:rows].contiguous()
            runtime = reconstruct_bf16_runtime(pre.to(device), gate.to(device))
        elif family == "gdn":
            gate = tensors["gate_preactivation"][:rows].contiguous()
            raw = tensors["raw_core"][:rows].contiguous()
            runtime = reconstruct_gdn_silu_runtime(
                raw.to(device),
                gate.to(device),
                tensors["norm_weight"].to(device),
                num_value_heads=int(geometry.num_value_heads),
                value_head_dim=int(geometry.value_head_dim),
                rms_norm_eps=float(geometry.rms_norm_eps),
            )
        else:
            raise ValueError(f"unsupported family {family!r}")
        teacher = F.linear(runtime.c_post_gate, weight_device, None)
        if teacher.dtype != torch.bfloat16:
            raise TypeError("checkpoint output projection did not preserve BF16")
        result[split] = {
            "A": gate,
            "B": runtime.h_pre_gate.cpu().contiguous(),
            "C": runtime.c_post_gate.cpu().contiguous(),
            "Y": teacher.cpu().contiguous(),
        }
        del tensors, runtime, teacher
    return result


def _polynomial_records_for_wire(
    *,
    layer: int,
    block_type: str,
    gate_family: GateFamily,
    wire: CanonicalWire,
    snapshots: Mapping[str, Mapping[str, Tensor]],
    radii: Mapping[float, float],
    quantile_metadata: Mapping[str, Any],
    degrees: Sequence[int],
    modes: Sequence[str],
    ridges: Sequence[float],
    chunk_size: int,
    metric_chunk_size: int,
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    maximum_degree = max(degrees)
    variants = variants_for_family(gate_family)
    tuning: list[dict[str, Any]] = []
    candidates: dict[tuple[int, GatePolynomialVariant, bool], list[dict[str, Any]]] = (
        defaultdict(list)
    )
    encoder = wire.encoder.to(device=device)
    for quantile, radius in radii.items():
        for mode in modes:
            clipped = mode == "clipped"
            train_components = build_gate_wire_components(
                snapshots["train"]["A"],
                snapshots["train"]["B"],
                snapshots["train"]["C"],
                encoder,
                radius=radius,
                maximum_degree=maximum_degree,
                family=gate_family,
                clipped=clipped,
                chunk_size=chunk_size,
                device=device,
            )
            dev_components = build_gate_wire_components(
                snapshots["dev"]["A"],
                snapshots["dev"]["B"],
                snapshots["dev"]["C"],
                encoder,
                radius=radius,
                maximum_degree=maximum_degree,
                family=gate_family,
                clipped=clipped,
                chunk_size=chunk_size,
                device=device,
            )
            for degree in degrees:
                for variant in variants:
                    equations = gate_normal_equations_from_components(
                        train_components,
                        degree=degree,
                        variant=variant,
                        chunk_size=metric_chunk_size,
                    )
                    ridge_sweep = []
                    for ridge in ridges:
                        coefficients, solve = solve_gate_ridge_coefficients(
                            equations,
                            relative_lambda=ridge,
                        )
                        ridge_sweep.append(
                            {
                                "relative_lambda": float(ridge),
                                "train_wire_relative_mse": gate_component_wire_relative_mse(
                                    train_components,
                                    coefficients,
                                    degree=degree,
                                    variant=variant,
                                ),
                                "dev_wire_relative_mse": gate_component_wire_relative_mse(
                                    dev_components,
                                    coefficients,
                                    degree=degree,
                                    variant=variant,
                                ),
                                "coefficients": coefficients.tolist(),
                                "solve": solve,
                            }
                        )
                    best_ridge = min(
                        ridge_sweep,
                        key=lambda item: (
                            item["dev_wire_relative_mse"],
                            item["relative_lambda"],
                        ),
                    )
                    candidate = {
                        "layer": int(layer),
                        "block_type": block_type,
                        "gate_family": gate_family,
                        "wire_rank": int(wire.rank),
                        "degree": int(degree),
                        "variant": variant,
                        "mode": mode,
                        "clipped": clipped,
                        "interval_quantile": float(quantile),
                        "interval_radius": float(radius),
                        "relative_lambda": best_ridge["relative_lambda"],
                        "coefficients": best_ridge["coefficients"],
                        "train_wire_relative_mse": best_ridge[
                            "train_wire_relative_mse"
                        ],
                        "dev_wire_relative_mse": best_ridge["dev_wire_relative_mse"],
                        "ridge_sweep": ridge_sweep,
                    }
                    tuning.append(candidate)
                    candidates[(degree, variant, clipped)].append(candidate)
            del train_components, dev_components
    selected = {
        key: min(
            values,
            key=lambda item: (
                item["dev_wire_relative_mse"],
                item["interval_quantile"],
                item["relative_lambda"],
            ),
        )
        for key, values in candidates.items()
    }
    records: dict[tuple[int, GatePolynomialVariant, bool], dict[str, Any]] = {}
    grouped: dict[
        tuple[float, bool],
        list[tuple[tuple[int, GatePolynomialVariant, bool], dict[str, Any]]],
    ] = defaultdict(list)
    for key, design in selected.items():
        grouped[(float(design["interval_quantile"]), bool(design["clipped"]))].append(
            (key, design)
        )
    for split in SPLITS:
        teacher = snapshots[split]["Y"].to(device=device)
        exact_activation = snapshots[split]["C"].to(device=device)
        decoder = wire.decoder.to(device=device)
        teacher_projection = teacher.float() @ decoder
        gate_device = snapshots[split]["A"].to(device=device)
        other_device = snapshots[split]["B"].to(device=device)
        for (quantile, clipped), designs in grouped.items():
            radius = float(radii[quantile])
            components = build_gate_wire_components(
                snapshots[split]["A"],
                snapshots[split]["B"],
                snapshots[split]["C"],
                encoder,
                radius=radius,
                maximum_degree=maximum_degree,
                family=gate_family,
                clipped=clipped,
                chunk_size=chunk_size,
                device=device,
            )
            for key, design in designs:
                degree, variant, _ = key
                coefficients = torch.tensor(design["coefficients"], dtype=torch.float64)
                candidate_latent = gate_latent_from_components(
                    components,
                    coefficients,
                    degree=degree,
                    variant=variant,
                )
                candidate_activation = gate_polynomial_candidate_activation(
                    gate_device,
                    other_device,
                    coefficients,
                    radius=radius,
                    degree=degree,
                    family=gate_family,
                    variant=variant,
                    clipped=clipped,
                )
                metrics = canonical_output_metrics(
                    teacher,
                    components.target,
                    candidate_latent,
                    decoder,
                    teacher_projection=teacher_projection,
                    dense_activation=exact_activation,
                    candidate_activation=candidate_activation,
                    chunk_size=metric_chunk_size,
                )
                approximation = gate_chebyshev_polynomial(
                    gate_device,
                    coefficients,
                    radius=radius,
                    degree=degree,
                    family=gate_family,
                    variant=variant,
                    clipped=clipped,
                )
                scalar = scalar_gate_diagnostics(
                    gate_device,
                    approximation,
                    radius=radius,
                    family=gate_family,
                )
                record = records.setdefault(
                    key,
                    {
                        "format": ROW_FORMAT,
                        "family": "chebyshev",
                        "block_type": block_type,
                        "gate_family": gate_family,
                        "layer": int(layer),
                        "wire_rank": int(wire.rank),
                        "wire_source": wire.source,
                        "method": f"cheb_{variant}_{'clipped' if clipped else 'global'}",
                        "variant": variant,
                        "mode": "clipped" if clipped else "global",
                        "degree": int(degree),
                        "interval_quantile": float(quantile),
                        "interval_radius": radius,
                        "relative_lambda": float(design["relative_lambda"]),
                        "coefficients": design["coefficients"],
                        "parameterized_coefficient_count": len(design["coefficients"]),
                        "estimated_extra_activation_flops_per_coordinate": 3 * degree,
                        "quantile_estimation": dict(quantile_metadata),
                        "selection": {
                            "selected_using": "development_wire_relative_mse",
                            "train_wire_relative_mse": design[
                                "train_wire_relative_mse"
                            ],
                            "dev_wire_relative_mse": design["dev_wire_relative_mse"],
                            "validation_used_for_selection": False,
                        },
                        "splits": {},
                    },
                )
                record["splits"][split] = {**metrics, "scalar": scalar}
                del candidate_latent, candidate_activation, approximation
            del components
        del teacher, exact_activation, decoder, teacher_projection
        del gate_device, other_device
    return list(records.values()), tuning


def _decisions(
    records: Sequence[Mapping[str, Any]],
    *,
    primary_degree: int,
    polynomial_threshold: float,
    topk_threshold: float,
    topk_maximum_keep: float,
) -> list[dict[str, Any]]:
    scopes = sorted(
        {
            (str(row["block_type"]), int(row["layer"]), int(row["wire_rank"]))
            for row in records
        }
    )
    decisions = []
    for block_type, layer, rank in scopes:
        relevant = [
            row
            for row in records
            if row["block_type"] == block_type
            and int(row["layer"]) == layer
            and int(row["wire_rank"]) == rank
        ]
        gate_family = str(relevant[0]["gate_family"])
        constrained_variant = (
            "sigmoid_parity" if gate_family == "sigmoid" else "silu_parity"
        )
        topk_passes = [
            row
            for row in relevant
            if row["family"] == "dynamic_topk"
            and row["scope"] == "tp_local"
            and int(row["block_size"]) == 1
            and float(row["realized_keep_ratio"]) <= topk_maximum_keep
            and float(row["splits"]["validation"]["wire_relative_mse"])
            <= topk_threshold
        ]
        polynomial_passes = [
            row
            for row in relevant
            if row["family"] == "chebyshev"
            and row["mode"] == "global"
            and int(row["degree"]) == primary_degree
            and row["variant"] == constrained_variant
            and float(row["splits"]["validation"]["wire_relative_mse"])
            <= polynomial_threshold
            and 0.5
            <= float(row["splits"]["validation"]["prediction_teacher_energy"])
            <= 1.5
        ]
        unconstrained_passes = [
            row
            for row in relevant
            if row["family"] == "chebyshev"
            and row["mode"] == "global"
            and int(row["degree"]) == primary_degree
            and row["variant"] == "unconstrained"
            and float(row["splits"]["validation"]["wire_relative_mse"])
            <= polynomial_threshold
            and 0.5
            <= float(row["splits"]["validation"]["prediction_teacher_energy"])
            <= 1.5
        ]
        clipped_passes = [
            row
            for row in relevant
            if row["family"] == "chebyshev"
            and row["mode"] == "clipped"
            and int(row["degree"]) == primary_degree
            and row["variant"] == constrained_variant
            and float(row["splits"]["validation"]["wire_relative_mse"])
            <= polynomial_threshold
        ]
        if topk_passes and polynomial_passes:
            decision = "BOTH_PASS"
        elif polynomial_passes:
            decision = "DEGREE8_CONSTRAINED_ONLY"
        elif topk_passes:
            decision = "TOPK_ONLY"
        elif unconstrained_passes:
            decision = "DEGREE8_UNCONSTRAINED_ONLY"
        elif clipped_passes:
            decision = "ONLY_CLIPPED_DEGREE8_PASSES"
        else:
            decision = "NO_PRIMARY_METHOD_PASSES"
        decisions.append(
            {
                "block_type": block_type,
                "gate_family": gate_family,
                "layer": layer,
                "wire_rank": rank,
                "primary_degree": int(primary_degree),
                "decision": decision,
                "topk_pass_count": len(topk_passes),
                "constrained_global_pass": bool(polynomial_passes),
                "unconstrained_global_pass": bool(unconstrained_passes),
                "constrained_clipped_pass": bool(clipped_passes),
            }
        )
    return decisions


def _summary_markdown(
    records: Sequence[Mapping[str, Any]],
    decisions: Sequence[Mapping[str, Any]],
    *,
    primary_degree: int,
) -> str:
    lines = [
        "# Qwen3.5 Activated-Wo Top-k and Chebyshev",
        "",
        (
            "Full attention uses exact BF16 Sigmoid gating; GDN uses exact "
            "headwise RMSNorm followed by FP32 SiLU and BF16 rounding. Global "
            "top-k and clipped polynomials are diagnostics, not deployment claims."
        ),
        "",
        "| Block | Layer | Rank | Output-PCA floor | Best TP-local top-k ≤50% | "
        f"Constrained degree {primary_degree} | Unconstrained degree {primary_degree} | Decision |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for decision in decisions:
        relevant = [
            row
            for row in records
            if row["block_type"] == decision["block_type"]
            and int(row["layer"]) == int(decision["layer"])
            and int(row["wire_rank"]) == int(decision["wire_rank"])
        ]
        keep_one = next(
            row
            for row in relevant
            if row["family"] == "dynamic_topk"
            and row["scope"] == "tp_local"
            and row["score"] == "magnitude"
            and int(row["block_size"]) == 1
            and float(row["requested_keep_ratio"]) == 1.0
        )
        baseline = float(
            keep_one["splits"]["validation"]["baseline_output_relative_mse"]
        )
        topk = [
            row
            for row in relevant
            if row["family"] == "dynamic_topk"
            and row["scope"] == "tp_local"
            and int(row["block_size"]) == 1
            and float(row["realized_keep_ratio"]) <= 0.500001
        ]
        topk_best = min(
            topk,
            key=lambda row: (
                float(row["splits"]["validation"]["wire_relative_mse"]),
                float(row["realized_keep_ratio"]),
            ),
        )
        constrained = (
            "sigmoid_parity" if decision["gate_family"] == "sigmoid" else "silu_parity"
        )
        constrained_row = next(
            row
            for row in relevant
            if row["family"] == "chebyshev"
            and row["mode"] == "global"
            and int(row["degree"]) == primary_degree
            and row["variant"] == constrained
        )
        unconstrained_row = next(
            row
            for row in relevant
            if row["family"] == "chebyshev"
            and row["mode"] == "global"
            and int(row["degree"]) == primary_degree
            and row["variant"] == "unconstrained"
        )
        lines.append(
            f"| {decision['block_type']} | {decision['layer']} | "
            f"{decision['wire_rank']} | {baseline:.6g} | "
            f"keep={float(topk_best['realized_keep_ratio']):.4f}, "
            f"MSE={float(topk_best['splits']['validation']['wire_relative_mse']):.6g} | "
            f"{float(constrained_row['splits']['validation']['wire_relative_mse']):.6g} | "
            f"{float(unconstrained_row['splits']['validation']['wire_relative_mse']):.6g} | "
            f"`{decision['decision']}` |"
        )
    lines.extend(
        [
            "",
            "Lower degrees 2/4/6, both interval quantiles, all ridge choices, "
            "clipped diagnostics, and every split are retained in results.jsonl "
            "and tuning.json.",
            "",
        ]
    )
    return "\n".join(lines)


def _summary_csv(records: Sequence[Mapping[str, Any]]) -> str:
    columns = (
        "block_type",
        "gate_family",
        "family",
        "layer",
        "wire_rank",
        "method",
        "variant",
        "mode",
        "degree",
        "requested_keep_ratio",
        "realized_keep_ratio",
        "scope",
        "score",
        "block_size",
        "interval_quantile",
        "interval_radius",
        "relative_lambda",
        "baseline_output_relative_mse",
        "wire_relative_mse",
        "candidate_incremental_output_relative_mse",
        "total_output_relative_mse",
        "normalized_cross",
        "prediction_teacher_energy",
        "p95_token_error",
    )
    lines = [",".join(columns)]
    for record in records:
        metrics = record["splits"]["validation"]
        values = {
            **record,
            **metrics,
            "p95_token_error": metrics["per_token_total_relative_squared_error"]["p95"],
        }
        lines.append(
            ",".join(
                "" if values.get(column) is None else str(values.get(column, ""))
                for column in columns
            )
        )
    return "\n".join(lines) + "\n"


def _plots(records: Sequence[Mapping[str, Any]], output_dir: Path) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as error:
        print(f"[ActivatedWoPhase1] plot skipped: {error}", flush=True)
        return []
    files: list[str] = []
    scopes = sorted(
        {
            (str(row["block_type"]), int(row["layer"]), int(row["wire_rank"]))
            for row in records
        }
    )
    for block_type, layer, rank in scopes:
        relevant = [
            row
            for row in records
            if row["block_type"] == block_type
            and int(row["layer"]) == layer
            and int(row["wire_rank"]) == rank
        ]
        gate_family = str(relevant[0]["gate_family"])
        constrained = "sigmoid_parity" if gate_family == "sigmoid" else "silu_parity"
        figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        for score in ("magnitude", "decoder_weighted"):
            rows = sorted(
                (
                    row
                    for row in relevant
                    if row["family"] == "dynamic_topk"
                    and row["scope"] == "tp_local"
                    and row["score"] == score
                    and int(row["block_size"]) == 1
                ),
                key=lambda row: float(row["realized_keep_ratio"]),
            )
            axes[0].plot(
                [float(row["realized_keep_ratio"]) for row in rows],
                [
                    float(row["splits"]["validation"]["wire_relative_mse"])
                    for row in rows
                ],
                marker="o",
                label=score,
            )
        for variant in ("unconstrained", constrained):
            rows = sorted(
                (
                    row
                    for row in relevant
                    if row["family"] == "chebyshev"
                    and row["mode"] == "global"
                    and row["variant"] == variant
                ),
                key=lambda row: int(row["degree"]),
            )
            axes[1].plot(
                [int(row["degree"]) for row in rows],
                [
                    float(row["splits"]["validation"]["wire_relative_mse"])
                    for row in rows
                ],
                marker="o",
                label=variant,
            )
        axes[0].set_xlabel("TP-local coordinate keep ratio")
        axes[1].set_xlabel("Chebyshev degree")
        for axis in axes:
            axis.set_ylabel("Validation canonical-wire relative MSE")
            axis.set_yscale("log")
            axis.grid(True, alpha=0.25)
            axis.legend()
        figure.suptitle(f"Qwen3.5 {block_type} layer {layer}, wire rank {rank}")
        figure.tight_layout()
        filename = f"{block_type}_layer_{layer:02d}_rank_{rank}_phase1.png"
        figure.savefig(output_dir / filename, dpi=160)
        plt.close(figure)
        files.append(filename)
    return files


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.torch_num_threads)
    full_layers = tuple(sorted(set(_csv_ints(args.full_layers))))
    gdn_layers = tuple(sorted(set(_csv_ints(args.gdn_layers))))
    ranks = tuple(sorted(set(_csv_ints(args.wire_ranks))))
    ratios = tuple(sorted(set(_csv_floats(args.topk_ratios))))
    scopes = _csv_strings(args.topk_scopes)
    scores = _csv_strings(args.topk_scores)
    block_sizes = tuple(sorted(set(_csv_ints(args.topk_block_sizes))))
    degrees = tuple(sorted(set(_csv_ints(args.cheb_degrees))))
    quantiles = tuple(sorted(set(_csv_floats(args.cheb_interval_quantiles))))
    modes = _csv_strings(args.cheb_modes)
    ridges = tuple(sorted(set(_csv_floats(args.ridge_values))))
    row_targets = {
        "train": int(args.train_rows),
        "dev": int(args.dev_rows),
        "validation": int(args.validation_rows),
    }
    if (
        not full_layers
        or not gdn_layers
        or not ranks
        or min(ranks) <= 0
        or max(ranks) > 4096
        or not ratios
        or any(not 0.0 < value <= 1.0 for value in ratios)
        or set(scopes) - {"global_oracle", "tp_local"}
        or set(scores) - {"magnitude", "decoder_weighted"}
        or not block_sizes
        or min(block_sizes) <= 0
        or not degrees
        or args.primary_degree not in degrees
        or any(degree < 2 for degree in degrees)
        or not quantiles
        or any(not 0.0 < value < 1.0 for value in quantiles)
        or set(modes) != {"global", "clipped"}
        or not ridges
        or min(ridges) < 0.0
        or min(row_targets.values()) <= 0
        or min(args.chunk_size, args.metric_chunk_size) <= 0
        or args.tp <= 0
    ):
        raise ValueError("invalid activated-Wo Phase-1 configuration")
    if any((4096 // args.tp) % block for block in block_sizes):
        raise ValueError("a block size does not divide the TP-local Wo width")

    output_dir = Path(args.output_dir).expanduser().resolve()
    partial_dir = output_dir.with_name(output_dir.name + ".partial")
    if output_dir.exists() or partial_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {output_dir}")
    full_dir = Path(args.full_snapshots).expanduser().resolve()
    gdn_dir = Path(args.gdn_snapshots).expanduser().resolve()
    full_manifest_path = full_dir / "manifest.json"
    gdn_manifest_path = gdn_dir / "manifest.json"
    full_manifest = _load_manifest(full_manifest_path)
    gdn_manifest = _load_manifest(gdn_manifest_path)
    full_geometry, full_records, full_provenance = validate_full_manifest(
        full_manifest,
        manifest_path=full_manifest_path,
    )
    gdn_geometry, gdn_records, gdn_provenance = validate_gdn_manifest(
        gdn_manifest,
        manifest_path=gdn_manifest_path,
    )
    if sorted(set(full_layers) - set(full_records)):
        raise ValueError("a requested full-attention layer is absent from snapshots")
    if sorted(set(gdn_layers) - set(gdn_records)):
        raise ValueError("a requested GDN layer is absent from snapshots")
    aligned_provenance = _validate_aligned_provenance(full_manifest, gdn_manifest)
    model_path = (
        Path(args.model_path or str(full_manifest["model"])).expanduser().resolve()
    )
    full_model, full_weight_map = validate_full_model(
        full_manifest,
        model_path,
        full_geometry,
    )
    gdn_model, gdn_weight_map = validate_gdn_model(
        gdn_manifest,
        model_path,
        gdn_geometry,
    )
    for key in (
        "model_id",
        "model_revision",
        "resolved_model_path",
        "config_sha256",
        "safetensors_index_sha256",
    ):
        if full_model[key] != gdn_model[key]:
            raise ValueError(f"full-attention/GDN model metadata differs at {key}")
    validated_full = _validate_snapshot_bank(
        full_dir,
        full_manifest,
        layers=full_layers,
        family="full_attention",
        row_targets=row_targets,
    )
    validated_gdn = _validate_snapshot_bank(
        gdn_dir,
        gdn_manifest,
        layers=gdn_layers,
        family="gdn",
        row_targets=row_targets,
    )

    partial_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    cuda_device_index = device.index if device.index is not None else 0
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA is unavailable for requested device {device}")
        torch.cuda.set_device(cuda_device_index)
        torch.cuda.init()
    timestamp_started = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    all_records: list[dict[str, Any]] = []
    all_tuning: list[dict[str, Any]] = []
    layer_metadata: dict[str, Any] = {}
    wire_tensors: dict[str, Tensor] = {}
    peak_cuda = 0
    layer_specs = [
        (layer, "full_attention", "sigmoid", full_geometry, validated_full[layer])
        for layer in full_layers
    ] + [
        (layer, "gdn", "silu", gdn_geometry, validated_gdn[layer])
        for layer in gdn_layers
    ]
    layer_specs.sort(key=lambda item: item[0])
    with torch.inference_mode():
        for layer, block_type, gate_family, geometry, validated in layer_specs:
            layer_started = time.perf_counter()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(cuda_device_index)
            print(
                f"[ActivatedWoPhase1] layer={layer} block={block_type} phase=load",
                flush=True,
            )
            if block_type == "full_attention":
                weight, weight_source = load_full_o_proj(
                    model_path,
                    full_weight_map,
                    layer,
                    geometry,
                )
            else:
                weight, weight_source = load_gdn_o_proj(
                    model_path,
                    gdn_weight_map,
                    layer,
                    geometry,
                )
            snapshots = _prepare_layer_snapshots(
                validated,
                family=block_type,
                geometry=geometry,
                weight=weight,
                device=device,
            )
            wires, factor_tensors, factor_metadata = _wire_bank(
                snapshots["train"]["Y"],
                weight,
                ranks,
                device=device,
                oversample=args.pod_oversample,
                niter=args.pod_niter,
                seed=args.seed + 7919 * layer,
            )
            for name, value in factor_tensors.items():
                wire_tensors[f"layer_{layer:02d}_{name}"] = value
            radii, quantile_metadata = _estimate_radii(
                snapshots["train"]["A"],
                quantiles,
                maximum_elements=args.quantile_max_elements,
                seed=args.seed + 104729 * layer,
            )
            print(
                f"[ActivatedWoPhase1] layer={layer} block={block_type} phase=topk",
                flush=True,
            )
            topk = _topk_records_for_layer(
                layer_index=layer,
                snapshots=snapshots,
                wires=wires,
                ratios=ratios,
                scopes=scopes,
                scores=scores,
                block_sizes=block_sizes,
                tp_size=args.tp,
                device=device,
                metric_chunk_size=args.metric_chunk_size,
            )
            for row in topk:
                row["format"] = ROW_FORMAT
                row["block_type"] = block_type
                row["gate_family"] = gate_family
            all_records.extend(topk)
            print(
                f"[ActivatedWoPhase1] layer={layer} block={block_type} phase=chebyshev",
                flush=True,
            )
            polynomial_count = 0
            for _, wire in wires.items():
                polynomial, tuning = _polynomial_records_for_wire(
                    layer=layer,
                    block_type=block_type,
                    gate_family=gate_family,
                    wire=wire,
                    snapshots=snapshots,
                    radii=radii,
                    quantile_metadata=quantile_metadata,
                    degrees=degrees,
                    modes=modes,
                    ridges=ridges,
                    chunk_size=args.chunk_size,
                    metric_chunk_size=args.metric_chunk_size,
                    device=device,
                )
                all_records.extend(polynomial)
                all_tuning.extend(tuning)
                polynomial_count += len(polynomial)
            layer_peak = (
                int(torch.cuda.max_memory_allocated(cuda_device_index))
                if device.type == "cuda"
                else 0
            )
            peak_cuda = max(peak_cuda, layer_peak)
            key = f"{block_type}:layer_{layer:02d}"
            layer_metadata[key] = {
                "block_type": block_type,
                "gate_family": gate_family,
                "elapsed_seconds": time.perf_counter() - layer_started,
                "peak_cuda_allocated_bytes": layer_peak,
                "weight": weight_source,
                "wire": factor_metadata,
                "radii": {str(name): value for name, value in radii.items()},
                "quantile_estimation": quantile_metadata,
                "topk_rows": len(topk),
                "polynomial_rows": polynomial_count,
                "snapshot_files": validated,
            }
            del snapshots, weight, wires
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(
                f"[ActivatedWoPhase1] layer={layer} block={block_type} done "
                f"seconds={layer_metadata[key]['elapsed_seconds']:.3f}",
                flush=True,
            )

    decisions = _decisions(
        all_records,
        primary_degree=args.primary_degree,
        polynomial_threshold=args.polynomial_pass_threshold,
        topk_threshold=args.topk_pass_threshold,
        topk_maximum_keep=args.topk_pass_maximum_keep,
    )
    save_file(wire_tensors, partial_dir / "wire_bank.safetensors")
    _atomic_text(
        partial_dir / "results.jsonl",
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in all_records),
    )
    _atomic_json(partial_dir / "tuning.json", all_tuning)
    _atomic_text(partial_dir / "summary.csv", _summary_csv(all_records))
    _atomic_text(
        partial_dir / "summary.md",
        _summary_markdown(
            all_records,
            decisions,
            primary_degree=args.primary_degree,
        ),
    )
    plot_files = _plots(all_records, partial_dir)
    elapsed = time.perf_counter() - started
    manifest = {
        "format": FORMAT,
        "schema_version": 1,
        "command": shlex.join(sys.argv),
        "git_commit": _git_commit(),
        "timestamp_started_utc": timestamp_started,
        "timestamp_finished_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed,
        "model": full_model,
        "snapshot_provenance": {
            "full_attention": full_provenance,
            "gdn": gdn_provenance,
            "aligned": aligned_provenance,
            "row_targets": row_targets,
            "row_selection": "deterministic_prefix",
        },
        "layers": {
            "full_attention": list(full_layers),
            "gdn": list(gdn_layers),
        },
        "wire_ranks": list(ranks),
        "topk": {
            "ratios": list(ratios),
            "scopes": list(scopes),
            "scores": list(scores),
            "block_sizes": list(block_sizes),
            "tp_size": args.tp,
            "pass_threshold": args.topk_pass_threshold,
            "pass_maximum_keep": args.topk_pass_maximum_keep,
        },
        "chebyshev": {
            "degrees": list(degrees),
            "primary_degree": args.primary_degree,
            "full_attention_variants": list(variants_for_family("sigmoid")),
            "gdn_variants": list(variants_for_family("silu")),
            "interval_quantiles": list(quantiles),
            "modes": list(modes),
            "ridge_values": list(ridges),
            "pass_threshold": args.polynomial_pass_threshold,
            "validation_used_for_hyperparameter_selection": False,
            "gdn_rmsnorm_approximated": False,
        },
        "decisions": decisions,
        "rows": len(all_records),
        "tuning_rows": len(all_tuning),
        "layer_metadata": layer_metadata,
        "wire_bank": "wire_bank.safetensors",
        "plot_files": plot_files,
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": _installed_version("transformers"),
            "safetensors": _installed_version("safetensors"),
            "device": str(device),
            "cuda_device_name": (
                torch.cuda.get_device_name(cuda_device_index)
                if device.type == "cuda"
                else None
            ),
            "torch_num_threads": torch.get_num_threads(),
            "peak_cuda_allocated_bytes": peak_cuda,
        },
    }
    _atomic_json(partial_dir / "manifest.json", manifest)
    os.replace(partial_dir, output_dir)
    print(
        f"[ActivatedWoPhase1] complete rows={len(all_records)} "
        f"output={output_dir} seconds={elapsed:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
