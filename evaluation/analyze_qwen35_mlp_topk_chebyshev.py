#!/usr/bin/env python3
"""Run the corrected Qwen3.5 MLP top-k/Chebyshev Phase-1 diagnostic."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor
from safetensors import safe_open
from safetensors.torch import save_file


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basisserve.analysis.mlp_topk_chebyshev import (  # noqa: E402
    CanonicalWire,
    build_chebyshev_wire_components,
    canonical_output_metrics,
    canonical_wire_from_output_basis,
    chebyshev_polynomial,
    component_wire_relative_mse,
    decoder_aware_channel_weights,
    dynamic_topk,
    exact_output_wire,
    latent_from_components,
    normal_equations_from_components,
    polar_retract_columns,
    polynomial_candidate_activation,
    scalar_silu_diagnostics,
    solve_ridge_coefficients,
)
from basisserve.calibration.mlp_polynomial_snapshots import (  # noqa: E402
    SNAPSHOT_FORMAT,
    SNAPSHOT_SIGNALS,
)
from basisserve.calibration.mlp_snapshots import SPLITS  # noqa: E402
from basisserve.sketching.coordinate_selection import (  # noqa: E402
    compute_uncentered_pod,
)


FORMAT = "basisserve.qwen35.mlp_topk_chebyshev_phase1.v1"
ROW_FORMAT = "basisserve.qwen35.mlp_topk_chebyshev_phase1.row.v1"
DEFAULT_TOPK_RATIOS = (0.1, 0.2, 0.3, 0.5, 0.75, 1.0)
DEFAULT_BLOCK_SIZES = (1, 8, 16, 32)
DEFAULT_DEGREES = (2, 4, 6, 8)
DEFAULT_INTERVAL_QUANTILES = (0.999, 0.9999)
DEFAULT_RIDGES = (0.0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-path")
    parser.add_argument("--layers", default="representative")
    parser.add_argument("--wire-ranks", default="1536,2048,4096")
    parser.add_argument(
        "--topk-ratios", default=",".join(map(str, DEFAULT_TOPK_RATIOS))
    )
    parser.add_argument("--topk-scopes", default="global_oracle,tp_local")
    parser.add_argument("--topk-scores", default="magnitude,decoder_weighted")
    parser.add_argument(
        "--topk-block-sizes", default=",".join(map(str, DEFAULT_BLOCK_SIZES))
    )
    parser.add_argument("--cheb-degrees", default=",".join(map(str, DEFAULT_DEGREES)))
    parser.add_argument(
        "--cheb-interval-quantiles",
        default=",".join(map(str, DEFAULT_INTERVAL_QUANTILES)),
    )
    parser.add_argument("--cheb-variants", default="unconstrained,silu_parity")
    parser.add_argument("--cheb-modes", default="global,clipped")
    parser.add_argument("--ridge-values", default=",".join(map(str, DEFAULT_RIDGES)))
    parser.add_argument("--tp", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--metric-chunk-size", type=int, default=256)
    parser.add_argument("--quantile-max-elements", type=int, default=4_000_000)
    parser.add_argument("--pod-oversample", type=int, default=16)
    parser.add_argument("--pod-niter", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--polynomial-pass-threshold", type=float, default=0.01)
    parser.add_argument("--topk-pass-threshold", type=float, default=0.01)
    parser.add_argument("--topk-pass-maximum-keep", type=float, default=0.5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--torch-num-threads", type=int, default=4)
    return parser.parse_args()


def _csv_strings(raw: str) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(piece.strip() for piece in raw.split(",") if piece.strip())
    )


def _csv_ints(raw: str) -> tuple[int, ...]:
    return tuple(
        dict.fromkeys(int(piece.strip()) for piece in raw.split(",") if piece.strip())
    )


def _csv_floats(raw: str) -> tuple[float, ...]:
    return tuple(
        dict.fromkeys(float(piece.strip()) for piece in raw.split(",") if piece.strip())
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _installed_version(name: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None


def _atomic_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: Any) -> None:
    _atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _parse_layers(
    raw: str,
    *,
    available: set[int],
    layer_count: int,
) -> tuple[int, ...]:
    normalized = raw.strip().lower()
    if normalized == "representative":
        selected = (0, layer_count // 2, layer_count - 1)
    elif normalized == "all":
        selected = tuple(sorted(available))
    else:
        selected = tuple(sorted(set(_csv_ints(raw))))
    missing = sorted(set(selected) - available)
    if not selected or missing:
        raise ValueError(f"invalid layer selection; missing snapshots={missing}")
    return selected


def _load_manifest(
    snapshot_dir: Path,
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    manifest_path = snapshot_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("format") != SNAPSHOT_FORMAT
        or int(manifest.get("schema_version", -1)) != 1
    ):
        raise ValueError("unsupported MLP Phase-1 snapshot manifest")
    if manifest.get("signals") != list(SNAPSHOT_SIGNALS):
        raise ValueError("MLP Phase-1 snapshot signals differ from the exact schema")
    if manifest.get("storage_dtype") != "bfloat16":
        raise ValueError("Phase-1 snapshots must use BF16 storage")
    if not bool(manifest["collection"].get("document_split_isolation")) or not bool(
        manifest["document_sampling"].get("document_split_isolation")
    ):
        raise ValueError("snapshot manifest does not prove document-level isolation")
    layers = {int(record["layer_index"]): record for record in manifest["layers"]}
    if set(map(int, manifest["selected_layers"])) != set(layers):
        raise ValueError("selected layer list differs from snapshot records")
    return manifest, layers


def _validate_snapshot_files(
    snapshot_dir: Path,
    layers: Mapping[int, Mapping[str, Any]],
    selected: Sequence[int],
) -> dict[int, dict[str, dict[str, Any]]]:
    validated: dict[int, dict[str, dict[str, Any]]] = {}
    for layer_index in selected:
        record = layers[layer_index]
        hidden = int(record["hidden_size"])
        intermediate = int(record["intermediate_size"])
        validated[layer_index] = {}
        for split in SPLITS:
            diagnostics = record["diagnostics"][split]
            if (
                int(diagnostics["calls"]) <= 0
                or float(diagnostics["post_swiglu_relative_mse"]) > 2e-5
                or float(diagnostics["direct_down_relative_mse"]) > 2e-5
            ):
                raise ValueError(
                    f"layer {layer_index} {split} failed exact-capture diagnostics"
                )
            source_record = record["splits"][split]
            component = Path(str(source_record["file"]))
            if component.is_absolute() or ".." in component.parts:
                raise ValueError("unsafe snapshot component path")
            path = snapshot_dir / component
            observed_hash = _file_sha256(path)
            if observed_hash != str(source_record["file_sha256"]):
                raise ValueError(f"snapshot hash mismatch: {path}")
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                if set(handle.keys()) != set(SNAPSHOT_SIGNALS):
                    raise ValueError(f"unexpected tensors in {path}")
                shapes = {
                    name: tuple(handle.get_slice(name).get_shape())
                    for name in SNAPSHOT_SIGNALS
                }
                dtypes = {
                    name: handle.get_slice(name).get_dtype()
                    for name in SNAPSHOT_SIGNALS
                }
            rows = int(source_record["rows"])
            expected = {
                "X": (rows, hidden),
                "A": (rows, intermediate),
                "B": (rows, intermediate),
                "C": (rows, intermediate),
                "Y": (rows, hidden),
            }
            if shapes != expected or set(dtypes.values()) != {"BF16"}:
                raise ValueError(f"snapshot tensor schema mismatch: {path}")
            validated[layer_index][split] = {
                "path": str(path),
                "file": str(component),
                "file_sha256": observed_hash,
                "rows": rows,
                "diagnostics": diagnostics,
            }
    return validated


def _load_tensors(record: Mapping[str, Any], names: Sequence[str]) -> dict[str, Tensor]:
    path = str(record["path"])
    with safe_open(path, framework="pt", device="cpu") as handle:
        return {name: handle.get_tensor(name) for name in names}


def _model_metadata(
    manifest: Mapping[str, Any],
    requested_model_path: str | None,
) -> tuple[Path, dict[str, Any], dict[str, str], dict[str, Any]]:
    model_path = (
        Path(requested_model_path or str(manifest["model"])).expanduser().resolve()
    )
    snapshot_model = Path(str(manifest["model"])).expanduser().resolve()
    if model_path != snapshot_model:
        raise ValueError("model path differs from the snapshot checkpoint")
    config_path = model_path / "config.json"
    index_path = model_path / "model.safetensors.index.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if config.get("model_type") != "qwen3_5":
        raise ValueError("checkpoint is not Qwen3.5")
    weight_map = dict(index["weight_map"])
    text_config = config.get("text_config", config)
    hidden = int(text_config["hidden_size"])
    intermediate = int(text_config["intermediate_size"])
    layer_count = int(text_config["num_hidden_layers"])
    return (
        model_path,
        config,
        weight_map,
        {
            "model_path": str(model_path),
            "config_sha256": _file_sha256(config_path),
            "safetensors_index_sha256": _file_sha256(index_path),
            "hidden_size": hidden,
            "intermediate_size": intermediate,
            "num_hidden_layers": layer_count,
            "model_id": str(config.get("_name_or_path", "Qwen/Qwen3.5-9B-Base")),
            "model_revision": model_path.name,
        },
    )


def _load_down_weight(
    model_path: Path,
    weight_map: Mapping[str, str],
    layer_index: int,
    *,
    expected_shape: tuple[int, int],
    shard_hashes: dict[str, str],
) -> tuple[Tensor, dict[str, Any]]:
    tensor_name = f"model.language_model.layers.{layer_index}.mlp.down_proj.weight"
    shard = weight_map.get(tensor_name)
    if shard is None:
        raise ValueError(
            f"checkpoint has no main-language-model layer {layer_index} MLP down projection"
        )
    bias_name = f"model.language_model.layers.{layer_index}.mlp.down_proj.bias"
    if bias_name in weight_map:
        raise ValueError("Phase-1 compiler currently requires bias-free down_proj")
    component = Path(shard)
    if component.is_absolute() or ".." in component.parts:
        raise ValueError("unsafe checkpoint shard path")
    shard_path = model_path / component
    if shard not in shard_hashes:
        shard_hashes[shard] = _file_sha256(shard_path)
    with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
        weight = handle.get_tensor(tensor_name)
    if tuple(weight.shape) != expected_shape or weight.dtype != torch.bfloat16:
        raise ValueError(
            f"down_proj has {weight.dtype} {tuple(weight.shape)}, expected BF16 {expected_shape}"
        )
    return weight.contiguous(), {
        "tensor_name": tensor_name,
        "shard": shard,
        "shard_sha256": shard_hashes[shard],
    }


def _estimate_radii(
    gate: Tensor,
    quantiles: Sequence[float],
    *,
    maximum_elements: int,
    seed: int,
) -> tuple[dict[float, float], dict[str, Any]]:
    if maximum_elements <= 0:
        raise ValueError("quantile sample size must be positive")
    flat = gate.flatten()
    population = int(flat.numel())
    sample_size = min(population, maximum_elements)
    if sample_size == population:
        sample = flat.float()
        method = "exact_all_elements"
    else:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        indices = torch.randint(
            population,
            (sample_size,),
            generator=generator,
            dtype=torch.int64,
        )
        sample = flat.index_select(0, indices).float()
        method = "deterministic_uniform_with_replacement"
    absolute = sample.abs()
    radii = {float(q): float(torch.quantile(absolute, float(q))) for q in quantiles}
    if any(not math.isfinite(value) or value <= 0.0 for value in radii.values()):
        raise ValueError("estimated Chebyshev radius is invalid")
    return radii, {
        "method": method,
        "seed": int(seed),
        "population_elements": population,
        "sampled_elements": sample_size,
    }


def _wire_bank(
    train_output: Tensor,
    down_weight: Tensor,
    ranks: Sequence[int],
    *,
    device: torch.device,
    oversample: int,
    niter: int,
    seed: int,
) -> tuple[dict[int, CanonicalWire], dict[str, Tensor], dict[str, Any]]:
    hidden = int(down_weight.shape[0])
    if not ranks or min(ranks) <= 0 or max(ranks) > hidden:
        raise ValueError(f"wire ranks must lie in [1,{hidden}]")
    low_ranks = tuple(rank for rank in sorted(set(ranks)) if rank < hidden)
    wires: dict[int, CanonicalWire] = {}
    tensors: dict[str, Tensor] = {}
    metadata: dict[str, Any] = {}
    weight_device = down_weight.to(device=device, dtype=torch.float32)
    if low_ranks:
        maximum = max(low_ranks)
        pod = compute_uncentered_pod(
            train_output.to(device),
            maximum,
            oversample=oversample,
            niter=niter,
            seed=seed,
        )
        basis, retraction_metadata = polar_retract_columns(pod.basis)
        maximum_wire = canonical_wire_from_output_basis(
            weight_device,
            basis,
            source="train_output_pca",
        )
        tensors["maximum_output_basis"] = maximum_wire.decoder.cpu().to(torch.bfloat16)
        tensors["maximum_encoder"] = maximum_wire.encoder.cpu().to(torch.bfloat16)
        metadata["output_pca"] = {
            **pod.diagnostics(),
            "polar_retraction": retraction_metadata,
        }
        for rank in low_ranks:
            wires[rank] = canonical_wire_from_output_basis(
                weight_device,
                maximum_wire.decoder[:, :rank],
                source="train_output_pca_nested",
            )
    if hidden in ranks:
        wires[hidden] = exact_output_wire(weight_device)
    return wires, tensors, metadata


def _metric_inputs(
    snapshot: Mapping[str, Tensor],
    wire: CanonicalWire,
    candidate_activation: Tensor,
    *,
    device: torch.device,
    metric_chunk_size: int,
) -> dict[str, Any]:
    exact_activation = snapshot["C"].to(device=device)
    teacher = snapshot["Y"].to(device=device)
    encoder = wire.encoder.to(device=device)
    dense_latent = exact_activation.float() @ encoder
    candidate_latent = candidate_activation.float() @ encoder
    metrics = canonical_output_metrics(
        teacher,
        dense_latent,
        candidate_latent,
        wire.decoder.to(device=device),
        teacher_projection=teacher.float() @ wire.decoder.to(device=device),
        dense_activation=exact_activation,
        candidate_activation=candidate_activation,
        chunk_size=metric_chunk_size,
    )
    del exact_activation, teacher, dense_latent, candidate_latent
    return metrics


def _topk_records_for_layer(
    *,
    layer_index: int,
    snapshots: Mapping[str, Mapping[str, Tensor]],
    wires: Mapping[int, CanonicalWire],
    ratios: Sequence[float],
    scopes: Sequence[str],
    scores: Sequence[str],
    block_sizes: Sequence[int],
    tp_size: int,
    device: torch.device,
    metric_chunk_size: int,
) -> list[dict[str, Any]]:
    records: dict[tuple[Any, ...], dict[str, Any]] = {}
    for rank, wire in wires.items():
        encoder = wire.encoder.to(device=device)
        decoder = wire.decoder.to(device=device)
        channel_weights = decoder_aware_channel_weights(wire).to(device=device)
        for split in SPLITS:
            exact = snapshots[split]["C"].to(device=device)
            teacher = snapshots[split]["Y"].to(device=device)
            dense_latent = exact.float() @ encoder
            teacher_projection = teacher.float() @ decoder
            for scope in scopes:
                for score in scores:
                    weights = channel_weights if score == "decoder_weighted" else None
                    for block_size in block_sizes:
                        for ratio in ratios:
                            sparse, diagnostics = dynamic_topk(
                                exact,
                                ratio,
                                scope=scope,  # type: ignore[arg-type]
                                tp_size=tp_size,
                                block_size=block_size,
                                score=score,  # type: ignore[arg-type]
                                channel_weights=weights,
                            )
                            candidate_latent = sparse.float() @ encoder
                            metrics = canonical_output_metrics(
                                teacher,
                                dense_latent,
                                candidate_latent,
                                decoder,
                                teacher_projection=teacher_projection,
                                dense_activation=exact,
                                candidate_activation=sparse,
                                chunk_size=metric_chunk_size,
                            )
                            if ratio == 1.0 and metrics["wire_relative_mse"] > 1e-12:
                                raise RuntimeError("top-k keep=1 endpoint is not exact")
                            key = (rank, scope, score, block_size, float(ratio))
                            record = records.setdefault(
                                key,
                                {
                                    "format": ROW_FORMAT,
                                    "family": "dynamic_topk",
                                    "layer": int(layer_index),
                                    "wire_rank": int(rank),
                                    "wire_source": wire.source,
                                    "method": f"{scope}_{score}_block{block_size}",
                                    "scope": scope,
                                    "score": score,
                                    "block_size": int(block_size),
                                    "requested_keep_ratio": float(ratio),
                                    "realized_keep_ratio": diagnostics.realized_keep_ratio,
                                    "tp_size": int(tp_size),
                                    "estimated_local_wire_mac_fraction": diagnostics.realized_keep_ratio,
                                    "parameterized_coefficient_count": 0,
                                    "splits": {},
                                },
                            )
                            record["splits"][split] = {
                                **metrics,
                                "selection": diagnostics.__dict__,
                            }
                            del sparse, candidate_latent
            del exact, teacher, dense_latent, teacher_projection
    return list(records.values())


def _polynomial_records_for_wire(
    *,
    layer_index: int,
    wire: CanonicalWire,
    snapshots: Mapping[str, Mapping[str, Tensor]],
    radii: Mapping[float, float],
    quantile_metadata: Mapping[str, Any],
    degrees: Sequence[int],
    variants: Sequence[str],
    modes: Sequence[str],
    ridges: Sequence[float],
    chunk_size: int,
    metric_chunk_size: int,
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    maximum_degree = max(degrees)
    tuning: list[dict[str, Any]] = []
    candidates: dict[tuple[int, str, bool], list[dict[str, Any]]] = defaultdict(list)
    encoder = wire.encoder.to(device=device)
    for quantile, radius in radii.items():
        for mode in modes:
            clipped = mode == "clipped"
            train_components = build_chebyshev_wire_components(
                snapshots["train"]["A"],
                snapshots["train"]["B"],
                snapshots["train"]["C"],
                encoder,
                radius=radius,
                maximum_degree=maximum_degree,
                clipped=clipped,
                chunk_size=chunk_size,
                device=device,
            )
            dev_components = build_chebyshev_wire_components(
                snapshots["dev"]["A"],
                snapshots["dev"]["B"],
                snapshots["dev"]["C"],
                encoder,
                radius=radius,
                maximum_degree=maximum_degree,
                clipped=clipped,
                chunk_size=chunk_size,
                device=device,
            )
            for degree in degrees:
                for variant in variants:
                    equations = normal_equations_from_components(
                        train_components,
                        degree=degree,
                        variant=variant,  # type: ignore[arg-type]
                        chunk_size=metric_chunk_size,
                    )
                    ridge_sweep: list[dict[str, Any]] = []
                    for ridge in ridges:
                        coefficients, solve = solve_ridge_coefficients(
                            equations,
                            relative_lambda=ridge,
                        )
                        train_error = component_wire_relative_mse(
                            train_components,
                            coefficients,
                            degree=degree,
                            variant=variant,  # type: ignore[arg-type]
                        )
                        dev_error = component_wire_relative_mse(
                            dev_components,
                            coefficients,
                            degree=degree,
                            variant=variant,  # type: ignore[arg-type]
                        )
                        ridge_sweep.append(
                            {
                                "relative_lambda": float(ridge),
                                "train_wire_relative_mse": train_error,
                                "dev_wire_relative_mse": dev_error,
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
                        "layer": int(layer_index),
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
    records: dict[tuple[int, str, bool], dict[str, Any]] = {}
    grouped: dict[
        tuple[float, bool], list[tuple[tuple[int, str, bool], dict[str, Any]]]
    ] = defaultdict(list)
    for key, design in selected.items():
        grouped[(float(design["interval_quantile"]), bool(design["clipped"]))].append(
            (key, design)
        )
    for split in SPLITS:
        teacher = snapshots[split]["Y"].to(device=device)
        exact_activation = snapshots[split]["C"].to(device=device)
        teacher_projection = teacher.float() @ wire.decoder.to(device=device)
        for (quantile, clipped), designs in grouped.items():
            radius = float(radii[quantile])
            components = build_chebyshev_wire_components(
                snapshots[split]["A"],
                snapshots[split]["B"],
                snapshots[split]["C"],
                encoder,
                radius=radius,
                maximum_degree=maximum_degree,
                clipped=clipped,
                chunk_size=chunk_size,
                device=device,
            )
            for key, design in designs:
                degree, variant, _ = key
                coefficients = torch.tensor(design["coefficients"], dtype=torch.float64)
                candidate_latent = latent_from_components(
                    components,
                    coefficients,
                    degree=degree,
                    variant=variant,  # type: ignore[arg-type]
                )
                candidate_activation = polynomial_candidate_activation(
                    snapshots[split]["A"].to(device=device),
                    snapshots[split]["B"].to(device=device),
                    coefficients,
                    radius=radius,
                    degree=degree,
                    variant=variant,  # type: ignore[arg-type]
                    clipped=clipped,
                )
                metrics = canonical_output_metrics(
                    teacher,
                    components.target,
                    candidate_latent,
                    wire.decoder.to(device=device),
                    teacher_projection=teacher_projection,
                    dense_activation=exact_activation,
                    candidate_activation=candidate_activation,
                    chunk_size=metric_chunk_size,
                )
                approximation = chebyshev_polynomial(
                    snapshots[split]["A"].to(device=device),
                    coefficients,
                    radius=radius,
                    degree=degree,
                    variant=variant,  # type: ignore[arg-type]
                    clipped=clipped,
                )
                scalar = scalar_silu_diagnostics(
                    snapshots[split]["A"].to(device=device),
                    approximation,
                    radius=radius,
                )
                record = records.setdefault(
                    key,
                    {
                        "format": ROW_FORMAT,
                        "family": "chebyshev",
                        "layer": int(layer_index),
                        "wire_rank": int(wire.rank),
                        "wire_source": wire.source,
                        "method": (
                            f"cheb_{variant}_{'clipped' if clipped else 'global'}"
                        ),
                        "variant": variant,
                        "mode": "clipped" if clipped else "global",
                        "degree": int(degree),
                        "interval_quantile": float(quantile),
                        "interval_radius": radius,
                        "relative_lambda": float(design["relative_lambda"]),
                        "coefficients": design["coefficients"],
                        "parameterized_coefficient_count": len(design["coefficients"]),
                        "estimated_extra_activation_flops_per_coordinate": (3 * degree),
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
        del teacher, exact_activation, teacher_projection
    return list(records.values()), tuning


def _summary_csv(records: Sequence[Mapping[str, Any]]) -> str:
    columns = (
        "family",
        "layer",
        "wire_rank",
        "method",
        "degree",
        "requested_keep_ratio",
        "realized_keep_ratio",
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


def _decisions(
    records: Sequence[Mapping[str, Any]],
    *,
    polynomial_threshold: float,
    topk_threshold: float,
    topk_maximum_keep: float,
) -> list[dict[str, Any]]:
    scopes = sorted({(int(row["layer"]), int(row["wire_rank"])) for row in records})
    decisions = []
    for layer, rank in scopes:
        relevant = [
            row
            for row in records
            if int(row["layer"]) == layer and int(row["wire_rank"]) == rank
        ]
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
            and int(row["degree"]) <= 6
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
            and int(row["degree"]) <= 6
            and float(row["splits"]["validation"]["wire_relative_mse"])
            <= polynomial_threshold
        ]
        if topk_passes and polynomial_passes:
            decision = "BOTH_PASS"
        elif topk_passes:
            decision = "TOPK_ONLY"
        elif polynomial_passes:
            decision = "CHEBYSHEV_ONLY_PASSES"
        elif clipped_passes:
            decision = "ONLY_CLIPPED_POLY_PASSES"
        else:
            decision = "NEITHER_PASSES_CURRENT_PARAMETERIZATION"
        decisions.append(
            {
                "layer": layer,
                "wire_rank": rank,
                "decision": decision,
                "topk_pass_count": len(topk_passes),
                "global_polynomial_pass_count": len(polynomial_passes),
                "clipped_polynomial_pass_count": len(clipped_passes),
            }
        )
    return decisions


def _summary_markdown(
    records: Sequence[Mapping[str, Any]],
    decisions: Sequence[Mapping[str, Any]],
) -> str:
    lines = [
        "# Qwen3.5 MLP Top-k and Chebyshev Phase 1",
        "",
        (
            "All wire metrics use an orthonormal output decoder. Final total error "
            "is measured against the exact dense MLP output. Global top-k and "
            "clipped polynomials are diagnostics, not deployable folding claims."
        ),
        "",
    ]
    for decision in decisions:
        layer, rank = int(decision["layer"]), int(decision["wire_rank"])
        relevant = [
            row
            for row in records
            if int(row["layer"]) == layer and int(row["wire_rank"]) == rank
        ]
        topk = sorted(
            (
                row
                for row in relevant
                if row["family"] == "dynamic_topk"
                and row["scope"] == "tp_local"
                and row["score"] == "decoder_weighted"
                and int(row["block_size"]) == 1
            ),
            key=lambda row: float(row["realized_keep_ratio"]),
        )
        polynomial = sorted(
            (row for row in relevant if row["family"] == "chebyshev"),
            key=lambda row: (
                row["mode"] != "global",
                int(row["degree"]),
                row["variant"],
            ),
        )
        lines.extend(
            [
                f"## Layer {layer}, wire rank {rank}",
                "",
                f"Decision: `{decision['decision']}`.",
                "",
                "| Method | Budget | Validation wire MSE | Incremental output MSE | Total output MSE |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for row in (*topk, *polynomial):
            metrics = row["splits"]["validation"]
            budget = (
                f"keep={float(row['realized_keep_ratio']):.4f}"
                if row["family"] == "dynamic_topk"
                else f"degree={int(row['degree'])}"
            )
            lines.append(
                f"| {row['method']} | {budget} | "
                f"{float(metrics['wire_relative_mse']):.6g} | "
                f"{float(metrics['candidate_incremental_output_relative_mse']):.6g} | "
                f"{float(metrics['total_output_relative_mse']):.6g} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def _plots(records: Sequence[Mapping[str, Any]], output_dir: Path) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as error:
        print(f"[MLPPhase1] plot skipped: {error}", flush=True)
        return []
    files: list[str] = []
    scopes = sorted({(int(row["layer"]), int(row["wire_rank"])) for row in records})
    for layer, rank in scopes:
        relevant = [
            row
            for row in records
            if int(row["layer"]) == layer and int(row["wire_rank"]) == rank
        ]
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
        for variant in ("unconstrained", "silu_parity"):
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
        figure.suptitle(f"Qwen3.5 MLP layer {layer}, wire rank {rank}")
        figure.tight_layout()
        filename = f"layer_{layer:02d}_rank_{rank}_phase1.png"
        figure.savefig(output_dir / filename, dpi=160)
        plt.close(figure)
        files.append(filename)
    return files


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.torch_num_threads)
    ranks = tuple(sorted(set(_csv_ints(args.wire_ranks))))
    ratios = tuple(sorted(set(_csv_floats(args.topk_ratios))))
    scopes = _csv_strings(args.topk_scopes)
    scores = _csv_strings(args.topk_scores)
    block_sizes = tuple(sorted(set(_csv_ints(args.topk_block_sizes))))
    degrees = tuple(sorted(set(_csv_ints(args.cheb_degrees))))
    quantiles = tuple(sorted(set(_csv_floats(args.cheb_interval_quantiles))))
    variants = _csv_strings(args.cheb_variants)
    modes = _csv_strings(args.cheb_modes)
    ridges = tuple(sorted(set(_csv_floats(args.ridge_values))))
    if (
        not ranks
        or not ratios
        or any(not 0.0 < value <= 1.0 for value in ratios)
        or set(scopes) - {"global_oracle", "tp_local"}
        or set(scores) - {"magnitude", "decoder_weighted"}
        or not block_sizes
        or min(block_sizes) <= 0
        or not degrees
        or any(degree < 2 for degree in degrees)
        or set(variants) - {"unconstrained", "silu_parity"}
        or set(modes) - {"global", "clipped"}
        or not quantiles
        or any(not 0.0 < value < 1.0 for value in quantiles)
        or not ridges
        or min(ridges) < 0.0
        or args.tp <= 0
        or min(args.chunk_size, args.metric_chunk_size) <= 0
    ):
        raise ValueError("invalid Phase-1 grid")

    output_dir = Path(args.output_dir).expanduser().resolve()
    partial_dir = output_dir.with_name(output_dir.name + ".partial")
    if output_dir.exists() or partial_dir.exists():
        raise FileExistsError(f"refusing to overwrite Phase-1 output: {output_dir}")
    snapshot_dir = Path(args.snapshots).expanduser().resolve()
    manifest, layer_records = _load_manifest(snapshot_dir)
    model_path, config, weight_map, model_metadata = _model_metadata(
        manifest,
        args.model_path,
    )
    layers = _parse_layers(
        args.layers,
        available=set(layer_records),
        layer_count=int(model_metadata["num_hidden_layers"]),
    )
    validated = _validate_snapshot_files(snapshot_dir, layer_records, layers)
    if any(
        int(layer_records[layer]["hidden_size"]) != int(model_metadata["hidden_size"])
        or int(layer_records[layer]["intermediate_size"])
        != int(model_metadata["intermediate_size"])
        for layer in layers
    ):
        raise ValueError("snapshot and checkpoint MLP dimensions differ")
    if int(model_metadata["intermediate_size"]) % args.tp:
        raise ValueError("MLP intermediate width is not divisible by TP size")
    if any(
        (int(model_metadata["intermediate_size"]) // args.tp) % block
        for block in block_sizes
    ):
        raise ValueError("a block size does not divide the TP-local MLP width")

    partial_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    cuda_device_index = (
        device.index if device.type == "cuda" and device.index is not None else 0
    )
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA is unavailable for requested device {device}")
        torch.cuda.set_device(cuda_device_index)
        torch.cuda.init()
    started = time.perf_counter()
    timestamp_started = datetime.now(timezone.utc).isoformat()
    all_records: list[dict[str, Any]] = []
    all_tuning: list[dict[str, Any]] = []
    layer_metadata: dict[str, Any] = {}
    wire_tensors: dict[str, Tensor] = {}
    shard_hashes: dict[str, str] = {}
    peak_cuda = 0
    for layer_index in layers:
        layer_started = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(cuda_device_index)
        print(f"[MLPPhase1] layer={layer_index} phase=load", flush=True)
        snapshots = {
            split: _load_tensors(validated[layer_index][split], ("A", "B", "C", "Y"))
            for split in SPLITS
        }
        hidden = int(layer_records[layer_index]["hidden_size"])
        intermediate = int(layer_records[layer_index]["intermediate_size"])
        down_weight, weight_metadata = _load_down_weight(
            model_path,
            weight_map,
            layer_index,
            expected_shape=(hidden, intermediate),
            shard_hashes=shard_hashes,
        )
        wires, factor_tensors, factor_metadata = _wire_bank(
            snapshots["train"]["Y"],
            down_weight,
            ranks,
            device=device,
            oversample=args.pod_oversample,
            niter=args.pod_niter,
            seed=args.seed + 7919 * layer_index,
        )
        for name, value in factor_tensors.items():
            wire_tensors[f"layer_{layer_index:02d}_{name}"] = value
        radii, quantile_metadata = _estimate_radii(
            snapshots["train"]["A"],
            quantiles,
            maximum_elements=args.quantile_max_elements,
            seed=args.seed + 104729 * layer_index,
        )
        print(f"[MLPPhase1] layer={layer_index} phase=topk", flush=True)
        topk = _topk_records_for_layer(
            layer_index=layer_index,
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
        all_records.extend(topk)
        print(f"[MLPPhase1] layer={layer_index} phase=chebyshev", flush=True)
        polynomial_count = 0
        for rank, wire in wires.items():
            polynomial, tuning = _polynomial_records_for_wire(
                layer_index=layer_index,
                wire=wire,
                snapshots=snapshots,
                radii=radii,
                quantile_metadata=quantile_metadata,
                degrees=degrees,
                variants=variants,
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
        layer_metadata[str(layer_index)] = {
            "elapsed_seconds": time.perf_counter() - layer_started,
            "peak_cuda_allocated_bytes": layer_peak,
            "weight": weight_metadata,
            "wire": factor_metadata,
            "radii": {str(key): value for key, value in radii.items()},
            "quantile_estimation": quantile_metadata,
            "topk_rows": len(topk),
            "polynomial_rows": polynomial_count,
        }
        del snapshots, down_weight, wires
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(
            f"[MLPPhase1] layer={layer_index} done "
            f"seconds={layer_metadata[str(layer_index)]['elapsed_seconds']:.3f}",
            flush=True,
        )

    decisions = _decisions(
        all_records,
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
        _summary_markdown(all_records, decisions),
    )
    plot_files = _plots(all_records, partial_dir)
    elapsed = time.perf_counter() - started
    result_manifest = {
        "format": FORMAT,
        "schema_version": 1,
        "command": shlex.join(sys.argv),
        "git_commit": _git_commit(),
        "timestamp_started_utc": timestamp_started,
        "timestamp_finished_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed,
        "model": model_metadata,
        "snapshot_manifest": str(snapshot_dir / "manifest.json"),
        "snapshot_manifest_sha256": _file_sha256(snapshot_dir / "manifest.json"),
        "validated_snapshots": validated,
        "layers": list(layers),
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
            "interval_quantiles": list(quantiles),
            "variants": list(variants),
            "modes": list(modes),
            "ridge_values": list(ridges),
            "polynomial_pass_threshold": args.polynomial_pass_threshold,
            "validation_used_for_hyperparameter_selection": False,
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
    _atomic_json(partial_dir / "manifest.json", result_manifest)
    os.replace(partial_dir, output_dir)
    print(
        f"[MLPPhase1] complete rows={len(all_records)} output={output_dir} "
        f"seconds={elapsed:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
