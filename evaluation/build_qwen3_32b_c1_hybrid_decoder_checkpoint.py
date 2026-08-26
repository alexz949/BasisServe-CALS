#!/usr/bin/env python3
"""Compose a uniform C1 checkpoint with guarded per-layer decoder fallback."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import sys
from typing import Any, Mapping, Sequence

from safetensors import safe_open
import torch


FACTOR_FORMAT = "basisserve.qwen3_32b.gqa_c1_v96_joint.v1"
HYBRID_FORMAT = "basisserve.qwen3_32b.gqa_c1_decoder_hybrid.v1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-factor-dir", type=Path, required=True)
    parser.add_argument("--fallback-factor-dir", type=Path, required=True)
    parser.add_argument("--fallback-layers", type=int, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_text(path: Path, payload: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


def _load_result(factor_dir: Path) -> dict[str, Any]:
    result_path = factor_dir / "results.json"
    if not result_path.is_file():
        raise FileNotFoundError(result_path)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("format") != FACTOR_FORMAT or result.get("status") != "complete":
        raise ValueError(f"incompatible uniform C1 checkpoint: {factor_dir}")
    layers = list(map(int, result.get("layers", [])))
    if layers != list(range(len(layers))) or not layers:
        raise ValueError(f"checkpoint has a non-contiguous layer schedule: {factor_dir}")
    if len(result.get("records", [])) != len(layers):
        raise ValueError(f"checkpoint records are incomplete: {factor_dir}")
    if set(map(int, result.get("artifacts", {}))) != set(layers):
        raise ValueError(f"checkpoint artifacts are incomplete: {factor_dir}")
    return result


def _geometry(fit_config: Mapping[str, Any]) -> dict[str, int | str]:
    keys = (
        "model_config_sha256",
        "attention_type",
        "num_hidden_layers",
        "num_query_heads",
        "num_physical_kv_heads",
        "head_dim",
        "hidden_size",
        "cache_rank_per_head",
        "factor_dtype",
    )
    missing = [key for key in keys if key not in fit_config]
    if missing:
        raise ValueError(f"factor config is missing geometry fields: {missing}")
    return {key: fit_config[key] for key in keys}


def _expected_tensors(fit_config: Mapping[str, Any]) -> dict[str, tuple[list[int], str]]:
    query_heads = int(fit_config["num_query_heads"])
    kv_heads = int(fit_config["num_physical_kv_heads"])
    head_dim = int(fit_config["head_dim"])
    hidden_size = int(fit_config["hidden_size"])
    rank = int(fit_config["cache_rank_per_head"])
    dtype = str(fit_config["factor_dtype"])
    if dtype != "bfloat16":
        raise ValueError(f"unsupported factor dtype: {dtype}")
    return {
        "value_coordinate_encoders": ([kv_heads, head_dim, rank], "BF16"),
        "head_output_decoders": ([query_heads, rank, hidden_size], "BF16"),
    }


def _validate_artifact(
    *,
    factor_dir: Path,
    artifact: Mapping[str, Any],
    expected_tensors: Mapping[str, tuple[list[int], str]],
) -> Path:
    path = factor_dir / str(artifact["file"])
    if not path.is_file():
        raise FileNotFoundError(path)
    if _sha256(path) != artifact.get("sha256"):
        raise ValueError(f"artifact hash mismatch: {path}")
    with safe_open(path, framework="pt", device="cpu") as handle:
        observed = {
            key: (
                list(handle.get_slice(key).get_shape()),
                handle.get_slice(key).get_dtype(),
            )
            for key in handle.keys()
        }
    if observed != dict(expected_tensors):
        raise ValueError(f"artifact tensor schema mismatch: {path}: {observed}")
    return path


def _load_encoder(path: Path) -> torch.Tensor:
    with safe_open(path, framework="pt", device="cpu") as handle:
        return handle.get_tensor("value_coordinate_encoders")


def _load_decoder_statistics(path: Path) -> tuple[float, float]:
    with safe_open(path, framework="pt", device="cpu") as handle:
        decoder = handle.get_tensor("head_output_decoders").to(torch.float64)
    return float(torch.linalg.vector_norm(decoder)), float(decoder.abs().max())


def _source_artifact(
    factor_dir: Path,
    result: Mapping[str, Any],
    layer: int,
    expected_tensors: Mapping[str, tuple[list[int], str]],
) -> tuple[Path, Mapping[str, Any]]:
    artifact = result["artifacts"][str(layer)]
    path = _validate_artifact(
        factor_dir=factor_dir,
        artifact=artifact,
        expected_tensors=expected_tensors,
    )
    return path, artifact


def _selected_metrics(
    primary_record: Mapping[str, Any],
    *,
    use_fallback: bool,
) -> tuple[float, float]:
    if not use_fallback:
        return (
            float(primary_record["fit"]["factor_dtype_relative_mse"]),
            float(primary_record["heldout"]["factor_dtype_relative_mse"]),
        )
    refit = primary_record.get("decoder_refit", {})
    try:
        return (
            float(refit["base_fit_factor_dtype_relative_mse"]),
            float(refit["base_heldout_factor_dtype_relative_mse"]),
        )
    except KeyError as error:
        raise ValueError(
            "primary checkpoint does not contain same-split fallback metrics"
        ) from error


def _hybrid_fit_config(
    primary: Mapping[str, Any],
    *,
    fallback_layers: Sequence[int],
    fallback: Mapping[str, Any],
) -> dict[str, Any]:
    fit_config = copy.deepcopy(primary["fit_config"])
    fit_config.update(
        {
            "decoder_solver": "hybrid_tsqr_qr0_with_ridge_guardrail",
            "decoder_primary_solver": primary["fit_config"].get("decoder_solver"),
            "decoder_fallback_covariance_damping": fallback["fit_config"].get(
                "covariance_damping"
            ),
            "decoder_fallback_layers": list(fallback_layers),
            "decoder_selection": (
                "TSQR QR(0) except layers whose same-split held-out relative "
                "MSE does not improve over the source ridge decoder"
            ),
        }
    )
    return fit_config


def _render_markdown(result: Mapping[str, Any], output_dir: Path) -> str:
    aggregate = result["aggregate"]
    composition = result["decoder_composition"]
    lines = [
        "# Qwen3-32B guarded TSQR/ridge hybrid checkpoint",
        "",
        "The Value encoders are identical across both source checkpoints. "
        "Each layer uses the 64k TSQR QR(0) decoder unless its same-split "
        "held-out error fails to improve over the original ridge decoder.",
        "",
        f"- Checkpoint: `{output_dir}`",
        f"- QR(0) layers: `{composition['primary_layer_count']}`",
        f"- Ridge fallback layers: `{composition['fallback_layers']}`",
        f"- Mean fit relMSE: "
        f"`{aggregate['mean_fit_factor_dtype_relative_mse']:.9e}`",
        f"- Mean held-out relMSE: "
        f"`{aggregate['mean_heldout_factor_dtype_relative_mse']:.9e}`",
        f"- Maximum held-out relMSE: "
        f"`{aggregate['maximum_heldout_factor_dtype_relative_mse']:.9e}`",
        "",
        "## Per-layer decoder selection",
        "",
        "| Layer | Source | Fit relMSE | Held-out relMSE |",
        "|---:|---|---:|---:|",
    ]
    for record in result["records"]:
        lines.append(
            f"| {record['layer']} | {record['decoder_selection']['source']} | "
            f"{record['fit']['factor_dtype_relative_mse']:.9e} | "
            f"{record['heldout']['factor_dtype_relative_mse']:.9e} |"
        )
    lines.extend(["", "## Command", "", "```bash", result["command"], "```", ""])
    return "\n".join(lines)


def compose_checkpoint(
    *,
    primary_factor_dir: Path,
    fallback_factor_dir: Path,
    fallback_layers: Sequence[int],
    output_dir: Path,
    output_markdown: Path,
    command: str,
) -> dict[str, Any]:
    primary_factor_dir = primary_factor_dir.expanduser().resolve()
    fallback_factor_dir = fallback_factor_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_markdown = output_markdown.expanduser().resolve()
    partial_dir = output_dir.with_name(output_dir.name + ".partial")
    for path in (output_dir, partial_dir, output_markdown):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite output: {path}")

    primary = _load_result(primary_factor_dir)
    fallback = _load_result(fallback_factor_dir)
    if _geometry(primary["fit_config"]) != _geometry(fallback["fit_config"]):
        raise ValueError("primary and fallback checkpoint geometry differs")
    layer_count = len(primary["layers"])
    if len(fallback["layers"]) != layer_count:
        raise ValueError("primary and fallback layer counts differ")
    fallback_set = set(map(int, fallback_layers))
    if len(fallback_set) != len(fallback_layers):
        raise ValueError("fallback layer list contains duplicates")
    if not fallback_set or not fallback_set.issubset(set(range(layer_count))):
        raise ValueError("fallback layer list is empty or out of range")

    primary_results_path = primary_factor_dir / "results.json"
    fallback_results_path = fallback_factor_dir / "results.json"
    primary_results_sha256 = _sha256(primary_results_path)
    fallback_results_sha256 = _sha256(fallback_results_path)
    expected_base_sha256 = primary.get("decoder_refit_source", {}).get(
        "base_results_sha256"
    )
    if expected_base_sha256 != fallback_results_sha256:
        raise ValueError(
            "fallback checkpoint is not the base decoder recorded by the primary"
        )

    expected_tensors = _expected_tensors(primary["fit_config"])
    fit_config = _hybrid_fit_config(
        primary,
        fallback_layers=sorted(fallback_set),
        fallback=fallback,
    )
    layer_sources: list[
        tuple[Path, Mapping[str, Any], Path, Mapping[str, Any]]
    ] = []
    for layer in range(layer_count):
        primary_path, primary_artifact = _source_artifact(
            primary_factor_dir,
            primary,
            layer,
            expected_tensors,
        )
        fallback_path, fallback_artifact = _source_artifact(
            fallback_factor_dir,
            fallback,
            layer,
            expected_tensors,
        )
        if not torch.equal(_load_encoder(primary_path), _load_encoder(fallback_path)):
            raise ValueError(f"layer {layer} Value encoders differ")
        layer_sources.append(
            (primary_path, primary_artifact, fallback_path, fallback_artifact)
        )

    partial_dir.mkdir(parents=True)
    records: list[dict[str, Any]] = []
    artifacts: dict[str, Any] = {}

    for layer in range(layer_count):
        (
            primary_path,
            primary_artifact,
            fallback_path,
            fallback_artifact,
        ) = layer_sources[layer]

        use_fallback = layer in fallback_set
        source_path = fallback_path if use_fallback else primary_path
        source_artifact = fallback_artifact if use_fallback else primary_artifact
        output_path = partial_dir / f"layer_{layer:03d}.safetensors"
        shutil.copyfile(source_path, output_path)
        output_sha256 = _sha256(output_path)
        if output_sha256 != source_artifact["sha256"]:
            raise RuntimeError(f"copied artifact hash mismatch: layer {layer}")
        artifact = copy.deepcopy(source_artifact)
        artifact["file"] = output_path.name
        artifact["sha256"] = output_sha256
        artifacts[str(layer)] = artifact

        primary_record = primary["records"][layer]
        fit_mse, heldout_mse = _selected_metrics(
            primary_record,
            use_fallback=use_fallback,
        )
        decoder_norm, decoder_max = _load_decoder_statistics(output_path)
        record = copy.deepcopy(primary_record)
        record.update(
            {
                "artifact": artifact,
                "fit_config": fit_config,
                "fit": {
                    "relative_mse": fit_mse,
                    "factor_dtype_relative_mse": fit_mse,
                },
                "heldout": {
                    "relative_mse": heldout_mse,
                    "factor_dtype_relative_mse": heldout_mse,
                },
                "decoder_selection": {
                    "source": "ridge_fallback" if use_fallback else "tsqr_qr0",
                    "source_factor_dir": str(
                        fallback_factor_dir if use_fallback else primary_factor_dir
                    ),
                    "source_results_sha256": (
                        fallback_results_sha256
                        if use_fallback
                        else primary_results_sha256
                    ),
                    "source_artifact_sha256": source_artifact["sha256"],
                    "same_split_fit_factor_dtype_relative_mse": fit_mse,
                    "same_split_heldout_factor_dtype_relative_mse": heldout_mse,
                    "primary_heldout_factor_dtype_relative_mse": float(
                        primary_record["heldout"]["factor_dtype_relative_mse"]
                    ),
                    "fallback_heldout_factor_dtype_relative_mse": float(
                        primary_record["decoder_refit"][
                            "base_heldout_factor_dtype_relative_mse"
                        ]
                    ),
                    "decoder_frobenius_norm": decoder_norm,
                    "decoder_maximum_absolute_value": decoder_max,
                },
            }
        )
        _atomic_json(partial_dir / f"layer_{layer:03d}.json", record)
        records.append(record)
        print(
            f"[Compose] layer={layer}/{layer_count - 1} "
            f"source={record['decoder_selection']['source']} "
            f"heldout={heldout_mse:.9e}",
            flush=True,
        )

    fit_values = [record["fit"]["factor_dtype_relative_mse"] for record in records]
    heldout_values = [
        record["heldout"]["factor_dtype_relative_mse"] for record in records
    ]
    fallback_heldout_values = [
        float(record["decoder_selection"]["fallback_heldout_factor_dtype_relative_mse"])
        for record in records
    ]
    aggregate = {
        "mean_fit_factor_dtype_relative_mse": sum(fit_values) / layer_count,
        "mean_heldout_factor_dtype_relative_mse": sum(heldout_values) / layer_count,
        "fallback_mean_heldout_factor_dtype_relative_mse": sum(
            fallback_heldout_values
        )
        / layer_count,
        "layers_improved_on_heldout": sum(
            value < fallback_value
            for value, fallback_value in zip(
                heldout_values,
                fallback_heldout_values,
                strict=True,
            )
        ),
        "maximum_heldout_factor_dtype_relative_mse": max(heldout_values),
    }
    result = copy.deepcopy(primary)
    result.update(
        {
            "format": FACTOR_FORMAT,
            "status": "complete",
            "command": command,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "fit_config": fit_config,
            "records": records,
            "artifacts": artifacts,
            "aggregate": aggregate,
            "decoder_composition": {
                "format": HYBRID_FORMAT,
                "primary_factor_dir": str(primary_factor_dir),
                "primary_results_sha256": primary_results_sha256,
                "primary_layer_count": layer_count - len(fallback_set),
                "fallback_factor_dir": str(fallback_factor_dir),
                "fallback_results_sha256": fallback_results_sha256,
                "fallback_layer_count": len(fallback_set),
                "fallback_layers": sorted(fallback_set),
                "encoder_validation": "exact_tensor_equality_all_layers",
                "artifact_copy": "byte_exact_independent_copy",
            },
            "environment": {
                "conda_environment": os.environ.get("CONDA_DEFAULT_ENV")
                or Path(sys.prefix).name,
                "python_executable": sys.executable,
                "python": sys.version,
                "torch": torch.__version__,
            },
        }
    )
    _atomic_json(partial_dir / "results.json", result)
    os.replace(partial_dir, output_dir)
    _atomic_text(output_markdown, _render_markdown(result, output_dir))
    print(
        f"[Checkpoint] complete output={output_dir} "
        f"mean_heldout={aggregate['mean_heldout_factor_dtype_relative_mse']:.9e}",
        flush=True,
    )
    return result


def main() -> None:
    args = _parse_args()
    compose_checkpoint(
        primary_factor_dir=args.primary_factor_dir,
        fallback_factor_dir=args.fallback_factor_dir,
        fallback_layers=args.fallback_layers,
        output_dir=args.output_dir,
        output_markdown=args.output_markdown,
        command=shlex.join([sys.executable, *sys.argv]),
    )


if __name__ == "__main__":
    main()
