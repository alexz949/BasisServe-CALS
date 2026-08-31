#!/usr/bin/env python3
"""Pack legacy uniform C1 layer artifacts into the canonical TP factor format."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import sys
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from safetensors.torch import load_file, save_file  # noqa: E402
import torch  # noqa: E402
from torch import Tensor  # noqa: E402

from basisserve.core.c1_tp_decode import (  # noqa: E402
    C1TPGeometry,
    RAGGED_C1_FACTOR_FORMAT,
    file_sha256,
)
from basisserve.core.decoder_closed_rank_candidates import (  # noqa: E402
    tensor_sha256,
)


LEGACY_UNIFORM_C1_LAYER_FORMAT = (
    "basisserve.qwen3_32b.gqa_c1_v96_joint.layer.v1"
)
SOURCE_TENSORS = {
    "value_coordinate_encoders",
    "head_output_decoders",
}


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON input must be an object: {path}")
    return payload


def _resolve_model_config(model: str | Path) -> Path:
    path = Path(model).expanduser().resolve()
    if path.is_dir():
        path = path / "config.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _expected_layer_paths(
    source_dir: Path,
    *,
    layer_count: int,
) -> tuple[tuple[Path, Path], ...]:
    expected = tuple(
        (
            source_dir / f"layer_{layer:03d}.json",
            source_dir / f"layer_{layer:03d}.safetensors",
        )
        for layer in range(layer_count)
    )
    expected_json = {pair[0] for pair in expected}
    expected_tensors = {pair[1] for pair in expected}
    observed_json = set(source_dir.glob("layer_*.json"))
    observed_tensors = set(source_dir.glob("layer_*.safetensors"))
    if observed_json != expected_json:
        raise ValueError(
            "uniform checkpoint JSON inventory does not cover the model exactly: "
            f"missing={sorted(expected_json - observed_json)}, "
            f"extra={sorted(observed_json - expected_json)}"
        )
    if observed_tensors != expected_tensors:
        raise ValueError(
            "uniform checkpoint tensor inventory does not cover the model exactly: "
            f"missing={sorted(expected_tensors - observed_tensors)}, "
            f"extra={sorted(observed_tensors - expected_tensors)}"
        )
    return expected


def _validate_tensor_metadata(
    metadata: Mapping[str, Any],
    tensors: Mapping[str, Tensor],
    *,
    layer: int,
) -> None:
    if set(metadata) != SOURCE_TENSORS:
        raise ValueError(f"layer {layer} has incomplete source tensor metadata")
    for name, tensor in tensors.items():
        record = metadata[name]
        if not isinstance(record, dict):
            raise ValueError(f"layer {layer} has invalid metadata for {name}")
        if record.get("shape") != list(tensor.shape):
            raise ValueError(f"layer {layer} metadata shape mismatch for {name}")
        if record.get("dtype") != str(tensor.dtype):
            raise ValueError(f"layer {layer} metadata dtype mismatch for {name}")


def _validate_fit_config(
    fit_config: Mapping[str, Any],
    geometry: C1TPGeometry,
    *,
    model_config_sha256: str,
    uniform_rank: int,
    layer: int,
) -> None:
    expected = {
        "model_config_sha256": model_config_sha256,
        "num_hidden_layers": geometry.num_layers,
        "num_physical_kv_heads": geometry.num_kv_heads,
        "num_query_heads": geometry.num_query_heads,
        "head_dim": geometry.head_dim,
        "hidden_size": geometry.hidden_size,
        "cache_rank_per_head": uniform_rank,
        "total_v_cache_rank": geometry.num_kv_heads * uniform_rank,
    }
    for key, value in expected.items():
        if fit_config.get(key) != value:
            raise ValueError(
                f"layer {layer} fit_config {key} mismatch: "
                f"{fit_config.get(key)!r} != {value!r}"
            )


def _pack_layer(
    *,
    layer: int,
    source_json_path: Path,
    source_tensor_path: Path,
    destination_dir: Path,
    geometry: C1TPGeometry,
    model_config_sha256: str,
    uniform_rank: int,
    expected_source_format: str,
    common_fit_config: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    source_json_sha256 = file_sha256(source_json_path)
    source = _read_json(source_json_path)
    if source.get("format") != expected_source_format:
        raise ValueError(
            f"layer {layer} source format mismatch: {source.get('format')!r}"
        )
    if source.get("layer") != layer:
        raise ValueError(f"layer index mismatch in {source_json_path}")
    fit_config = source.get("fit_config")
    if not isinstance(fit_config, dict):
        raise ValueError(f"layer {layer} has no fit_config")
    _validate_fit_config(
        fit_config,
        geometry,
        model_config_sha256=model_config_sha256,
        uniform_rank=uniform_rank,
        layer=layer,
    )
    if common_fit_config is not None and fit_config != common_fit_config:
        raise ValueError(f"layer {layer} fit_config differs from layer 0")

    source_artifact = source.get("artifact")
    if not isinstance(source_artifact, dict):
        raise ValueError(f"layer {layer} has no artifact metadata")
    if source_artifact.get("file") != source_tensor_path.name:
        raise ValueError(f"layer {layer} artifact file name mismatch")
    source_artifact_sha256 = file_sha256(source_tensor_path)
    if source_artifact.get("sha256") != source_artifact_sha256:
        raise ValueError(f"layer {layer} source artifact hash mismatch")

    tensors = load_file(str(source_tensor_path), device="cpu")
    if set(tensors) != SOURCE_TENSORS:
        raise ValueError(f"layer {layer} has unexpected source tensors")
    tensor_metadata = source_artifact.get("tensors")
    if not isinstance(tensor_metadata, dict):
        raise ValueError(f"layer {layer} has no tensor metadata")
    _validate_tensor_metadata(tensor_metadata, tensors, layer=layer)

    encoders = tensors["value_coordinate_encoders"]
    decoders = tensors["head_output_decoders"]
    expected_encoder_shape = (
        geometry.num_kv_heads,
        geometry.head_dim,
        uniform_rank,
    )
    expected_decoder_shape = (
        geometry.num_query_heads,
        uniform_rank,
        geometry.hidden_size,
    )
    if tuple(encoders.shape) != expected_encoder_shape:
        raise ValueError(
            f"layer {layer} encoder shape mismatch: "
            f"{tuple(encoders.shape)} != {expected_encoder_shape}"
        )
    if tuple(decoders.shape) != expected_decoder_shape:
        raise ValueError(
            f"layer {layer} decoder shape mismatch: "
            f"{tuple(decoders.shape)} != {expected_decoder_shape}"
        )
    if (
        encoders.dtype != decoders.dtype
        or not encoders.is_floating_point()
        or not decoders.is_floating_point()
    ):
        raise TypeError(f"layer {layer} C1 factor dtypes are incompatible")
    factor_dtype = str(encoders.dtype).removeprefix("torch.")
    if fit_config.get("factor_dtype") != factor_dtype:
        raise ValueError(f"layer {layer} factor_dtype does not match its tensors")

    source_ranks = torch.full(
        (geometry.tp_size,),
        uniform_rank,
        dtype=torch.int32,
    )
    canonical_tensors = {
        "value_coordinate_encoders": encoders.contiguous(),
        "head_output_decoders": decoders.contiguous(),
        "source_ranks": source_ranks,
    }
    destination_name = f"layer_{layer:03d}.safetensors"
    destination_path = destination_dir / destination_name
    save_file(canonical_tensors, str(destination_path))
    canonical_artifact_sha256 = file_sha256(destination_path)

    artifact = {
        "file": destination_name,
        "sha256": canonical_artifact_sha256,
        "encoder_sha256": tensor_sha256(encoders),
        "decoder_sha256": tensor_sha256(decoders),
        "tensors": {
            name: {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
            }
            for name, tensor in canonical_tensors.items()
        },
    }
    heldout = source.get("heldout")
    heldout_relative_mse = (
        heldout.get("relative_mse") if isinstance(heldout, dict) else None
    )
    if heldout_relative_mse is not None:
        heldout_relative_mse = float(heldout_relative_mse)
        if not math.isfinite(heldout_relative_mse) or heldout_relative_mse < 0.0:
            raise ValueError(f"layer {layer} has invalid heldout relative MSE")
    selection = source.get("selection")
    selected_sweep = selection.get("sweep") if isinstance(selection, dict) else None
    record = {
        "layer": layer,
        "source_layer_json": source_json_path.name,
        "source_layer_json_sha256": source_json_sha256,
        "source_artifact_sha256": source_artifact_sha256,
        "canonical_artifact_sha256": canonical_artifact_sha256,
        "source_ranks": [uniform_rank] * geometry.tp_size,
        "heldout_relative_mse": heldout_relative_mse,
        "selected_sweep": selected_sweep,
    }
    identity = {
        "layer": layer,
        "source_layer_json_sha256": source_json_sha256,
        "source_artifact_sha256": source_artifact_sha256,
    }
    return artifact, record, identity


def pack_uniform_c1_checkpoint(
    source_dir: str | Path,
    *,
    model: str | Path,
    output_dir: str | Path,
    uniform_rank: int,
    tp_size: int = 8,
    expected_source_format: str = LEGACY_UNIFORM_C1_LAYER_FORMAT,
    command: Sequence[str] = (),
) -> dict[str, Any]:
    """Validate and atomically pack a complete legacy uniform C1 checkpoint."""

    source_path = Path(source_dir).expanduser().resolve()
    if not source_path.is_dir():
        raise FileNotFoundError(source_path)
    destination_path = Path(output_dir).expanduser().resolve()
    if source_path == destination_path:
        raise ValueError("source and output checkpoint directories must differ")
    if destination_path.exists():
        raise FileExistsError(destination_path)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    staging_path = destination_path.parent / f".{destination_path.name}.packing"
    if staging_path.exists():
        raise FileExistsError(
            f"stale packing directory must be removed explicitly: {staging_path}"
        )

    rank = int(uniform_rank)
    if rank <= 0:
        raise ValueError("uniform rank must be positive")
    model_config_path = _resolve_model_config(model)
    geometry = C1TPGeometry.from_model_config(model_config_path, tp_size=tp_size)
    if rank > geometry.head_dim:
        raise ValueError(
            f"uniform rank {rank} exceeds head dimension {geometry.head_dim}"
        )
    model_config_sha256 = file_sha256(model_config_path)
    source_layers = _expected_layer_paths(
        source_path,
        layer_count=geometry.num_layers,
    )

    staging_path.mkdir()
    try:
        artifacts: dict[str, dict[str, Any]] = {}
        records: list[dict[str, Any]] = []
        identities: list[dict[str, Any]] = []
        common_fit_config: dict[str, Any] | None = None
        for layer, (source_json_path, source_tensor_path) in enumerate(source_layers):
            source_payload = _read_json(source_json_path)
            layer_fit_config = source_payload.get("fit_config")
            if not isinstance(layer_fit_config, dict):
                raise ValueError(f"layer {layer} has no fit_config")
            if common_fit_config is None:
                common_fit_config = layer_fit_config
            artifact, record, identity = _pack_layer(
                layer=layer,
                source_json_path=source_json_path,
                source_tensor_path=source_tensor_path,
                destination_dir=staging_path,
                geometry=geometry,
                model_config_sha256=model_config_sha256,
                uniform_rank=rank,
                expected_source_format=expected_source_format,
                common_fit_config=common_fit_config,
            )
            artifacts[str(layer)] = artifact
            records.append(record)
            identities.append(identity)
            print(
                json.dumps(
                    {
                        "event": "layer_packed",
                        "layer": layer,
                        "artifact_sha256": artifact["sha256"],
                    }
                ),
                flush=True,
            )
        if common_fit_config is None:
            raise AssertionError("uniform checkpoint has no layers")

        schedule = [
            [rank] * geometry.tp_size for _ in range(geometry.num_layers)
        ]
        source_rank_sum = geometry.num_layers * geometry.tp_size * rank
        rank_histogram = {str(rank): geometry.num_layers * geometry.tp_size}
        fit_config = dict(common_fit_config)
        fit_config.update(
            {
                "rank_schedule": schedule,
                "source_rank_sum": source_rank_sum,
                "dense_source_rank_sum": (
                    geometry.num_layers * geometry.tp_size * geometry.head_dim
                ),
                "rank_histogram": rank_histogram,
                "uniform_rank": rank,
                "source_checkpoint_dir": str(source_path),
                "source_checkpoint_format": expected_source_format,
                "source_checkpoint_identity_sha256": _canonical_sha256(identities),
            }
        )
        heldout_values = [
            float(record["heldout_relative_mse"])
            for record in records
            if record["heldout_relative_mse"] is not None
        ]
        aggregate = {
            "layer_count": geometry.num_layers,
            "uniform_rank": rank,
            "source_rank_sum": source_rank_sum,
            "dense_source_rank_sum": (
                geometry.num_layers * geometry.tp_size * geometry.head_dim
            ),
            "v_retained_ratio": rank / geometry.head_dim,
            "mean_heldout_relative_mse": (
                sum(heldout_values) / len(heldout_values)
                if heldout_values
                else None
            ),
            "maximum_heldout_relative_mse": (
                max(heldout_values) if heldout_values else None
            ),
        }
        result = {
            "format": RAGGED_C1_FACTOR_FORMAT,
            "status": "complete",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "command": shlex.join(tuple(command)),
            "layers": list(range(geometry.num_layers)),
            "fit_config": fit_config,
            "selection": {
                "selected_candidate": f"uniform_v{rank}",
                "selected_schedule": schedule,
                "source_rank_sum": source_rank_sum,
            },
            "records": records,
            "aggregate": aggregate,
            "artifacts": artifacts,
        }
        result_path = staging_path / "result.json"
        result_path.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(staging_path, destination_path)
    except BaseException:
        shutil.rmtree(staging_path, ignore_errors=True)
        raise

    result_path = destination_path / "result.json"
    return {
        "output_dir": str(destination_path),
        "result_path": str(result_path),
        "result_sha256": file_sha256(result_path),
        "source_checkpoint_identity_sha256": fit_config[
            "source_checkpoint_identity_sha256"
        ],
        "layer_count": geometry.num_layers,
        "uniform_rank": rank,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--tp-size", type=int, default=8)
    parser.add_argument(
        "--expected-source-format",
        default=LEGACY_UNIFORM_C1_LAYER_FORMAT,
    )
    args = parser.parse_args()
    result = pack_uniform_c1_checkpoint(
        args.input_dir,
        model=args.model,
        output_dir=args.output_dir,
        uniform_rank=args.rank,
        tp_size=args.tp_size,
        expected_source_format=args.expected_source_format,
        command=(sys.executable, *sys.argv),
    )
    print(json.dumps({"event": "packing_complete", **result}, indent=2), flush=True)


if __name__ == "__main__":
    main()
