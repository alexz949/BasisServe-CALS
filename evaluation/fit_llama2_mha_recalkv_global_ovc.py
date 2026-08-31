#!/usr/bin/env python3
"""Fit the ReCalKV global-OVC dense-reconstruction reference for Llama-2-7B.

K remains dense.  The complete 4096-wide Value projection is represented by
one rank-r encoder and one dense reconstruction matrix.  This is the
full-matrix/global Value path used by ReCalKV, not a per-head factorization.

The OVC least-squares update is evaluated from a Cholesky sufficient statistic
instead of retaining calibration rows.  If C = X^T X = S S^T, then

    ||X W^T - X E^T D^T||_F = ||(W - D E) S||_F,

so the decoder update is ``lstsq((E S)^T, (W S)^T)``.  The encoder update is
``lstsq(D, W)``, algebraically equal to ReCalKV's ``pinv(D) @ W``.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys
import time
from typing import Any, Mapping, Sequence

from safetensors import safe_open
from safetensors.torch import save_file
import torch
from torch import Tensor


FORMAT = "basisserve.llama2_7b.recalkv_global_ovc_reference.v1"
LAYER_FORMAT = "basisserve.llama2_7b.recalkv_global_ovc_reference.layer.v1"
WHITENING_FORMAT = "basisserve.llama2_7b.mha_v25_c4_whitening.v1"
UPSTREAM_REPOSITORY = "https://github.com/XIANGLONGYAN/ReCalKV.git"
UPSTREAM_COMMIT = "ea548d5c6c648316f009770a2e54b80fa16c6b06"
NUM_LAYERS = 32
NUM_HEADS = 32
HEAD_DIM = 128
HIDDEN_SIZE = NUM_HEADS * HEAD_DIM


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_safetensors(path: Path, payload: Mapping[str, Tensor]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    save_file(dict(payload), str(temporary))
    os.replace(temporary, path)


def _parse_layers(raw: str) -> tuple[int, ...]:
    if raw.strip().lower() == "all":
        return tuple(range(NUM_LAYERS))
    layers: set[int] = set()
    for piece in raw.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "-" in piece:
            first, last = map(int, piece.split("-", 1))
            if last < first:
                raise ValueError(f"descending layer range: {piece}")
            layers.update(range(first, last + 1))
        else:
            layers.add(int(piece))
    selected = tuple(sorted(layers))
    if not selected or min(selected) < 0 or max(selected) >= NUM_LAYERS:
        raise ValueError("selected layers are outside Llama-2-7B")
    return selected


def _work_dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float64": torch.float64}[name]


def _factor_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def _model_metadata(model_path: Path) -> tuple[dict[str, Any], dict[str, str]]:
    config_path = model_path / "config.json"
    index_path = model_path / "model.safetensors.index.json"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    if not index_path.is_file():
        raise FileNotFoundError(index_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    observed = (
        str(config.get("model_type")),
        int(config.get("num_hidden_layers", -1)),
        int(config.get("hidden_size", -1)),
        int(config.get("num_attention_heads", -1)),
        int(config.get("num_key_value_heads", -1)),
    )
    expected = ("llama", NUM_LAYERS, HIDDEN_SIZE, NUM_HEADS, NUM_HEADS)
    if observed != expected:
        raise ValueError(f"expected Llama-2-7B MHA, found {observed}")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("model safetensors index has no weight_map")
    for layer in range(NUM_LAYERS):
        name = f"model.layers.{layer}.self_attn.v_proj.weight"
        shard = weight_map.get(name)
        if not isinstance(shard, str) or Path(shard).name != shard:
            raise ValueError(f"missing or unsafe Value tensor mapping: {name}")
        if not (model_path / shard).is_file():
            raise FileNotFoundError(model_path / shard)
    return {
        "path": str(model_path),
        "config_sha256": _sha256(config_path),
        "safetensors_index_sha256": _sha256(index_path),
    }, dict(weight_map)


def _whitening_metadata(
    directory: Path,
    *,
    model_config_sha256: str,
) -> tuple[dict[str, Any], Path]:
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != WHITENING_FORMAT:
        raise ValueError("incompatible whitening artifact")
    if manifest.get("model_config_sha256") != model_config_sha256:
        raise ValueError("whitening artifact belongs to another model")
    artifact = directory / manifest["artifact"]["file"]
    if not artifact.is_file():
        raise FileNotFoundError(artifact)
    if _sha256(artifact) != manifest["artifact"]["sha256"]:
        raise ValueError("whitening artifact hash changed")
    return {
        "directory": str(directory),
        "manifest_sha256": _sha256(manifest_path),
        "artifact": str(artifact),
        "artifact_sha256": manifest["artifact"]["sha256"],
        "format": manifest["format"],
        "fit_windows": manifest.get("fit_windows"),
        "sequence_length": manifest.get("sequence_length"),
        "windows": manifest.get("windows"),
        "windows_sha256": manifest.get("windows_sha256"),
        "aggregation": manifest.get("aggregation"),
    }, artifact


def _load_tensor(path: Path, name: str) -> Tensor:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        if name not in handle.keys():
            raise KeyError(f"{name} is absent from {path}")
        return handle.get_tensor(name).contiguous()


def _load_v_weight(
    model_path: Path,
    weight_map: Mapping[str, str],
    layer: int,
) -> tuple[Tensor, dict[str, Any]]:
    name = f"model.layers.{layer}.self_attn.v_proj.weight"
    shard = model_path / weight_map[name]
    weight = _load_tensor(shard, name)
    if tuple(weight.shape) != (HIDDEN_SIZE, HIDDEN_SIZE):
        raise ValueError(f"unexpected Value weight shape at layer {layer}")
    return weight, {
        "tensor": name,
        "shard": shard.name,
        "dtype": str(weight.dtype),
    }


def _load_whitening_layer(path: Path, layer: int) -> Tensor:
    factor = _load_tensor(path, f"layer_{layer:03d}")
    if tuple(factor.shape) != (HIDDEN_SIZE, HIDDEN_SIZE):
        raise ValueError(f"unexpected whitening shape at layer {layer}")
    return factor


def _relative_covariance_error(
    weight: Tensor,
    decoder: Tensor,
    encoder: Tensor,
    covariance_factor: Tensor,
) -> float:
    target = weight @ covariance_factor
    residual = target - decoder @ (encoder @ covariance_factor)
    denominator = torch.linalg.vector_norm(target)
    if float(denominator) == 0.0:
        raise ValueError("zero calibration target norm")
    return float(torch.linalg.vector_norm(residual) / denominator)


@torch.no_grad()
def global_ovc_from_sufficient_statistics(
    weight: Tensor,
    initialization_whitening: Tensor,
    *,
    rank: int,
    iterations: int,
    ovc_whitening: Tensor | None = None,
) -> tuple[Tensor, Tensor, dict[str, Any]]:
    """Return dense decoder D and latent encoder E for ``W ~= D @ E``."""

    if weight.ndim != 2:
        raise ValueError("weight must be a matrix")
    output_width, input_width = map(int, weight.shape)
    if tuple(initialization_whitening.shape) != (input_width, input_width):
        raise ValueError("initialization whitening has incompatible geometry")
    if not 0 < rank <= min(output_width, input_width):
        raise ValueError("rank exceeds the matrix geometry")
    if iterations < 0:
        raise ValueError("OVC iterations must be nonnegative")
    if ovc_whitening is None:
        ovc_whitening = initialization_whitening
    if tuple(ovc_whitening.shape) != (input_width, input_width):
        raise ValueError("OVC whitening has incompatible geometry")

    scaled_weight = weight @ initialization_whitening
    left, singular_values, right_t = torch.linalg.svd(
        scaled_weight, full_matrices=False
    )
    roots = singular_values[:rank].sqrt()
    decoder = left[:, :rank] * roots.unsqueeze(0)
    scaled_encoder = roots.unsqueeze(1) * right_t[:rank]
    encoder = torch.linalg.solve_triangular(
        initialization_whitening.transpose(0, 1),
        scaled_encoder.transpose(0, 1),
        upper=True,
    ).transpose(0, 1).contiguous()
    diagnostics: dict[str, Any] = {
        "initial_relative_rmse_initialization_covariance": _relative_covariance_error(
            weight, decoder, encoder, initialization_whitening
        ),
        "updates": [],
        "truncated_singular_value_first": float(singular_values[0]),
        "truncated_singular_value_last": float(singular_values[rank - 1]),
    }
    target = weight @ ovc_whitening
    for iteration in range(iterations):
        before = _relative_covariance_error(
            weight, decoder, encoder, ovc_whitening
        )
        encoded = encoder @ ovc_whitening
        decoder_t = torch.linalg.lstsq(
            encoded.transpose(0, 1), target.transpose(0, 1)
        ).solution
        decoder = decoder_t.transpose(0, 1).contiguous()
        after_decoder = _relative_covariance_error(
            weight, decoder, encoder, ovc_whitening
        )
        encoder = torch.linalg.lstsq(decoder, weight).solution.contiguous()
        after_encoder = _relative_covariance_error(
            weight, decoder, encoder, ovc_whitening
        )
        diagnostics["updates"].append(
            {
                "iteration": iteration + 1,
                "relative_rmse_before": before,
                "relative_rmse_after_decoder": after_decoder,
                "relative_rmse_after_encoder": after_encoder,
            }
        )
    diagnostics["final_relative_rmse_initialization_covariance"] = (
        _relative_covariance_error(
            weight, decoder, encoder, initialization_whitening
        )
    )
    diagnostics["final_relative_rmse_ovc_covariance"] = _relative_covariance_error(
        weight, decoder, encoder, ovc_whitening
    )
    return decoder, encoder, diagnostics


def _fit_config(
    args: argparse.Namespace,
    *,
    model: Mapping[str, Any],
    whitening: Mapping[str, Any],
    ovc_whitening: Mapping[str, Any],
) -> dict[str, Any]:
    if not 0 < args.rank <= HIDDEN_SIZE:
        raise ValueError("global Value rank is outside the Llama hidden width")
    if args.ovc_iterations < 0:
        raise ValueError("OVC iterations must be nonnegative")
    return {
        "method": "ReCalKV global OVC",
        "classification": "dense-reconstruction oracle/reference",
        "model": model["path"],
        "model_config_sha256": model["config_sha256"],
        "model_safetensors_index_sha256": model["safetensors_index_sha256"],
        "key_projection": "dense and unchanged",
        "value_factorization_scope": "one global/full-layer W_V matrix",
        "value_rank": args.rank,
        "dense_value_width": HIDDEN_SIZE,
        "value_retained_ratio": args.rank / HIDDEN_SIZE,
        "total_kv_retained_ratio_with_dense_k": (HIDDEN_SIZE + args.rank)
        / (2 * HIDDEN_SIZE),
        "dense_reconstruction_before_mha": True,
        "deployable_fused_kernel_claim": False,
        "ovc_iterations": args.ovc_iterations,
        "work_dtype": args.work_dtype,
        "factor_dtype": args.factor_dtype,
        "initialization_whitening": dict(whitening),
        "ovc_whitening": dict(ovc_whitening),
        "ovc_covariance_matches_initialization": whitening == ovc_whitening,
        "upstream": {
            "repository": UPSTREAM_REPOSITORY,
            "commit": UPSTREAM_COMMIT,
            "value_rank_scope": (
                "official decomposition sums selected V head ranks and passes "
                "the total as a single group"
            ),
            "ovc_update": (
                "decoder least squares followed by encoder pinv refit"
            ),
        },
    }


def _resume_record(
    record_path: Path,
    artifact_path: Path,
    *,
    layer: int,
    fit_config: Mapping[str, Any],
) -> dict[str, Any] | None:
    if not record_path.exists() and not artifact_path.exists():
        return None
    if not record_path.is_file() or not artifact_path.is_file():
        raise FileExistsError(f"partial output exists for layer {layer}")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if (
        record.get("format") != LAYER_FORMAT
        or int(record.get("layer", -1)) != layer
        or record.get("fit_config") != fit_config
        or record.get("artifact", {}).get("sha256") != _sha256(artifact_path)
    ):
        raise ValueError(f"incompatible resume output at layer {layer}")
    print(f"[ReCalKV OVC] layer={layer} resume hit", flush=True)
    return record


@torch.no_grad()
def _fit_layer(
    *,
    layer: int,
    model_path: Path,
    weight_map: Mapping[str, str],
    whitening_path: Path,
    ovc_whitening_path: Path,
    output_dir: Path,
    fit_config: Mapping[str, Any],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    record_path = output_dir / f"layer_{layer:03d}.json"
    artifact_path = output_dir / f"layer_{layer:03d}.safetensors"
    if args.resume:
        resumed = _resume_record(
            record_path,
            artifact_path,
            layer=layer,
            fit_config=fit_config,
        )
        if resumed is not None:
            return resumed
    elif record_path.exists() or artifact_path.exists():
        raise FileExistsError(f"layer {layer} output already exists")

    started = time.perf_counter()
    work_dtype = _work_dtype(args.work_dtype)
    factor_dtype = _factor_dtype(args.factor_dtype)
    weight_cpu, weight_source = _load_v_weight(model_path, weight_map, layer)
    whitening_cpu = _load_whitening_layer(whitening_path, layer)
    ovc_whitening_cpu = (
        whitening_cpu
        if ovc_whitening_path == whitening_path
        else _load_whitening_layer(ovc_whitening_path, layer)
    )
    weight = weight_cpu.to(device=device, dtype=work_dtype)
    whitening = whitening_cpu.to(device=device, dtype=work_dtype)
    ovc_whitening = ovc_whitening_cpu.to(device=device, dtype=work_dtype)
    decoder, encoder, diagnostics = global_ovc_from_sufficient_statistics(
        weight,
        whitening,
        rank=args.rank,
        iterations=args.ovc_iterations,
        ovc_whitening=ovc_whitening,
    )
    artifact_decoder = decoder.to(device="cpu", dtype=factor_dtype).contiguous()
    artifact_encoder = encoder.to(device="cpu", dtype=factor_dtype).contiguous()
    quantized_decoder = artifact_decoder.to(device=device, dtype=work_dtype)
    quantized_encoder = artifact_encoder.to(device=device, dtype=work_dtype)
    diagnostics["factor_dtype_relative_rmse_initialization_covariance"] = (
        _relative_covariance_error(
            weight, quantized_decoder, quantized_encoder, whitening
        )
    )
    diagnostics["factor_dtype_relative_rmse_ovc_covariance"] = (
        _relative_covariance_error(
            weight, quantized_decoder, quantized_encoder, ovc_whitening
        )
    )
    tensors = {
        "v_encoder_weight": artifact_encoder,
        "v_reconstruction_weight": artifact_decoder,
    }
    _atomic_safetensors(artifact_path, tensors)
    record = {
        "format": LAYER_FORMAT,
        "layer": layer,
        "fit_config": dict(fit_config),
        "source_weight": weight_source,
        "artifact": {
            "file": artifact_path.name,
            "sha256": _sha256(artifact_path),
            "tensors": {
                name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                for name, value in tensors.items()
            },
        },
        "diagnostics": diagnostics,
        "elapsed_seconds": time.perf_counter() - started,
    }
    _atomic_json(record_path, record)
    print(
        f"[ReCalKV OVC] layer={layer} "
        f"rmse={diagnostics['factor_dtype_relative_rmse_initialization_covariance']:.8g} "
        f"seconds={record['elapsed_seconds']:.2f}",
        flush=True,
    )
    del (
        weight_cpu,
        whitening_cpu,
        ovc_whitening_cpu,
        weight,
        whitening,
        ovc_whitening,
        decoder,
        encoder,
        artifact_decoder,
        artifact_encoder,
        quantized_decoder,
        quantized_encoder,
        tensors,
    )
    torch.cuda.empty_cache()
    return record


def _fit_shard(args: argparse.Namespace) -> None:
    if args.layer_shard_count <= 0 or not 0 <= args.layer_shard_index < args.layer_shard_count:
        raise ValueError("invalid layer shard")
    torch.set_num_threads(args.torch_num_threads)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("global OVC fitting requires CUDA")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    model_path = Path(args.model).expanduser().resolve()
    whitening_dir = Path(args.whitening_dir).expanduser().resolve()
    ovc_whitening_dir = (
        Path(args.ovc_whitening_dir).expanduser().resolve()
        if args.ovc_whitening_dir is not None
        else whitening_dir
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    model, weight_map = _model_metadata(model_path)
    whitening, whitening_path = _whitening_metadata(
        whitening_dir, model_config_sha256=model["config_sha256"]
    )
    if ovc_whitening_dir == whitening_dir:
        ovc_whitening, ovc_whitening_path = whitening, whitening_path
    else:
        ovc_whitening, ovc_whitening_path = _whitening_metadata(
            ovc_whitening_dir, model_config_sha256=model["config_sha256"]
        )
    fit_config = _fit_config(
        args,
        model=model,
        whitening=whitening,
        ovc_whitening=ovc_whitening,
    )
    requested = _parse_layers(args.layers)
    layers = tuple(
        layer
        for position, layer in enumerate(requested)
        if position % args.layer_shard_count == args.layer_shard_index
    )
    if not layers:
        raise ValueError("layer shard is empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    records = []
    for layer in layers:
        records.append(
            _fit_layer(
                layer=layer,
                model_path=model_path,
                weight_map=weight_map,
                whitening_path=whitening_path,
                ovc_whitening_path=ovc_whitening_path,
                output_dir=output_dir,
                fit_config=fit_config,
                args=args,
                device=device,
            )
        )
    shard = {
        "format": FORMAT,
        "status": "shard_complete",
        "command": shlex.join(sys.argv),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "layer_shard_index": args.layer_shard_index,
        "layer_shard_count": args.layer_shard_count,
        "layers": list(layers),
        "fit_config": fit_config,
        "records": [int(record["layer"]) for record in records],
        "elapsed_seconds": time.perf_counter() - started,
        "environment": {
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_device": torch.cuda.get_device_name(device),
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "torch_num_threads": torch.get_num_threads(),
        },
    }
    _atomic_json(output_dir / f"shard_{args.layer_shard_index:02d}.json", shard)
    print(
        f"[ReCalKV OVC] shard={args.layer_shard_index}/{args.layer_shard_count} complete",
        flush=True,
    )


def _summary(records: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> str:
    metric = "factor_dtype_relative_rmse_initialization_covariance"
    errors = [float(record["diagnostics"][metric]) for record in records]
    lines = [
        "# ReCalKV global-OVC dense-reconstruction reference",
        "",
        "- K remains completely dense and unchanged.",
        (
            f"- The full-layer Value matrix uses one global rank-{config['value_rank']} "
            f"factor ({100 * config['value_retained_ratio']:.6g}% V retention)."
        ),
        (
            f"- Total KV-cache retention is "
            f"{100 * config['total_kv_retained_ratio_with_dense_k']:.6g}%."
        ),
        "- This path reconstructs dense 4096-wide V before stock MHA; it is an oracle/reference, not a fused-kernel result.",
        f"- OVC alternating updates: {config['ovc_iterations']}.",
        f"- Mean factor-dtype calibration relative RMSE: `{sum(errors) / len(errors):.9g}`.",
        "",
        "| Layer | Factor-dtype relative RMSE |",
        "|---:|---:|",
    ]
    for record in records:
        lines.append(f"| {record['layer']} | {record['diagnostics'][metric]:.9g} |")
    lines.append("")
    return "\n".join(lines)


def _merge(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not output_dir.is_dir():
        raise FileNotFoundError(output_dir)
    result_path = output_dir / "results.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    layers = _parse_layers(args.layers)
    records = []
    fit_config = None
    for layer in layers:
        record_path = output_dir / f"layer_{layer:03d}.json"
        artifact_path = output_dir / f"layer_{layer:03d}.safetensors"
        if not record_path.is_file() or not artifact_path.is_file():
            raise FileNotFoundError(f"missing layer {layer} output")
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record.get("format") != LAYER_FORMAT or int(record.get("layer", -1)) != layer:
            raise ValueError(f"invalid layer record {record_path}")
        if record["artifact"]["sha256"] != _sha256(artifact_path):
            raise ValueError(f"artifact hash mismatch at layer {layer}")
        if fit_config is None:
            fit_config = record["fit_config"]
        elif record["fit_config"] != fit_config:
            raise ValueError("layer fit configurations differ")
        records.append(record)
    assert fit_config is not None
    metric = "factor_dtype_relative_rmse_initialization_covariance"
    payload = {
        "format": FORMAT,
        "status": "complete",
        "command": shlex.join(sys.argv),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "label": "ReCalKV global-OVC rank3072 dense-reconstruction oracle/reference",
        "layers": list(layers),
        "fit_config": fit_config,
        "artifacts": {str(record["layer"]): record["artifact"] for record in records},
        "records": records,
        "aggregate": {
            "mean_factor_dtype_relative_rmse": sum(
                float(record["diagnostics"][metric]) for record in records
            )
            / len(records),
        },
    }
    _atomic_json(result_path, payload)
    (output_dir / "summary.md").write_text(
        _summary(records, fit_config), encoding="utf-8"
    )
    print(f"[ReCalKV OVC] merged {len(records)} layers into {result_path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command_name", required=True)
    fit = subparsers.add_parser("fit-shard")
    fit.add_argument("--model", required=True)
    fit.add_argument("--whitening-dir", required=True)
    fit.add_argument(
        "--ovc-whitening-dir",
        help="Optional distinct OVC covariance; defaults to initialization whitening",
    )
    fit.add_argument("--output-dir", required=True)
    fit.add_argument("--layers", default="all")
    fit.add_argument("--rank", type=int, default=3072)
    fit.add_argument("--ovc-iterations", type=int, default=1)
    fit.add_argument("--work-dtype", choices=("float32", "float64"), default="float32")
    fit.add_argument(
        "--factor-dtype",
        choices=("float16", "bfloat16", "float32"),
        default="float16",
    )
    fit.add_argument("--layer-shard-index", type=int, required=True)
    fit.add_argument("--layer-shard-count", type=int, required=True)
    fit.add_argument("--device", default="cuda:0")
    fit.add_argument("--torch-num-threads", type=int, default=2)
    fit.add_argument("--resume", action="store_true")
    merge = subparsers.add_parser("merge")
    merge.add_argument("--output-dir", required=True)
    merge.add_argument("--layers", default="all")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command_name == "fit-shard":
        _fit_shard(args)
    else:
        _merge(args)


if __name__ == "__main__":
    main()
