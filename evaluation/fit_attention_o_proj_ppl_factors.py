#!/usr/bin/env python3
"""Fit activation-aware equal-byte AR/AG factors for end-to-end PPL.

``fit-shard`` processes a disjoint subset of layers so several Slurm GPU tasks
can share one snapshot directory. ``merge`` validates complete layer coverage
and writes the factor manifest consumed by the PPL evaluator. This script does
not compute reconstruction error.
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
from typing import Any, Mapping

from safetensors.torch import load_file, save_file
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basisserve.analysis.o_proj_collective_ppl import (  # noqa: E402
    equal_byte_plan,
    fit_private_output_bases,
    fit_shared_output_basis,
    validate_equal_ring_bytes,
)
from evaluation.analyze_qwen35_gated_attention_sparsity import (  # noqa: E402
    _git_commit,
)
from evaluation.capture_attention_o_proj_ppl_snapshots import (  # noqa: E402
    FORMAT as SNAPSHOT_FORMAT,
)


FORMAT = "basisserve.attention_o_proj_ppl_factors.v1"
SHARD_FORMAT = "basisserve.attention_o_proj_ppl_factor_shard.v1"
METHOD_ORDER = ("ar", "tp_ag", "head_ag")
METHOD_KEYS = {
    "ar": "ar_basis",
    "tp_ag": "tp_ag_bases",
    "head_ag": "head_ag_bases",
}


def _add_common_fit_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--snapshot-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--methods", default="ar,tp_ag")
    parser.add_argument("--tp", type=int, default=8)
    parser.add_argument("--baseline-rank", type=int, default=1536)
    parser.add_argument("--dtype-bytes", type=int, default=2)
    parser.add_argument(
        "--basis-dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16"
    )
    parser.add_argument("--pod-oversample", type=int, default=16)
    parser.add_argument("--pod-niter", type=int, default=4)
    parser.add_argument("--output-chunk-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260902)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command_name", required=True)
    fit = subparsers.add_parser("fit-shard")
    _add_common_fit_args(fit)
    fit.add_argument("--layer-shard-index", type=int, required=True)
    fit.add_argument("--layer-shard-count", type=int, required=True)
    fit.add_argument("--device", default="cuda:0")
    fit.add_argument("--torch-num-threads", type=int, default=4)
    fit.add_argument("--resume", action="store_true")

    merge = subparsers.add_parser("merge")
    _add_common_fit_args(merge)
    merge.add_argument("--layer-shard-count", type=int, required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _atomic_safetensors(path: Path, tensors: Mapping[str, torch.Tensor]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    save_file(dict(tensors), str(temporary))
    os.replace(temporary, path)


def _methods(raw: str) -> tuple[str, ...]:
    requested = {piece.strip() for piece in raw.split(",") if piece.strip()}
    unknown = requested - set(METHOD_ORDER)
    if not requested or unknown:
        raise ValueError(f"invalid methods: {sorted(unknown or requested)}")
    return tuple(method for method in METHOD_ORDER if method in requested)


def _load_snapshot_manifest(snapshot_dir: Path) -> dict[str, Any]:
    path = snapshot_dir / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("format") != SNAPSHOT_FORMAT:
        raise ValueError("snapshot artifact has an incompatible format")
    return manifest


def _basis_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def _fit_config(
    args: argparse.Namespace,
    snapshot_dir: Path,
    snapshot_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    model = snapshot_manifest["model"]
    methods = _methods(args.methods)
    if "head_ag" in methods and model["attention_type"] != "mha":
        raise ValueError("head_ag is reserved for the MHA control")
    plan = equal_byte_plan(
        hidden_size=int(model["hidden_size"]),
        tp_size=args.tp,
        baseline_rank=args.baseline_rank,
        dtype_bytes=args.dtype_bytes,
    )
    accounting = validate_equal_ring_bytes(
        plan, num_attention_heads=int(model["num_attention_heads"])
    )
    heads_per_rank = int(model["num_attention_heads"]) // args.tp
    if int(model["num_attention_heads"]) % args.tp:
        raise ValueError("attention heads are not divisible by TP")
    if plan.private_rank_per_tp % heads_per_rank:
        raise ValueError("equal-byte private rank is not divisible across heads")
    return {
        "snapshot_dir": str(snapshot_dir),
        "snapshot_manifest_sha256": _sha256(snapshot_dir / "manifest.json"),
        "methods": list(methods),
        "tp_size": args.tp,
        "baseline_rank": args.baseline_rank,
        "private_total_rank": plan.private_total_rank,
        "private_rank_per_tp": plan.private_rank_per_tp,
        "head_private_rank": plan.private_rank_per_tp // heads_per_rank,
        "dtype_bytes": args.dtype_bytes,
        "basis_dtype": args.basis_dtype,
        "pod_oversample": args.pod_oversample,
        "pod_niter": args.pod_niter,
        "output_chunk_size": args.output_chunk_size,
        "seed": args.seed,
        "accounting": {method: accounting[method] for method in methods},
    }


def _record_matches(
    record: Mapping[str, Any], path: Path, methods: tuple[str, ...]
) -> bool:
    if not path.is_file() or record.get("sha256") != _sha256(path):
        return False
    try:
        payload = load_file(str(path), device="cpu")
    except Exception:
        return False
    return set(payload) == {METHOD_KEYS[method] for method in methods}


@torch.inference_mode()
def fit_shard(args: argparse.Namespace) -> None:
    if (
        args.layer_shard_count <= 0
        or not 0 <= args.layer_shard_index < args.layer_shard_count
        or min(
            args.tp,
            args.baseline_rank,
            args.dtype_bytes,
            args.output_chunk_size,
        )
        <= 0
        or min(args.pod_oversample, args.pod_niter, args.seed) < 0
    ):
        raise ValueError("fit-shard configuration is invalid")
    torch.set_num_threads(args.torch_num_threads)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("factor fitting requires CUDA")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)

    snapshot_dir = Path(args.snapshot_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    snapshot_manifest = _load_snapshot_manifest(snapshot_dir)
    config = _fit_config(args, snapshot_dir, snapshot_manifest)
    methods = tuple(config["methods"])
    all_layers = tuple(map(int, snapshot_manifest["layers"]))
    layers = tuple(
        layer
        for position, layer in enumerate(all_layers)
        if position % args.layer_shard_count == args.layer_shard_index
    )
    if not layers:
        raise ValueError("this layer shard is empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_path = output_dir / f"shard_{args.layer_shard_index:02d}.json"
    if shard_path.exists() and not args.resume:
        raise FileExistsError(shard_path)

    started = time.perf_counter()
    records: dict[str, Any] = {}
    basis_dtype = _basis_dtype(args.basis_dtype)
    hidden_size = int(snapshot_manifest["model"]["hidden_size"])
    num_heads = int(snapshot_manifest["model"]["num_attention_heads"])
    for layer in layers:
        layer_started = time.perf_counter()
        artifact_path = output_dir / f"layer_{layer:03d}.safetensors"
        record_path = output_dir / f"layer_{layer:03d}.json"
        if args.resume and record_path.is_file():
            prior = json.loads(record_path.read_text(encoding="utf-8"))
            if prior.get("fit_config") == config and _record_matches(
                prior, artifact_path, methods
            ):
                records[str(layer)] = prior
                print(f"[Fit] resumed layer={layer}", flush=True)
                continue
        elif artifact_path.exists() or record_path.exists():
            raise FileExistsError(f"partial layer artifact exists: {artifact_path}")

        source_record = snapshot_manifest["artifacts"][str(layer)]
        source_path = snapshot_dir / source_record["file"]
        if _sha256(source_path) != source_record["sha256"]:
            raise ValueError(f"snapshot hash mismatch for layer {layer}")
        source = load_file(str(source_path), device="cpu")
        activation = source["activation"]
        weight = source["weight"]
        if tuple(weight.shape) != (hidden_size, hidden_size):
            raise ValueError(f"unexpected o_proj weight shape at layer {layer}")
        tensors: dict[str, torch.Tensor] = {}
        diagnostics: dict[str, Any] = {}
        if "ar" in methods:
            print(f"[Fit] layer={layer} method=ar", flush=True)
            basis, fit = fit_shared_output_basis(
                activation,
                weight,
                args.baseline_rank,
                device=device,
                oversample=args.pod_oversample,
                niter=args.pod_niter,
                seed=args.seed + 7919 * layer,
                output_chunk_size=args.output_chunk_size,
            )
            tensors[METHOD_KEYS["ar"]] = basis.to(dtype=basis_dtype)
            diagnostics["ar"] = {
                "solver": fit["fit"],
                "rank": args.baseline_rank,
                "pod_method": fit["pod"]["method"],
                "polar_retraction": fit["polar_retraction"],
            }
            del basis
            torch.cuda.empty_cache()
        if "tp_ag" in methods:
            print(f"[Fit] layer={layer} method=tp_ag", flush=True)
            bases, fit = fit_private_output_bases(
                activation,
                weight,
                group_count=args.tp,
                rank_per_group=int(config["private_rank_per_tp"]),
                device=device,
            )
            tensors[METHOD_KEYS["tp_ag"]] = bases.to(dtype=basis_dtype)
            diagnostics["tp_ag"] = {
                "solver": fit[0]["solver"],
                "groups": args.tp,
                "rank_per_group": int(config["private_rank_per_tp"]),
            }
            del bases
            torch.cuda.empty_cache()
        if "head_ag" in methods:
            print(f"[Fit] layer={layer} method=head_ag", flush=True)
            bases, fit = fit_private_output_bases(
                activation,
                weight,
                group_count=num_heads,
                rank_per_group=int(config["head_private_rank"]),
                device=device,
            )
            tensors[METHOD_KEYS["head_ag"]] = bases.to(dtype=basis_dtype)
            diagnostics["head_ag"] = {
                "solver": fit[0]["solver"],
                "groups": num_heads,
                "rank_per_group": int(config["head_private_rank"]),
            }
            del bases
            torch.cuda.empty_cache()

        _atomic_safetensors(artifact_path, tensors)
        record = {
            "format": FORMAT,
            "layer": layer,
            "file": artifact_path.name,
            "sha256": _sha256(artifact_path),
            "fit_config": config,
            "tensors": {
                key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                for key, value in tensors.items()
            },
            "diagnostics": diagnostics,
            "elapsed_seconds": time.perf_counter() - layer_started,
        }
        _atomic_json(record_path, record)
        records[str(layer)] = record
        del source, activation, weight, tensors
        torch.cuda.empty_cache()
        print(
            f"[Fit] complete layer={layer} seconds={record['elapsed_seconds']:.3f}",
            flush=True,
        )

    shard_manifest = {
        "format": SHARD_FORMAT,
        "schema_version": 1,
        "command": shlex.join(sys.argv),
        "git_commit": _git_commit(),
        "timestamp_finished_utc": datetime.now(timezone.utc).isoformat(),
        "layer_shard_index": args.layer_shard_index,
        "layer_shard_count": args.layer_shard_count,
        "layers": list(layers),
        "fit_config": config,
        "records": records,
        "elapsed_seconds": time.perf_counter() - started,
        "environment": {
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_device": torch.cuda.get_device_name(device),
            "peak_cuda_allocated_bytes": int(
                torch.cuda.max_memory_allocated(device)
            ),
            "torch_num_threads": torch.get_num_threads(),
        },
    }
    _atomic_json(shard_path, shard_manifest)
    print(f"[Fit] wrote shard manifest={shard_path}", flush=True)


def merge(args: argparse.Namespace) -> None:
    if args.layer_shard_count <= 0:
        raise ValueError("layer shard count must be positive")
    snapshot_dir = Path(args.snapshot_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    snapshot_manifest = _load_snapshot_manifest(snapshot_dir)
    config = _fit_config(args, snapshot_dir, snapshot_manifest)
    methods = tuple(config["methods"])
    all_layers = tuple(map(int, snapshot_manifest["layers"]))
    records: dict[str, Any] = {}
    shard_files = []
    for shard in range(args.layer_shard_count):
        path = output_dir / f"shard_{shard:02d}.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("format") != SHARD_FORMAT
            or payload.get("fit_config") != config
            or int(payload.get("layer_shard_count", -1)) != args.layer_shard_count
            or int(payload.get("layer_shard_index", -1)) != shard
        ):
            raise ValueError(f"factor shard configuration differs: {path}")
        shard_files.append({"file": path.name, "sha256": _sha256(path)})
        for layer, record in payload["records"].items():
            if layer in records:
                raise ValueError(f"duplicate factor layer: {layer}")
            artifact_path = output_dir / record["file"]
            if not _record_matches(record, artifact_path, methods):
                raise ValueError(f"factor artifact is invalid: {artifact_path}")
            records[layer] = record
    if set(records) != {str(layer) for layer in all_layers}:
        raise ValueError("factor shards do not cover every snapshot layer")
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    manifest = {
        "format": FORMAT,
        "schema_version": 1,
        "command": shlex.join(sys.argv),
        "git_commit": _git_commit(),
        "timestamp_finished_utc": datetime.now(timezone.utc).isoformat(),
        "model": snapshot_manifest["model"],
        "layers": list(all_layers),
        "fit_config": config,
        "activation_calibration": snapshot_manifest["calibration"],
        "error_evaluation": False,
        "quality_endpoint": "wikitext2_ppl",
        "evaluation_representation": (
            "dense_reconstruction_of_collective_linear_map"
        ),
        "factor_shards": shard_files,
        "artifacts": records,
    }
    _atomic_json(manifest_path, manifest)
    print(f"[Merge] complete manifest={manifest_path}", flush=True)


def main() -> None:
    args = parse_args()
    if args.command_name == "fit-shard":
        fit_shard(args)
    else:
        merge(args)


if __name__ == "__main__":
    main()
