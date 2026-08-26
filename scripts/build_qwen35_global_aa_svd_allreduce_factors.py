#!/usr/bin/env python3
"""Build communication-matched global AA-SVD factors for Qwen3.5 output wires."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import shlex
import sys
import time
from typing import Any

from safetensors import safe_open
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basisserve.core.qwen35_global_aa_svd import (  # noqa: E402
    FACTOR_FORMAT,
    fit_global_activation_aware_svd,
)
from scripts.build_qwen35_full_attention_private_ag_joint_factors import (  # noqa: E402
    FULL_ATTENTION_BUILD_SPEC,
)
from scripts.build_qwen35_gdn_private_ag_joint_factors import (  # noqa: E402
    GDN_BUILD_SPEC,
    Qwen35PrivateAGBuildSpec,
    _load_moments,
    _validate_moment_pair,
    _weight_map,
)


NUM_LAYERS = 32
HIDDEN_SIZE = 4096
GDN_LAYERS = tuple(layer for layer in range(NUM_LAYERS) if layer % 4 != 3)
FULL_LAYERS = tuple(layer for layer in range(NUM_LAYERS) if layer % 4 == 3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--gdn-fit-moments", required=True)
    parser.add_argument("--gdn-heldout-moments", required=True)
    parser.add_argument("--full-fit-moments", required=True)
    parser.add_argument("--full-heldout-moments", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rank", type=int, default=768)
    parser.add_argument("--tp", type=int, default=8)
    parser.add_argument("--covariance-damping", type=float, default=1.0e-5)
    parser.add_argument(
        "--factor-dtype",
        choices=("bfloat16", "float16", "float32"),
        default="float16",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--torch-num-threads", type=int, default=2)
    return parser.parse_args()


def _dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def _load_pair(
    fit_path: Path,
    heldout_path: Path,
    spec: Qwen35PrivateAGBuildSpec,
) -> tuple[dict[str, Any], dict[str, Any]]:
    fit = _load_moments(fit_path, spec)
    heldout = _load_moments(heldout_path, spec)
    _validate_moment_pair(fit, heldout)
    expected = GDN_LAYERS if spec is GDN_BUILD_SPEC else FULL_LAYERS
    fit_layers = tuple(int(layer["layer_index"]) for layer in fit["layers"])
    if fit_layers != expected:
        raise ValueError(f"{spec.block_type} moments do not cover expected layers")
    if int(fit["geometry"]["wire_input_width"]) != HIDDEN_SIZE:
        raise ValueError(f"{spec.block_type} moments have unexpected wire width")
    return fit, heldout


def _by_layer(payload: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(layer["layer_index"]): layer for layer in payload["layers"]}


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.torch_num_threads)
    if args.tp <= 1 or HIDDEN_SIZE % args.tp:
        raise ValueError("TP size must divide the Qwen3.5 hidden width")
    if not 0 < args.rank <= HIDDEN_SIZE:
        raise ValueError("global rank is outside the Qwen3.5 hidden width")
    model_path = Path(args.model_path).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite global AA-SVD factors: {output_path}")
    paths = {
        "gdn_fit": Path(args.gdn_fit_moments).expanduser().resolve(),
        "gdn_heldout": Path(args.gdn_heldout_moments).expanduser().resolve(),
        "full_fit": Path(args.full_fit_moments).expanduser().resolve(),
        "full_heldout": Path(args.full_heldout_moments).expanduser().resolve(),
    }
    gdn_fit, gdn_heldout = _load_pair(
        paths["gdn_fit"], paths["gdn_heldout"], GDN_BUILD_SPEC
    )
    full_fit, full_heldout = _load_pair(
        paths["full_fit"], paths["full_heldout"], FULL_ATTENTION_BUILD_SPEC
    )
    reference_model = gdn_fit["model"]
    if any(
        payload["model"] != reference_model
        for payload in (gdn_heldout, full_fit, full_heldout)
    ):
        raise ValueError("GDN and full-attention moments belong to different models")
    if Path(reference_model["source"]).resolve() != model_path:
        raise ValueError("activation moments belong to another model path")

    fit_banks = {
        "linear_attention": _by_layer(gdn_fit),
        "full_attention": _by_layer(full_fit),
    }
    heldout_banks = {
        "linear_attention": _by_layer(gdn_heldout),
        "full_attention": _by_layer(full_heldout),
    }
    specs = {
        "linear_attention": GDN_BUILD_SPEC,
        "full_attention": FULL_ATTENTION_BUILD_SPEC,
    }
    weight_map = _weight_map(model_path)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Qwen3.5 global AA-SVD checkpoint build requires CUDA")
    cuda_index = device.index if device.index is not None else 0
    torch.cuda.set_device(cuda_index)
    factor_dtype = _dtype(args.factor_dtype)
    started = time.perf_counter()
    output_layers = []
    peak_cuda = 0
    for ordinal, layer_index in enumerate(range(NUM_LAYERS), start=1):
        block_type = (
            "linear_attention" if layer_index in GDN_LAYERS else "full_attention"
        )
        spec = specs[block_type]
        tensor_name = spec.tensor_name_template.format(layer_index=layer_index)
        shard_name = weight_map.get(tensor_name)
        if shard_name is None:
            raise KeyError(tensor_name)
        fit_payload = fit_banks[block_type][layer_index]["moments"]
        heldout_payload = heldout_banks[block_type][layer_index]["moments"]
        fit_rows = int(fit_payload["rows"])
        heldout_rows = int(heldout_payload["rows"])
        torch.cuda.reset_peak_memory_stats(cuda_index)
        layer_started = time.perf_counter()
        with safe_open(
            str(model_path / shard_name), framework="pt", device="cpu"
        ) as handle:
            weight = handle.get_tensor(tensor_name).to(device)
        fit_second_moment = fit_payload["gram"].to(device) / float(fit_rows)
        heldout_second_moment = (
            heldout_payload["gram"].to(device) / float(heldout_rows)
        )
        print(
            f"[Q35 Global AA-SVD] layer={layer_index} {ordinal}/{NUM_LAYERS} "
            f"type={block_type} phase=fit",
            flush=True,
        )
        fitted = fit_global_activation_aware_svd(
            weight,
            fit_second_moment,
            heldout_second_moment,
            rank=args.rank,
            tp_size=args.tp,
            covariance_damping=args.covariance_damping,
            factor_dtype=factor_dtype,
        )
        layer_peak = int(torch.cuda.max_memory_allocated(cuda_index))
        peak_cuda = max(peak_cuda, layer_peak)
        output_layers.append(
            {
                "layer_index": layer_index,
                "block_type": block_type,
                "tensor_name": tensor_name,
                "fit_rows": fit_rows,
                "heldout_rows": heldout_rows,
                "input_factor": fitted.input_factor,
                "output_basis": fitted.output_basis,
                "singular_values": fitted.singular_values,
                "metrics": fitted.metrics,
                "elapsed_seconds": time.perf_counter() - layer_started,
                "peak_cuda_allocated_bytes": layer_peak,
            }
        )
        print(
            f"[Q35 Global AA-SVD] layer={layer_index} complete "
            f"fit={fitted.metrics['fit_raw_relative_output_mse']:.8g} "
            f"heldout={fitted.metrics['heldout_relative_output_mse']:.8g} "
            f"quantized_heldout="
            f"{fitted.metrics['quantized_heldout_relative_output_mse']:.8g} "
            f"seconds={time.perf_counter() - layer_started:.3f}",
            flush=True,
        )
        del weight, fit_second_moment, heldout_second_moment, fitted
        torch.cuda.empty_cache()

    elapsed = time.perf_counter() - started
    result = {
        "format": FACTOR_FORMAT,
        "schema_version": 1,
        "command": shlex.join(sys.argv),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed,
        "model_path": str(model_path),
        "model_commit": reference_model["commit_hash"],
        "moments": {key: str(path) for key, path in paths.items()},
        "fit_collections": {
            "gdn": gdn_fit["collection"],
            "full_attention": full_fit["collection"],
        },
        "heldout_collections": {
            "gdn": gdn_heldout["collection"],
            "full_attention": full_heldout["collection"],
        },
        "tp_size": args.tp,
        "global_rank": args.rank,
        "hidden_size": HIDDEN_SIZE,
        "factor_dtype": args.factor_dtype,
        "covariance_damping": args.covariance_damping,
        "collective": "common_code_allreduce",
        "communication": {
            "fraction_of_dense_allreduce": args.rank / HIDDEN_SIZE,
            "reduction_fraction": 1.0 - args.rank / HIDDEN_SIZE,
            "dense_ring_elements_per_token_per_rank": (
                2.0 * (args.tp - 1) / args.tp * HIDDEN_SIZE
            ),
            "compressed_ring_elements_per_token_per_rank": (
                2.0 * (args.tp - 1) / args.tp * args.rank
            ),
        },
        "layers": output_layers,
        "environment": {
            "conda": os.environ.get("CONDA_DEFAULT_ENV"),
            "python": sys.version,
            "torch": str(torch.__version__),
            "cuda_device": torch.cuda.get_device_name(cuda_index),
            "peak_cuda_allocated_bytes": peak_cuda,
            "torch_num_threads": torch.get_num_threads(),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(result, temporary)
    os.replace(temporary, output_path)
    print(
        f"[Q35 Global AA-SVD] complete layers={len(output_layers)} "
        f"output={output_path} seconds={elapsed:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
