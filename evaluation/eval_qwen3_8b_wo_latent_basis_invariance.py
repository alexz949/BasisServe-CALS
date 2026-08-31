#!/usr/bin/env python3
"""Test real Qwen3-8B Wo-C1 invariance to private latent rotations."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

from safetensors.torch import load_file
import torch
from torch import Tensor


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basisserve.analysis.c1_rigorous import rotate_private_factors  # noqa: E402


FORMAT = "basisserve.qwen3_8b.wo_c1_latent_basis_invariance.v1"
PHASE1_FORMAT = "basisserve.qwen3_8b.wo_c1_lr_ar_phase1.v2"
COVARIANCE_FORMAT = "basisserve.attention_o_proj_covariances.v1"
TP_SIZE = 4
SOURCE_WIDTH = 1024
SOURCE_RANK = 512
HIDDEN_SIZE = 4096


def _atomic_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str | None:
    completed = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return completed.stdout.strip() or None


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_integer_list(raw: str) -> tuple[int, ...]:
    values = tuple(dict.fromkeys(int(piece.strip()) for piece in raw.split(",") if piece.strip()))
    if not values:
        raise ValueError("at least one integer is required")
    return values


def _parse_layers(raw: str, *, num_layers: int) -> tuple[int, ...]:
    if raw.strip().lower() == "all":
        return tuple(range(num_layers))
    selected = tuple(sorted(set(_parse_integer_list(raw))))
    if min(selected) < 0 or max(selected) >= num_layers:
        raise ValueError("selected layers are outside the checkpoint")
    return selected


def _validated_inputs(
    phase1_dir: Path,
    covariance_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    phase1_path = phase1_dir / "results.json"
    covariance_path = covariance_dir / "manifest.json"
    phase1 = _load_json(phase1_path)
    covariance = _load_json(covariance_path)
    if phase1.get("format") != PHASE1_FORMAT or phase1.get("status") != "complete":
        raise ValueError("Phase-1 input is not complete")
    if covariance.get("format") != COVARIANCE_FORMAT:
        raise ValueError("unexpected covariance manifest format")
    if _sha256(covariance_path) != phase1["source"]["covariance_manifest_sha256"]:
        raise ValueError("covariance manifest does not match Phase-1")
    signature = phase1["method"]["run_signature"]
    if int(signature["tp_size"]) != TP_SIZE or int(signature["source_rank"]) != SOURCE_RANK:
        raise ValueError("latent invariance requires the TP4 source-rank-512 checkpoint")
    layers = sorted(phase1["layers"], key=lambda row: int(row["layer"]))
    if [int(row["layer"]) for row in layers] != list(range(36)):
        raise ValueError("Phase-1 must contain all 36 ordered layers")
    phase1["layers"] = layers
    return phase1, covariance


def _orthogonal_rotations(
    *,
    layer: int,
    seed: int,
    device: torch.device,
) -> tuple[Tensor, ...]:
    rotations = []
    for source in range(TP_SIZE):
        generator = torch.Generator(device=device)
        generator.manual_seed(seed + 104729 * layer + 1009 * source)
        sample = torch.randn(
            SOURCE_RANK,
            SOURCE_RANK,
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
        rotation, upper = torch.linalg.qr(sample)
        signs = torch.diagonal(upper).sign()
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)
        rotations.append((rotation * signs.unsqueeze(0)).contiguous())
    return tuple(rotations)


def _fold_products(encoders: Sequence[Tensor], decoders: Sequence[Tensor]) -> Tensor:
    products = torch.bmm(torch.stack(tuple(encoders)), torch.stack(tuple(decoders)))
    return products.reshape(HIDDEN_SIZE, HIDDEN_SIZE).transpose(0, 1).contiguous()


def _covariance_relative_l2(
    delta_weight: Tensor,
    reference_weight: Tensor,
    covariance: Tensor,
) -> float:
    numerator = ((delta_weight @ covariance) * delta_weight).sum().clamp_min(0)
    denominator = ((reference_weight @ covariance) * reference_weight).sum().clamp_min(
        torch.finfo(reference_weight.dtype).tiny
    )
    return math.sqrt(float(numerator / denominator))


def _bf16_deployment_output(
    activations: Tensor,
    encoders: Sequence[Tensor],
    decoders: Sequence[Tensor],
) -> Tensor:
    source_inputs = activations.reshape(
        activations.shape[0], TP_SIZE, SOURCE_WIDTH
    ).permute(1, 0, 2).contiguous()
    latent_by_source = torch.bmm(source_inputs, torch.stack(tuple(encoders)))
    gathered = latent_by_source.permute(1, 0, 2).reshape(
        activations.shape[0], TP_SIZE * SOURCE_RANK
    )
    joint_decoder = torch.stack(tuple(decoders)).reshape(
        TP_SIZE * SOURCE_RANK, HIDDEN_SIZE
    )
    return gathered @ joint_decoder


def _summarize(values: Sequence[float]) -> dict[str, float]:
    return {
        "minimum": min(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "maximum": max(values),
    }


def _aggregate(layers: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    trials = [trial for layer in layers for trial in layer["trials"]]
    metric_names = (
        "rotation_orthogonality_max_abs",
        "fp32_product_relative_l2",
        "fp32_product_max_abs",
        "fp32_heldout_output_relative_l2",
        "bf16_probe_output_relative_l2",
        "bf16_probe_output_max_abs",
    )
    return {
        "layers": len(layers),
        "trials": len(trials),
        **{
            name: _summarize([float(trial[name]) for trial in trials])
            for name in metric_names
        },
    }


def _summary_markdown(payload: Mapping[str, Any]) -> str:
    aggregate = payload["aggregate"]
    lines = [
        "# Qwen3-8B Wo-C1 latent-basis invariance",
        "",
        (
            "Each source-private latent basis is independently rotated as "
            "`E'_p = E_p Q_p`, `D'_p = Q_p^T D_p`."
        ),
        "",
        "## Aggregate",
        "",
        f"- Layers: `{aggregate['layers']}`; rotation trials: `{aggregate['trials']}`",
        (
            "- Maximum FP32 product relative L2: "
            f"`{aggregate['fp32_product_relative_l2']['maximum']:.8g}`"
        ),
        (
            "- Maximum FP32 heldout-output relative L2: "
            f"`{aggregate['fp32_heldout_output_relative_l2']['maximum']:.8g}`"
        ),
        (
            "- Median BF16 deployment-probe relative L2 after factor requantization: "
            f"`{aggregate['bf16_probe_output_relative_l2']['median']:.8g}`"
        ),
        (
            "- Maximum BF16 deployment-probe relative L2 after factor requantization: "
            f"`{aggregate['bf16_probe_output_relative_l2']['maximum']:.8g}`"
        ),
        "",
        "| Layer | FP32 product rel-L2 max | FP32 heldout rel-L2 max | BF16 probe rel-L2 median | BF16 probe rel-L2 max |",
        "|---:|---:|---:|---:|---:|",
    ]
    for layer in payload["layers"]:
        trials = layer["trials"]
        lines.append(
            f"| {layer['layer']} | "
            f"{max(float(row['fp32_product_relative_l2']) for row in trials):.8g} | "
            f"{max(float(row['fp32_heldout_output_relative_l2']) for row in trials):.8g} | "
            f"{statistics.median(float(row['bf16_probe_output_relative_l2']) for row in trials):.8g} | "
            f"{max(float(row['bf16_probe_output_relative_l2']) for row in trials):.8g} |"
        )
    lines.extend(
        [
            "",
            "FP32 uses the complete heldout covariance and tests the same linear operator.",
            (
                "BF16 uses the deployed two-GEMM ordering with deterministic "
                "RMS-scaled probes. Its difference includes factor "
                "requantization and floating-point operation ordering, so it "
                "is a numerical gauge-sensitivity result, not representational error."
            ),
            "",
            "## Command",
            "",
            "```bash",
            payload["command"],
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase1-dir", required=True)
    parser.add_argument("--covariance-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--layers", default="all")
    parser.add_argument("--seeds", default="20260828,20260829,20260830")
    parser.add_argument("--bf16-probe-rows", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--torch-num-threads", type=int, default=4)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if args.bf16_probe_rows <= 0:
        raise ValueError("BF16 probe rows must be positive")
    torch.set_num_threads(args.torch_num_threads)
    torch.set_float32_matmul_precision("highest")
    phase1_dir = Path(args.phase1_dir).expanduser().resolve()
    covariance_dir = Path(args.covariance_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    phase1, covariance_manifest = _validated_inputs(phase1_dir, covariance_dir)
    selected_layers = _parse_layers(args.layers, num_layers=36)
    seeds = _parse_integer_list(args.seeds)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the real-model latent invariance test requires CUDA")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    output_dir.mkdir(parents=True, exist_ok=True)

    phase1_by_layer = {int(row["layer"]): row for row in phase1["layers"]}
    covariance_artifacts = covariance_manifest["artifacts"]
    started = time.perf_counter()
    layer_records = []
    for ordinal, layer in enumerate(selected_layers, start=1):
        phase_record = phase1_by_layer[layer]
        factor_path = phase1_dir / phase_record["artifact"]["file"]
        if _sha256(factor_path) != phase_record["artifact"]["sha256"]:
            raise RuntimeError(f"factor hash mismatch at layer {layer}")
        covariance_record = covariance_artifacts[str(layer)]
        covariance_path = covariance_dir / covariance_record["file"]
        if _sha256(covariance_path) != covariance_record["sha256"]:
            raise RuntimeError(f"covariance hash mismatch at layer {layer}")
        factors = load_file(str(factor_path), device="cpu")
        sufficient = load_file(str(covariance_path), device="cpu")
        stored_encoders = factors["c1_source_encoders"].to(device=device)
        stored_decoders = factors["c1_source_decoders"].to(device=device)
        if stored_encoders.dtype != torch.bfloat16 or stored_decoders.dtype != torch.bfloat16:
            raise TypeError("expected stored BF16 C1 factors")
        fp32_encoders = tuple(row.float() for row in stored_encoders)
        fp32_decoders = tuple(row.float() for row in stored_decoders)
        reference_weight = _fold_products(fp32_encoders, fp32_decoders)
        covariance = sufficient["heldout_covariance"].to(
            device=device, dtype=torch.float32
        )
        covariance = 0.5 * (covariance + covariance.transpose(0, 1))
        rms = covariance.diagonal().clamp_min(0).sqrt()
        probe_generator = torch.Generator(device=device)
        probe_generator.manual_seed(7919 + layer)
        probe = torch.randn(
            args.bf16_probe_rows,
            HIDDEN_SIZE,
            generator=probe_generator,
            device=device,
            dtype=torch.float32,
        )
        probe = (probe * rms).to(dtype=torch.bfloat16)
        reference_bf16_output = _bf16_deployment_output(
            probe,
            tuple(stored_encoders),
            tuple(stored_decoders),
        )
        reference_bf16_norm = torch.linalg.vector_norm(
            reference_bf16_output.float()
        ).clamp_min(torch.finfo(torch.float32).tiny)

        trials = []
        for seed in seeds:
            rotations = _orthogonal_rotations(
                layer=layer,
                seed=seed,
                device=device,
            )
            rotated_encoders, rotated_decoders = rotate_private_factors(
                fp32_encoders,
                fp32_decoders,
                rotations,
            )
            rotated_weight = _fold_products(rotated_encoders, rotated_decoders)
            delta = rotated_weight - reference_weight
            fp32_product_relative_l2 = float(
                torch.linalg.vector_norm(delta)
                / torch.linalg.vector_norm(reference_weight).clamp_min(
                    torch.finfo(torch.float32).tiny
                )
            )
            fp32_heldout = _covariance_relative_l2(
                delta,
                reference_weight,
                covariance,
            )
            rotated_bf16_encoders = tuple(
                value.to(dtype=torch.bfloat16) for value in rotated_encoders
            )
            rotated_bf16_decoders = tuple(
                value.to(dtype=torch.bfloat16) for value in rotated_decoders
            )
            rotated_bf16_output = _bf16_deployment_output(
                probe,
                rotated_bf16_encoders,
                rotated_bf16_decoders,
            )
            bf16_delta = rotated_bf16_output.float() - reference_bf16_output.float()
            orthogonality = max(
                float(
                    (
                        rotation.transpose(0, 1) @ rotation
                        - torch.eye(SOURCE_RANK, device=device)
                    ).abs().max()
                )
                for rotation in rotations
            )
            trial = {
                "seed": seed,
                "rotation_orthogonality_max_abs": orthogonality,
                "fp32_product_relative_l2": fp32_product_relative_l2,
                "fp32_product_max_abs": float(delta.abs().max()),
                "fp32_heldout_output_relative_l2": fp32_heldout,
                "bf16_probe_output_relative_l2": float(
                    torch.linalg.vector_norm(bf16_delta) / reference_bf16_norm
                ),
                "bf16_probe_output_max_abs": float(bf16_delta.abs().max()),
            }
            if not all(math.isfinite(float(value)) for key, value in trial.items() if key != "seed"):
                raise FloatingPointError(f"non-finite rotation metric at layer {layer}")
            trials.append(trial)
            del rotations, rotated_encoders, rotated_decoders, rotated_weight, delta
            del rotated_bf16_encoders, rotated_bf16_decoders
            del rotated_bf16_output, bf16_delta

        layer_record = {
            "layer": layer,
            "factor_file": str(factor_path),
            "factor_sha256": phase_record["artifact"]["sha256"],
            "covariance_file": str(covariance_path),
            "covariance_sha256": covariance_record["sha256"],
            "trials": trials,
        }
        layer_records.append(layer_record)
        print(
            json.dumps(
                {
                    "event": "layer_complete",
                    "layer": layer,
                    "progress": f"{ordinal}/{len(selected_layers)}",
                    "fp32_heldout_relative_l2_max": max(
                        row["fp32_heldout_output_relative_l2"] for row in trials
                    ),
                    "bf16_probe_relative_l2_median": statistics.median(
                        row["bf16_probe_output_relative_l2"] for row in trials
                    ),
                }
            ),
            flush=True,
        )
        del factors, sufficient, stored_encoders, stored_decoders
        del fp32_encoders, fp32_decoders, reference_weight, covariance, rms
        del probe, reference_bf16_output, reference_bf16_norm
        torch.cuda.empty_cache()

    payload = {
        "format": FORMAT,
        "schema_version": 1,
        "status": "complete",
        "command": shlex.join((sys.executable, *sys.argv)),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "elapsed_seconds": time.perf_counter() - started,
        "environment": {
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "torch": torch.__version__,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device),
            "torch_num_threads": args.torch_num_threads,
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        },
        "model": phase1["model"],
        "source": {
            "phase1_dir": str(phase1_dir),
            "phase1_results_sha256": _sha256(phase1_dir / "results.json"),
            "covariance_dir": str(covariance_dir),
            "covariance_manifest_sha256": _sha256(covariance_dir / "manifest.json"),
        },
        "protocol": {
            "tp_size": TP_SIZE,
            "source_width": SOURCE_WIDTH,
            "source_rank": SOURCE_RANK,
            "layers": list(selected_layers),
            "seeds": list(seeds),
            "stored_factor_dtype": "bfloat16",
            "fp32_replay": "complete heldout covariance",
            "bf16_probe_rows": args.bf16_probe_rows,
            "bf16_probe_scaling": "per-coordinate heldout RMS",
            "bf16_path": "source encoder GEMM, ordered latent concatenate, joint decoder GEMM",
            "collective_numerics": "AllGather is a bitwise transport and is not executed",
        },
        "aggregate": _aggregate(layer_records),
        "layers": layer_records,
    }
    results_path = output_dir / "results.json"
    summary_path = output_dir / "summary.md"
    _atomic_text(results_path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _atomic_text(summary_path, _summary_markdown(payload))
    print(
        json.dumps(
            {
                "event": "result_written",
                "results": str(results_path),
                "results_sha256": _sha256(results_path),
                "summary": str(summary_path),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
