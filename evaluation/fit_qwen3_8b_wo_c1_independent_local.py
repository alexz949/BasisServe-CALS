#!/usr/bin/env python3
"""Compare independent-local and joint-final-output Wo-C1 fitting on Qwen3-8B."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
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

from safetensors.torch import load_file, save_file
import torch
from torch import Tensor


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basisserve.core.gqa_routed_ov_joint import (  # noqa: E402
    RoutedOVQuadratic,
    covariance_with_trace_damping,
    evaluate_quadratic,
    initialize_group_pooled_routed_svd,
    mask_routed_covariance,
    quadratic_from_target,
    solve_free_decoder,
)
from basisserve.core.tp_source_wo_c1 import (  # noqa: E402
    TPSourceWOLayout,
    covariance_to_source_blocks,
    weight_to_source_targets,
)


FORMAT = "basisserve.qwen3_8b.wo_c1_independent_local.v1"
PHASE1_FORMAT = "basisserve.qwen3_8b.wo_c1_lr_ar_phase1.v2"
COVARIANCE_FORMAT = "basisserve.attention_o_proj_covariances.v1"
TP_SIZE = 4
HIDDEN_SIZE = 4096
SOURCE_RANK = 512


@dataclass(frozen=True)
class IndependentLocalFit:
    source_encoders: Tensor
    source_decoders: Tensor
    metrics: dict[str, float]
    diagnostics: dict[str, Any]


def _atomic_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_safetensors(path: Path, tensors: Mapping[str, Tensor]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    save_file(dict(tensors), str(temporary))
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


def _parse_layers(raw: str, *, num_layers: int) -> tuple[int, ...]:
    if raw.strip().lower() == "all":
        return tuple(range(num_layers))
    selected = tuple(
        sorted({int(piece.strip()) for piece in raw.split(",") if piece.strip()})
    )
    if not selected or min(selected) < 0 or max(selected) >= num_layers:
        raise ValueError("selected layers are outside the checkpoint")
    return selected


def _validated_inputs(
    phase1_dir: Path,
    covariance_dir: Path,
    *,
    expected_source_rank: int,
    expected_joint_sweeps: int,
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
        raise ValueError("covariance manifest does not match joint Phase-1")
    signature = phase1["method"]["run_signature"]
    if int(signature["tp_size"]) != TP_SIZE:
        raise ValueError("comparison requires a TP4 checkpoint")
    if int(signature["source_rank"]) != expected_source_rank:
        raise ValueError(
            f"checkpoint source rank is {signature['source_rank']}; "
            f"expected {expected_source_rank}"
        )
    observed_sweeps = int(signature["c1_fit_config"]["encoder_sweeps"])
    if observed_sweeps != expected_joint_sweeps:
        raise ValueError(
            f"joint checkpoint has {observed_sweeps} sweeps; "
            f"expected {expected_joint_sweeps}"
        )
    layers = sorted(phase1["layers"], key=lambda row: int(row["layer"]))
    if [int(row["layer"]) for row in layers] != list(range(36)):
        raise ValueError("joint Phase-1 must contain all 36 ordered layers")
    phase1["layers"] = layers
    return phase1, covariance


def _objectives(
    weight: Tensor,
    fit_covariance: Tensor,
    heldout_covariance: Tensor,
    layout: TPSourceWOLayout,
    *,
    covariance_damping: float,
) -> tuple[dict[str, RoutedOVQuadratic], Tensor, Tensor, float]:
    fit_blocks = covariance_to_source_blocks(
        fit_covariance, layout, device=weight.device, dtype=weight.dtype
    )
    heldout_blocks = covariance_to_source_blocks(
        heldout_covariance, layout, device=weight.device, dtype=weight.dtype
    )
    fit_damped, absolute_damping = covariance_with_trace_damping(
        fit_blocks,
        relative_damping=covariance_damping,
    )
    mapping = torch.arange(layout.tp_size, device=weight.device, dtype=torch.long)
    target = weight_to_source_targets(
        weight, layout, device=weight.device, dtype=weight.dtype
    )
    covariances = {
        "local_fit_damped": mask_routed_covariance(
            fit_damped, head_to_kv_group=mapping, mode="diagonal"
        ),
        "local_heldout": mask_routed_covariance(
            heldout_blocks, head_to_kv_group=mapping, mode="diagonal"
        ),
        "final_fit_damped": fit_damped,
        "final_heldout": heldout_blocks,
    }
    objectives = {
        name: quadratic_from_target(
            covariance=covariance,
            target=target,
            name=name,
            trace_normalize=False,
        )
        for name, covariance in covariances.items()
    }
    return objectives, target, mapping, absolute_damping


def _evaluate_factors(
    objectives: Mapping[str, RoutedOVQuadratic],
    source_encoders: Tensor,
    source_decoders: Tensor,
    mapping: Tensor,
) -> dict[str, float]:
    metrics = {}
    for name, objective in objectives.items():
        relative_mse = float(
            evaluate_quadratic(
                objective,
                source_encoders,
                source_decoders,
                mapping,
            )
        ) / max(abs(float(objective.constant)), 1.0e-300)
        metrics[f"{name}_relative_mse"] = relative_mse
        metrics[f"{name}_relative_l2"] = math.sqrt(max(relative_mse, 0.0))
    return metrics


@torch.no_grad()
def fit_independent_local(
    objectives: Mapping[str, RoutedOVQuadratic],
    target: Tensor,
    mapping: Tensor,
    *,
    source_rank: int,
) -> IndependentLocalFit:
    """Fit the exact sourcewise activation-aware rank-constrained optimum."""

    local_objective = objectives["local_fit_damped"]
    initialization = initialize_group_pooled_routed_svd(
        covariance=local_objective.covariance,
        target=target,
        head_to_kv_group=mapping,
        group_ranks=(source_rank,) * int(mapping.numel()),
        covariance_ridge=0.0,
    )
    encoders = initialization.A_unique
    decoders, solve = solve_free_decoder(
        objective=local_objective,
        A_unique=encoders,
        head_to_kv_group=mapping,
        coupling_mode="diagonal",
        relative_jitter=0.0,
        linear_solve_dtype=encoders.dtype,
    )
    metrics = _evaluate_factors(objectives, encoders, decoders, mapping)
    return IndependentLocalFit(
        source_encoders=encoders,
        source_decoders=decoders,
        metrics=metrics,
        diagnostics={
            "method": "independent_source_activation_aware_svd_plus_exact_decoder",
            "encoder_initialization": [asdict(row) for row in initialization.groups],
            "decoder_solve": asdict(solve),
        },
    )


def _mean(records: Sequence[Mapping[str, Any]], method: str, metric: str) -> float:
    return statistics.fmean(float(record[method][metric]) for record in records)


def _aggregate(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metric_names = (
        "local_fit_damped_relative_mse",
        "local_heldout_relative_mse",
        "final_fit_damped_relative_mse",
        "final_heldout_relative_mse",
    )
    aggregate: dict[str, Any] = {
        "layers": len(records),
        "local_fit_wins_local_objective_layers": sum(
            float(record["independent_local_bfloat16"]["local_fit_damped_relative_mse"])
            <= float(record["joint_bfloat16"]["local_fit_damped_relative_mse"])
            for record in records
        ),
        "joint_fit_wins_final_objective_layers": sum(
            float(record["joint_bfloat16"]["final_fit_damped_relative_mse"])
            <= float(
                record["independent_local_bfloat16"]["final_fit_damped_relative_mse"]
            )
            for record in records
        ),
        "joint_fit_wins_final_heldout_layers": sum(
            float(record["joint_bfloat16"]["final_heldout_relative_mse"])
            <= float(record["independent_local_bfloat16"]["final_heldout_relative_mse"])
            for record in records
        ),
    }
    for method in (
        "independent_local_exact",
        "independent_local_bfloat16",
        "joint_bfloat16",
    ):
        aggregate[method] = {
            f"mean_{metric}": _mean(records, method, metric) for metric in metric_names
        }
    local_heldout_advantages = [
        float(record["joint_bfloat16"]["local_heldout_relative_mse"])
        - float(record["independent_local_bfloat16"]["local_heldout_relative_mse"])
        for record in records
    ]
    joint_heldout_advantages = [
        float(record["independent_local_bfloat16"]["final_heldout_relative_mse"])
        - float(record["joint_bfloat16"]["final_heldout_relative_mse"])
        for record in records
    ]
    aggregate["independent_local_advantage_on_local_heldout_mse"] = {
        "mean": statistics.fmean(local_heldout_advantages),
        "median": statistics.median(local_heldout_advantages),
        "minimum": min(local_heldout_advantages),
        "maximum": max(local_heldout_advantages),
    }
    aggregate["joint_advantage_on_final_heldout_mse"] = {
        "mean": statistics.fmean(joint_heldout_advantages),
        "median": statistics.median(joint_heldout_advantages),
        "minimum": min(joint_heldout_advantages),
        "maximum": max(joint_heldout_advantages),
    }
    return aggregate


def _summary_markdown(payload: Mapping[str, Any]) -> str:
    aggregate = payload["aggregate"]
    local = aggregate["independent_local_bfloat16"]
    joint = aggregate["joint_bfloat16"]
    lines = [
        "# Qwen3-8B Wo-C1 independent-local versus joint-final-output fitting",
        "",
        (
            "Both methods use TP4, source rank 512, identical C4 covariance "
            "statistics, and stored BF16 factors. Independent-local optimizes "
            "only diagonal source objectives; joint reuses the existing "
            f"{payload['protocol']['joint_encoder_sweeps']}-sweep full-layer "
            "C1 checkpoint."
        ),
        "",
        "## Aggregate",
        "",
        f"- Layers: `{aggregate['layers']}`",
        (
            "- Independent-local wins its local fit objective: "
            f"`{aggregate['local_fit_wins_local_objective_layers']}/{aggregate['layers']}` layers"
        ),
        (
            "- Joint wins the final fit objective: "
            f"`{aggregate['joint_fit_wins_final_objective_layers']}/{aggregate['layers']}` layers"
        ),
        (
            "- Joint wins final heldout output: "
            f"`{aggregate['joint_fit_wins_final_heldout_layers']}/{aggregate['layers']}` layers"
        ),
        "",
        "| BF16 method | Mean local heldout MSE | Mean final heldout MSE |",
        "|---|---:|---:|",
        (
            "| Independent local | "
            f"{local['mean_local_heldout_relative_mse']:.8g} | "
            f"{local['mean_final_heldout_relative_mse']:.8g} |"
        ),
        (
            "| Joint final-output | "
            f"{joint['mean_local_heldout_relative_mse']:.8g} | "
            f"{joint['mean_final_heldout_relative_mse']:.8g} |"
        ),
        "",
        "## Per-layer BF16 results",
        "",
        "| Layer | Local-fit local heldout | Joint local heldout | Local-fit final heldout | Joint final heldout | Joint final advantage |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for record in payload["layers"]:
        local_row = record["independent_local_bfloat16"]
        joint_row = record["joint_bfloat16"]
        advantage = float(local_row["final_heldout_relative_mse"]) - float(
            joint_row["final_heldout_relative_mse"]
        )
        lines.append(
            f"| {record['layer']} | "
            f"{local_row['local_heldout_relative_mse']:.8g} | "
            f"{joint_row['local_heldout_relative_mse']:.8g} | "
            f"{local_row['final_heldout_relative_mse']:.8g} | "
            f"{joint_row['final_heldout_relative_mse']:.8g} | "
            f"{advantage:.8g} |"
        )
    lines.extend(
        [
            "",
            "These are attention-layer output errors, not terminal-logit KL or PPL.",
            "",
            "## Collective interpretation",
            "",
            (
                "For either factor bank, `AllGather private latent -> joint "
                "decoder` and `local private decoder -> hidden AllReduce` "
                "compute the same approximate function. At TP4/rank512/BF16, "
                "their ideal ring traffic is respectively 3072 and 12288 "
                "bytes per activation row per rank."
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
    parser.add_argument("--source-rank", type=int, default=SOURCE_RANK)
    parser.add_argument("--covariance-damping", type=float, default=1.0e-5)
    parser.add_argument("--expected-joint-sweeps", type=int, default=20)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--work-dtype", choices=("float32", "float64"), default="float64"
    )
    parser.add_argument("--factor-dtype", choices=("bfloat16",), default="bfloat16")
    parser.add_argument("--torch-num-threads", type=int, default=4)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    source_width = HIDDEN_SIZE // TP_SIZE
    if not 0 < args.source_rank <= source_width:
        raise ValueError(f"source rank must lie in [1, {source_width}]")
    if args.covariance_damping < 0:
        raise ValueError("covariance damping must be nonnegative")
    if args.expected_joint_sweeps <= 0:
        raise ValueError("expected joint sweeps must be positive")
    torch.set_num_threads(args.torch_num_threads)
    phase1_dir = Path(args.phase1_dir).expanduser().resolve()
    covariance_dir = Path(args.covariance_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    phase1, covariance_manifest = _validated_inputs(
        phase1_dir,
        covariance_dir,
        expected_source_rank=args.source_rank,
        expected_joint_sweeps=args.expected_joint_sweeps,
    )
    selected_layers = _parse_layers(args.layers, num_layers=36)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the real-model joint/local comparison requires CUDA")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    work_dtype = {"float32": torch.float32, "float64": torch.float64}[args.work_dtype]
    factor_dtype = torch.bfloat16
    layout = TPSourceWOLayout(
        input_width=HIDDEN_SIZE,
        output_width=HIDDEN_SIZE,
        tp_size=TP_SIZE,
        source_rank=args.source_rank,
        dtype_bytes=2,
    )
    phase1_by_layer = {int(row["layer"]): row for row in phase1["layers"]}
    covariance_artifacts = covariance_manifest["artifacts"]
    output_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    records = []
    for ordinal, layer in enumerate(selected_layers, start=1):
        phase_record = phase1_by_layer[layer]
        joint_path = phase1_dir / phase_record["artifact"]["file"]
        if _sha256(joint_path) != phase_record["artifact"]["sha256"]:
            raise RuntimeError(f"joint factor hash mismatch at layer {layer}")
        covariance_record = covariance_artifacts[str(layer)]
        covariance_path = covariance_dir / covariance_record["file"]
        if _sha256(covariance_path) != covariance_record["sha256"]:
            raise RuntimeError(f"covariance hash mismatch at layer {layer}")
        joint_payload = load_file(str(joint_path), device="cpu")
        sufficient = load_file(str(covariance_path), device="cpu")
        weight = sufficient["weight"].to(device=device, dtype=work_dtype)
        fit_covariance = sufficient["fit_covariance"].to(
            device=device, dtype=work_dtype
        )
        heldout_covariance = sufficient["heldout_covariance"].to(
            device=device, dtype=work_dtype
        )
        objectives, target, mapping, absolute_damping = _objectives(
            weight,
            fit_covariance,
            heldout_covariance,
            layout,
            covariance_damping=args.covariance_damping,
        )
        local = fit_independent_local(
            objectives,
            target,
            mapping,
            source_rank=args.source_rank,
        )
        stored_local_encoders = local.source_encoders.to(dtype=factor_dtype)
        stored_local_decoders = local.source_decoders.to(dtype=factor_dtype)
        local_bfloat16 = _evaluate_factors(
            objectives,
            stored_local_encoders.to(dtype=work_dtype),
            stored_local_decoders.to(dtype=work_dtype),
            mapping,
        )
        joint_encoders = joint_payload["c1_source_encoders"].to(
            device=device, dtype=work_dtype
        )
        joint_decoders = joint_payload["c1_source_decoders"].to(
            device=device, dtype=work_dtype
        )
        joint_bfloat16 = _evaluate_factors(
            objectives,
            joint_encoders,
            joint_decoders,
            mapping,
        )
        artifact_path = output_dir / f"layer_{layer:03d}.safetensors"
        _atomic_safetensors(
            artifact_path,
            {
                "c1_source_encoders": stored_local_encoders.cpu().contiguous(),
                "c1_source_decoders": stored_local_decoders.cpu().contiguous(),
            },
        )
        local_wins = (
            local_bfloat16["local_fit_damped_relative_mse"]
            <= joint_bfloat16["local_fit_damped_relative_mse"]
        )
        joint_wins = (
            joint_bfloat16["final_fit_damped_relative_mse"]
            <= local_bfloat16["final_fit_damped_relative_mse"]
        )
        record = {
            "layer": layer,
            "independent_local_exact": local.metrics,
            "independent_local_bfloat16": local_bfloat16,
            "joint_bfloat16": joint_bfloat16,
            "joint_exact_reference": {
                "final_fit_damped_relative_mse": float(
                    phase_record["c1_allgather"]["exact_fit_damped_relative_mse"]
                ),
                "final_heldout_relative_mse": float(
                    phase_record["c1_allgather"]["exact_heldout_relative_mse"]
                ),
                "selected_sweep": int(phase_record["c1_allgather"]["selected_sweep"]),
            },
            "checks": {
                "independent_local_bfloat16_wins_local_fit": local_wins,
                "joint_bfloat16_wins_final_fit": joint_wins,
            },
            "solver": local.diagnostics,
            "numerics": {
                "absolute_covariance_damping": absolute_damping,
            },
            "artifact": {
                "file": artifact_path.name,
                "sha256": _sha256(artifact_path),
                "tensors": {
                    "c1_source_encoders": {
                        "shape": list(stored_local_encoders.shape),
                        "dtype": str(stored_local_encoders.dtype),
                    },
                    "c1_source_decoders": {
                        "shape": list(stored_local_decoders.shape),
                        "dtype": str(stored_local_decoders.dtype),
                    },
                },
            },
        }
        records.append(record)
        print(
            json.dumps(
                {
                    "event": "layer_complete",
                    "layer": layer,
                    "progress": f"{ordinal}/{len(selected_layers)}",
                    "local_local_heldout": local_bfloat16["local_heldout_relative_mse"],
                    "joint_local_heldout": joint_bfloat16["local_heldout_relative_mse"],
                    "local_final_heldout": local_bfloat16["final_heldout_relative_mse"],
                    "joint_final_heldout": joint_bfloat16["final_heldout_relative_mse"],
                    "local_wins_local_fit": local_wins,
                    "joint_wins_final_fit": joint_wins,
                }
            ),
            flush=True,
        )
        del joint_payload, sufficient, weight, fit_covariance, heldout_covariance
        del objectives, target, mapping, local, stored_local_encoders
        del stored_local_decoders, joint_encoders, joint_decoders
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
            "work_dtype": args.work_dtype,
            "factor_dtype": args.factor_dtype,
            "torch_num_threads": args.torch_num_threads,
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        },
        "model": phase1["model"],
        "source": {
            "joint_phase1_dir": str(phase1_dir),
            "joint_phase1_results_sha256": _sha256(phase1_dir / "results.json"),
            "covariance_dir": str(covariance_dir),
            "covariance_manifest_sha256": _sha256(covariance_dir / "manifest.json"),
        },
        "protocol": {
            "tp_size": TP_SIZE,
            "source_rank": args.source_rank,
            "layers": list(selected_layers),
            "fit_windows": 256,
            "heldout_windows": 64,
            "positions_per_window": 2048,
            "covariance_damping": args.covariance_damping,
            "independent_local": (
                "diagonal source covariance plus exact activation-aware "
                f"rank-{args.source_rank} fit"
            ),
            "joint": "existing full-source covariance decoder-closed C1",
            "joint_encoder_sweeps": args.expected_joint_sweeps,
            "quality_scope": "attention layer output; dense V and dense KV cache",
            "c1_allgather_ideal_ring_bytes_per_rank_per_row": (
                layout.compressed_allgather_ring_bytes_per_rank
            ),
            "local_decode_allreduce_ideal_ring_bytes_per_rank_per_row": 12288,
            "collective_functions_are_algebraically_equal_for_fixed_factors": True,
        },
        "aggregate": _aggregate(records),
        "layers": records,
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
