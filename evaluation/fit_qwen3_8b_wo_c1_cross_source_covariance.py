#!/usr/bin/env python3
"""Ablate cross-source covariance in fixed-encoder Qwen3-8B Wo-C1 decoders."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
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
    solve_free_decoder,
)
from basisserve.core.tp_source_wo_c1 import TPSourceWOLayout  # noqa: E402
from evaluation.fit_qwen3_8b_wo_c1_independent_local import (  # noqa: E402
    COVARIANCE_FORMAT,
    HIDDEN_SIZE,
    PHASE1_FORMAT,
    SOURCE_RANK,
    TP_SIZE,
    _evaluate_factors,
    _objectives,
)


FORMAT = "basisserve.qwen3_8b.wo_c1_cross_source_covariance.v1"
INDEPENDENT_LOCAL_FORMAT = "basisserve.qwen3_8b.wo_c1_independent_local.v1"


@dataclass(frozen=True)
class FixedEncoderDecoderRefits:
    full_decoders: Tensor
    block_diagonal_decoders: Tensor
    full_diagnostics: dict[str, Any]
    block_diagonal_diagnostics: dict[str, Any]


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


def _validate_layer_records(payload: Mapping[str, Any], *, label: str) -> None:
    layers = sorted(payload["layers"], key=lambda row: int(row["layer"]))
    if [int(row["layer"]) for row in layers] != list(range(36)):
        raise ValueError(f"{label} must contain all 36 ordered layers")


def _validated_inputs(
    phase1_dir: Path,
    independent_local_dir: Path,
    covariance_dir: Path,
    *,
    expected_joint_sweeps: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    phase1_path = phase1_dir / "results.json"
    independent_path = independent_local_dir / "results.json"
    covariance_path = covariance_dir / "manifest.json"
    phase1 = _load_json(phase1_path)
    independent = _load_json(independent_path)
    covariance = _load_json(covariance_path)
    if phase1.get("format") != PHASE1_FORMAT or phase1.get("status") != "complete":
        raise ValueError("joint Phase-1 input is not complete")
    if (
        independent.get("format") != INDEPENDENT_LOCAL_FORMAT
        or independent.get("status") != "complete"
    ):
        raise ValueError("independent-local input is not complete")
    if covariance.get("format") != COVARIANCE_FORMAT:
        raise ValueError("unexpected covariance manifest format")
    covariance_hash = _sha256(covariance_path)
    phase1_hash = _sha256(phase1_path)
    if covariance_hash != phase1["source"]["covariance_manifest_sha256"]:
        raise ValueError("covariance manifest does not match joint Phase-1")
    if covariance_hash != independent["source"]["covariance_manifest_sha256"]:
        raise ValueError("covariance manifest does not match independent-local")
    if phase1_hash != independent["source"]["joint_phase1_results_sha256"]:
        raise ValueError("independent-local input does not reference joint Phase-1")
    signature = phase1["method"]["run_signature"]
    if (
        int(signature["tp_size"]) != TP_SIZE
        or int(signature["source_rank"]) != SOURCE_RANK
    ):
        raise ValueError("Experiment J requires TP4 and source rank 512")
    observed_sweeps = int(signature["c1_fit_config"]["encoder_sweeps"])
    if observed_sweeps != expected_joint_sweeps:
        raise ValueError(
            f"joint checkpoint has {observed_sweeps} sweeps; "
            f"expected {expected_joint_sweeps}"
        )
    if float(signature["c1_fit_config"]["covariance_damping"]) != float(
        independent["protocol"]["covariance_damping"]
    ):
        raise ValueError("joint and independent-local covariance damping differ")
    _validate_layer_records(phase1, label="joint Phase-1")
    _validate_layer_records(independent, label="independent-local input")
    phase1["layers"] = sorted(phase1["layers"], key=lambda row: int(row["layer"]))
    independent["layers"] = sorted(
        independent["layers"], key=lambda row: int(row["layer"])
    )
    return phase1, independent, covariance


@torch.no_grad()
def refit_fixed_encoders(
    objectives: Mapping[str, RoutedOVQuadratic],
    fixed_encoders: Tensor,
    mapping: Tensor,
) -> FixedEncoderDecoderRefits:
    """Solve full and source-block-diagonal decoders for identical encoders."""

    full_decoders, full_solve = solve_free_decoder(
        objective=objectives["final_fit_damped"],
        A_unique=fixed_encoders,
        head_to_kv_group=mapping,
        coupling_mode="full_layer",
        relative_jitter=0.0,
        linear_solve_dtype=torch.float64,
    )
    block_decoders, block_solve = solve_free_decoder(
        objective=objectives["local_fit_damped"],
        A_unique=fixed_encoders,
        head_to_kv_group=mapping,
        coupling_mode="diagonal",
        relative_jitter=0.0,
        linear_solve_dtype=torch.float64,
    )
    return FixedEncoderDecoderRefits(
        full_decoders=full_decoders,
        block_diagonal_decoders=block_decoders,
        full_diagnostics=asdict(full_solve),
        block_diagonal_diagnostics=asdict(block_solve),
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
    methods = (
        "full_fixed_encoder_exact",
        "full_fixed_encoder_bfloat16",
        "block_diagonal_fixed_encoder_exact",
        "block_diagonal_fixed_encoder_bfloat16",
        "joint_stored_bfloat16",
        "independent_local_bfloat16",
    )
    aggregate: dict[str, Any] = {
        "layers": len(records),
        "block_diagonal_wins_local_fit_layers": sum(
            float(
                row["block_diagonal_fixed_encoder_bfloat16"][
                    "local_fit_damped_relative_mse"
                ]
            )
            <= float(
                row["full_fixed_encoder_bfloat16"]["local_fit_damped_relative_mse"]
            )
            for row in records
        ),
        "full_covariance_wins_final_fit_layers": sum(
            float(row["full_fixed_encoder_bfloat16"]["final_fit_damped_relative_mse"])
            <= float(
                row["block_diagonal_fixed_encoder_bfloat16"][
                    "final_fit_damped_relative_mse"
                ]
            )
            for row in records
        ),
        "full_covariance_wins_final_heldout_layers": sum(
            float(row["full_fixed_encoder_bfloat16"]["final_heldout_relative_mse"])
            <= float(
                row["block_diagonal_fixed_encoder_bfloat16"][
                    "final_heldout_relative_mse"
                ]
            )
            for row in records
        ),
    }
    for method in methods:
        aggregate[method] = {
            f"mean_{metric}": _mean(records, method, metric) for metric in metric_names
        }
    cross_source_gains = [
        float(
            row["block_diagonal_fixed_encoder_bfloat16"]["final_heldout_relative_mse"]
        )
        - float(row["full_fixed_encoder_bfloat16"]["final_heldout_relative_mse"])
        for row in records
    ]
    relative_cross_source_gains = [
        gain
        / max(
            float(
                row["block_diagonal_fixed_encoder_bfloat16"][
                    "final_heldout_relative_mse"
                ]
            ),
            1.0e-300,
        )
        for gain, row in zip(cross_source_gains, records)
    ]
    aggregate["full_covariance_advantage_on_final_heldout_mse"] = {
        "mean_absolute": statistics.fmean(cross_source_gains),
        "median_absolute": statistics.median(cross_source_gains),
        "minimum_absolute": min(cross_source_gains),
        "maximum_absolute": max(cross_source_gains),
        "mean_relative": statistics.fmean(relative_cross_source_gains),
        "median_relative": statistics.median(relative_cross_source_gains),
    }
    return aggregate


def _summary_markdown(payload: Mapping[str, Any]) -> str:
    aggregate = payload["aggregate"]
    full = aggregate["full_fixed_encoder_bfloat16"]
    block = aggregate["block_diagonal_fixed_encoder_bfloat16"]
    local = aggregate["independent_local_bfloat16"]
    gain = aggregate["full_covariance_advantage_on_final_heldout_mse"]
    lines = [
        "# Qwen3-8B Wo-C1 cross-source covariance ablation",
        "",
        (
            "The causal comparison fixes the stored BF16 encoders from the "
            f"joint {payload['protocol']['joint_encoder_sweeps']}-sweep checkpoint. "
            "Only the decoder normal equations change: the full arm keeps all "
            "cross-source covariance blocks, while the block-diagonal arm zeros "
            "every off-diagonal block."
        ),
        "",
        "## Aggregate",
        "",
        f"- Layers: `{aggregate['layers']}`",
        (
            "- Block-diagonal wins its local fit objective: "
            f"`{aggregate['block_diagonal_wins_local_fit_layers']}/{aggregate['layers']}` layers"
        ),
        (
            "- Full covariance wins the final fit objective: "
            f"`{aggregate['full_covariance_wins_final_fit_layers']}/{aggregate['layers']}` layers"
        ),
        (
            "- Full covariance wins final heldout output: "
            f"`{aggregate['full_covariance_wins_final_heldout_layers']}/{aggregate['layers']}` layers"
        ),
        (
            "- Mean relative final-heldout MSE reduction from cross-source covariance: "
            f"`{100.0 * gain['mean_relative']:.4f}%`"
        ),
        "",
        "| BF16 arm | Mean local heldout MSE | Mean final heldout MSE |",
        "|---|---:|---:|",
        (
            "| Fixed encoder + block-diagonal covariance | "
            f"{block['mean_local_heldout_relative_mse']:.8g} | "
            f"{block['mean_final_heldout_relative_mse']:.8g} |"
        ),
        (
            "| Fixed encoder + full covariance | "
            f"{full['mean_local_heldout_relative_mse']:.8g} | "
            f"{full['mean_final_heldout_relative_mse']:.8g} |"
        ),
        (
            "| Independent-local encoder and decoder (Experiment I reference) | "
            f"{local['mean_local_heldout_relative_mse']:.8g} | "
            f"{local['mean_final_heldout_relative_mse']:.8g} |"
        ),
        "",
        "## Per-layer BF16 results",
        "",
        "| Layer | Block final heldout | Full final heldout | Relative reduction |",
        "|---:|---:|---:|---:|",
    ]
    for row in payload["layers"]:
        block_mse = float(
            row["block_diagonal_fixed_encoder_bfloat16"]["final_heldout_relative_mse"]
        )
        full_mse = float(
            row["full_fixed_encoder_bfloat16"]["final_heldout_relative_mse"]
        )
        reduction = (block_mse - full_mse) / max(block_mse, 1.0e-300)
        lines.append(
            f"| {row['layer']} | {block_mse:.8g} | {full_mse:.8g} | "
            f"{100.0 * reduction:.4f}% |"
        )
    lines.extend(
        [
            "",
            (
                "For fixed factors, private-latent AllGather followed by the "
                "decoder and local private decoding followed by hidden AllReduce "
                "compute the same function. The block-diagonal quality arm does "
                "not prescribe a different collective."
            ),
            "",
            (
                "The independent-local arm changes both encoders and decoders, so "
                "its difference from either fixed-encoder arm is descriptive and "
                "is not attributed solely to cross-source covariance."
            ),
            "",
            "These are attention-layer output errors, not terminal-logit KL or PPL.",
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
    parser.add_argument("--independent-local-dir", required=True)
    parser.add_argument("--covariance-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--layers", default="all")
    parser.add_argument("--source-rank", type=int, default=SOURCE_RANK)
    parser.add_argument("--covariance-damping", type=float, default=1.0e-5)
    parser.add_argument("--expected-joint-sweeps", type=int, default=20)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--work-dtype", choices=("float64",), default="float64")
    parser.add_argument("--factor-dtype", choices=("bfloat16",), default="bfloat16")
    parser.add_argument("--torch-num-threads", type=int, default=4)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if args.source_rank != SOURCE_RANK:
        raise ValueError("Experiment J is fixed to source rank 512")
    if args.covariance_damping < 0:
        raise ValueError("covariance damping must be nonnegative")
    if args.expected_joint_sweeps <= 0:
        raise ValueError("expected joint sweeps must be positive")
    torch.set_num_threads(args.torch_num_threads)
    phase1_dir = Path(args.phase1_dir).expanduser().resolve()
    independent_dir = Path(args.independent_local_dir).expanduser().resolve()
    covariance_dir = Path(args.covariance_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    phase1, independent, covariance_manifest = _validated_inputs(
        phase1_dir,
        independent_dir,
        covariance_dir,
        expected_joint_sweeps=args.expected_joint_sweeps,
    )
    expected_damping = float(
        phase1["method"]["run_signature"]["c1_fit_config"]["covariance_damping"]
    )
    if args.covariance_damping != expected_damping:
        raise ValueError(
            f"requested covariance damping {args.covariance_damping} differs "
            f"from checkpoint damping {expected_damping}"
        )
    selected_layers = _parse_layers(args.layers, num_layers=36)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the real-model covariance ablation requires CUDA")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    work_dtype = torch.float64
    factor_dtype = torch.bfloat16
    layout = TPSourceWOLayout(
        input_width=HIDDEN_SIZE,
        output_width=HIDDEN_SIZE,
        tp_size=TP_SIZE,
        source_rank=SOURCE_RANK,
        dtype_bytes=2,
    )
    phase1_by_layer = {int(row["layer"]): row for row in phase1["layers"]}
    independent_by_layer = {int(row["layer"]): row for row in independent["layers"]}
    covariance_artifacts = covariance_manifest["artifacts"]
    output_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    records = []
    for ordinal, layer in enumerate(selected_layers, start=1):
        phase_record = phase1_by_layer[layer]
        independent_record = independent_by_layer[layer]
        joint_path = phase1_dir / phase_record["artifact"]["file"]
        local_path = independent_dir / independent_record["artifact"]["file"]
        covariance_record = covariance_artifacts[str(layer)]
        covariance_path = covariance_dir / covariance_record["file"]
        for label, path, expected_hash in (
            ("joint", joint_path, phase_record["artifact"]["sha256"]),
            ("independent-local", local_path, independent_record["artifact"]["sha256"]),
            ("covariance", covariance_path, covariance_record["sha256"]),
        ):
            if _sha256(path) != expected_hash:
                raise RuntimeError(f"{label} artifact hash mismatch at layer {layer}")

        joint_payload = load_file(str(joint_path), device="cpu")
        local_payload = load_file(str(local_path), device="cpu")
        sufficient = load_file(str(covariance_path), device="cpu")
        weight = sufficient["weight"].to(device=device, dtype=work_dtype)
        fit_covariance = sufficient["fit_covariance"].to(
            device=device, dtype=work_dtype
        )
        heldout_covariance = sufficient["heldout_covariance"].to(
            device=device, dtype=work_dtype
        )
        objectives, _, mapping, absolute_damping = _objectives(
            weight,
            fit_covariance,
            heldout_covariance,
            layout,
            covariance_damping=args.covariance_damping,
        )
        fixed_encoders = joint_payload["c1_source_encoders"].to(
            device=device, dtype=work_dtype
        )
        stored_joint_decoders = joint_payload["c1_source_decoders"].to(
            device=device, dtype=work_dtype
        )
        local_encoders = local_payload["c1_source_encoders"].to(
            device=device, dtype=work_dtype
        )
        local_decoders = local_payload["c1_source_decoders"].to(
            device=device, dtype=work_dtype
        )
        refits = refit_fixed_encoders(objectives, fixed_encoders, mapping)

        full_exact = _evaluate_factors(
            objectives, fixed_encoders, refits.full_decoders, mapping
        )
        block_exact = _evaluate_factors(
            objectives, fixed_encoders, refits.block_diagonal_decoders, mapping
        )
        stored_full_decoders = refits.full_decoders.to(dtype=factor_dtype)
        stored_block_decoders = refits.block_diagonal_decoders.to(dtype=factor_dtype)
        full_bfloat16 = _evaluate_factors(
            objectives,
            fixed_encoders,
            stored_full_decoders.to(dtype=work_dtype),
            mapping,
        )
        block_bfloat16 = _evaluate_factors(
            objectives,
            fixed_encoders,
            stored_block_decoders.to(dtype=work_dtype),
            mapping,
        )
        joint_stored_bfloat16 = _evaluate_factors(
            objectives, fixed_encoders, stored_joint_decoders, mapping
        )
        independent_bfloat16 = _evaluate_factors(
            objectives, local_encoders, local_decoders, mapping
        )
        block_wins_local_fit = (
            block_exact["local_fit_damped_relative_mse"]
            <= full_exact["local_fit_damped_relative_mse"] + 1.0e-12
        )
        full_wins_final_fit = (
            full_exact["final_fit_damped_relative_mse"]
            <= block_exact["final_fit_damped_relative_mse"] + 1.0e-12
        )
        if not block_wins_local_fit or not full_wins_final_fit:
            raise AssertionError(
                f"decoder optimum gate failed at layer {layer}: "
                f"block_local={block_wins_local_fit}, full_final={full_wins_final_fit}"
            )

        artifact_path = output_dir / f"layer_{layer:03d}.safetensors"
        stored_fixed_encoders = joint_payload["c1_source_encoders"].contiguous()
        _atomic_safetensors(
            artifact_path,
            {
                "fixed_source_encoders": stored_fixed_encoders,
                "full_source_decoders": stored_full_decoders.cpu().contiguous(),
                "block_diagonal_source_decoders": stored_block_decoders.cpu().contiguous(),
            },
        )
        record = {
            "layer": layer,
            "full_fixed_encoder_exact": full_exact,
            "full_fixed_encoder_bfloat16": full_bfloat16,
            "block_diagonal_fixed_encoder_exact": block_exact,
            "block_diagonal_fixed_encoder_bfloat16": block_bfloat16,
            "joint_stored_bfloat16": joint_stored_bfloat16,
            "independent_local_bfloat16": independent_bfloat16,
            "checks": {
                "block_diagonal_exact_wins_local_fit": block_wins_local_fit,
                "full_covariance_exact_wins_final_fit": full_wins_final_fit,
            },
            "solver": {
                "full_covariance": refits.full_diagnostics,
                "block_diagonal_covariance": refits.block_diagonal_diagnostics,
            },
            "numerics": {"absolute_covariance_damping": absolute_damping},
            "artifact": {
                "file": artifact_path.name,
                "sha256": _sha256(artifact_path),
                "tensors": {
                    "fixed_source_encoders": {
                        "shape": list(stored_fixed_encoders.shape),
                        "dtype": str(stored_fixed_encoders.dtype),
                    },
                    "full_source_decoders": {
                        "shape": list(stored_full_decoders.shape),
                        "dtype": str(stored_full_decoders.dtype),
                    },
                    "block_diagonal_source_decoders": {
                        "shape": list(stored_block_decoders.shape),
                        "dtype": str(stored_block_decoders.dtype),
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
                    "block_final_heldout": block_bfloat16["final_heldout_relative_mse"],
                    "full_final_heldout": full_bfloat16["final_heldout_relative_mse"],
                    "full_wins_heldout": (
                        full_bfloat16["final_heldout_relative_mse"]
                        <= block_bfloat16["final_heldout_relative_mse"]
                    ),
                }
            ),
            flush=True,
        )
        del joint_payload, local_payload, sufficient
        del weight, fit_covariance, heldout_covariance, objectives, mapping
        del fixed_encoders, stored_joint_decoders, local_encoders, local_decoders
        del refits, stored_full_decoders, stored_block_decoders
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
            "independent_local_dir": str(independent_dir),
            "independent_local_results_sha256": _sha256(
                independent_dir / "results.json"
            ),
            "covariance_dir": str(covariance_dir),
            "covariance_manifest_sha256": _sha256(covariance_dir / "manifest.json"),
        },
        "protocol": {
            "experiment": "J_cross_source_covariance_ablation",
            "tp_size": TP_SIZE,
            "source_rank": SOURCE_RANK,
            "layers": list(selected_layers),
            "fit_windows": 256,
            "heldout_windows": 64,
            "positions_per_window": 2048,
            "covariance_damping": args.covariance_damping,
            "joint_encoder_sweeps": args.expected_joint_sweeps,
            "fixed_encoder_source": "stored BF16 joint-s20 checkpoint",
            "full_covariance_decoder": "exact full-layer normal-equation solve",
            "block_diagonal_decoder": "exact sourcewise normal-equation solve after zeroing off-diagonal covariance blocks",
            "independent_local_role": "external Experiment-I reference; encoders are not fixed",
            "quality_scope": "attention layer output; dense V and dense KV cache",
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
