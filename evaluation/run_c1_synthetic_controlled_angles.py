#!/usr/bin/env python3
"""Run the deterministic FP64 controlled-angle C1 representation experiment."""

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

import scipy.stats
import torch
from torch import Tensor


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basisserve.analysis.c1_rigorous import (  # noqa: E402
    decoder_union_energy_ranks,
    fit_strong_lr_allreduce_rank_bank,
    pairwise_decoder_subspace_metrics,
    private_products_weight,
    relative_output_mse,
    ring_allgather_bytes_per_rank,
    ring_allreduce_bytes_per_rank,
    simulate_c1_allgather,
    simulate_lr_allreduce,
    wire_matched_allreduce_rank,
)


FORMAT = "basisserve.c1.synthetic_controlled_angles.v1"


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


def _orthonormal(
    rows: int,
    columns: int,
    *,
    generator: torch.Generator,
) -> Tensor:
    if not 0 < columns <= rows:
        raise ValueError("orthonormal frame requires 0 < columns <= rows")
    values = torch.randn(
        rows,
        columns,
        generator=generator,
        dtype=torch.float64,
    )
    return torch.linalg.qr(values, mode="reduced").Q


def _source_activations(
    *,
    rows: int,
    tp_size: int,
    source_width: int,
    generator: torch.Generator,
) -> tuple[Tensor, ...]:
    total_width = tp_size * source_width
    if rows < total_width:
        raise ValueError("activation rows must cover the joint source width")
    joint = math.sqrt(rows) * _orthonormal(
        rows,
        total_width,
        generator=generator,
    )
    return tuple(joint.split(source_width, dim=1))


def _controlled_coefficients(
    *,
    angle_degrees: float,
    tp_size: int,
    source_width: int,
    source_rank: int,
    output_width: int,
    generator: torch.Generator,
) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...], tuple[Tensor, ...]]:
    required_output_width = (tp_size + 1) * source_rank
    if output_width < required_output_width:
        raise ValueError(
            f"output width must be at least {required_output_width}"
        )
    output_frame = _orthonormal(
        output_width,
        required_output_width,
        generator=generator,
    )
    common = output_frame[:, :source_rank].transpose(0, 1)
    cosine = math.cos(math.radians(angle_degrees))
    if math.isclose(angle_degrees, 0.0, abs_tol=1.0e-14):
        cosine = 1.0
    elif math.isclose(angle_degrees, 90.0, abs_tol=1.0e-14):
        cosine = 0.0
    common_scale = math.sqrt(cosine)
    private_scale = math.sqrt(1.0 - cosine)

    target_encoders = []
    target_decoders = []
    coefficients = []
    for source in range(tp_size):
        start = (source + 1) * source_rank
        private = output_frame[:, start : start + source_rank].transpose(0, 1)
        decoder = common_scale * common + private_scale * private
        encoder = _orthonormal(
            source_width,
            source_rank,
            generator=generator,
        )
        target_encoders.append(encoder)
        target_decoders.append(decoder)
        coefficients.append(encoder @ decoder)
    return (
        tuple(target_encoders),
        tuple(target_decoders),
        tuple(coefficients),
    )


def _fit_private_exact(
    coefficients: Sequence[Tensor],
    *,
    source_rank: int,
) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...], float]:
    started = time.perf_counter()
    encoders = []
    decoders = []
    for coefficient in coefficients:
        left, singular_values, right = torch.linalg.svd(
            coefficient,
            full_matrices=False,
        )
        retained = singular_values[:source_rank]
        balance = retained.sqrt()
        encoders.append(left[:, :source_rank] * balance.unsqueeze(0))
        decoders.append(balance.unsqueeze(1) * right[:source_rank])
    return tuple(encoders), tuple(decoders), time.perf_counter() - started


def _relative_tensor_mse(target: Tensor, approximation: Tensor) -> float:
    residual = target - approximation
    numerator = residual.square().sum().clamp_min(0)
    denominator = target.square().sum().clamp_min(torch.finfo(target.dtype).tiny)
    return float(numerator / denominator)


def _aggregate_pairwise(
    rows: Sequence[Mapping[str, float | int]],
) -> dict[str, float]:
    angles = []
    for row in rows:
        angles.extend(
            (
                float(row["minimum_angle_degrees"]),
                float(row["median_angle_degrees"]),
                float(row["mean_angle_degrees"]),
                float(row["maximum_angle_degrees"]),
            )
        )
    return {
        "minimum_angle_degrees": min(angles),
        "median_angle_degrees": statistics.median(angles),
        "mean_angle_degrees": statistics.fmean(angles),
        "maximum_angle_degrees": max(angles),
        "mean_cosine_squared": statistics.fmean(
            float(row["mean_cosine_squared"]) for row in rows
        ),
        "mean_projection_overlap": statistics.fmean(
            float(row["projection_overlap"]) for row in rows
        ),
        "mean_normalized_chordal_distance_squared": statistics.fmean(
            float(row["normalized_chordal_distance_squared"]) for row in rows
        ),
    }


@torch.no_grad()
def run_trial(
    *,
    angle_degrees: float,
    seed: int,
    tp_size: int,
    source_width: int,
    source_rank: int,
    output_width: int,
    rows: int,
    wire_dtype_bytes: int,
) -> dict[str, Any]:
    if not 0.0 <= angle_degrees <= 90.0:
        raise ValueError("controlled angles must lie in [0, 90] degrees")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    _, target_decoders, coefficients = _controlled_coefficients(
        angle_degrees=angle_degrees,
        tp_size=tp_size,
        source_width=source_width,
        source_rank=source_rank,
        output_width=output_width,
        generator=generator,
    )
    fit_activations = _source_activations(
        rows=rows,
        tp_size=tp_size,
        source_width=source_width,
        generator=generator,
    )
    heldout_activations = _source_activations(
        rows=rows,
        tp_size=tp_size,
        source_width=source_width,
        generator=generator,
    )
    fit_joint = torch.cat(fit_activations, dim=1)
    heldout_joint = torch.cat(heldout_activations, dim=1)
    fit_covariance = fit_joint.transpose(0, 1) @ fit_joint / rows
    heldout_covariance = heldout_joint.transpose(0, 1) @ heldout_joint / rows

    c1_encoders, c1_decoders, c1_fit_seconds = _fit_private_exact(
        coefficients,
        source_rank=source_rank,
    )
    target_weight = torch.cat(tuple(coefficients), dim=0).transpose(0, 1)
    c1_weight = private_products_weight(c1_encoders, c1_decoders)
    source_ranks = (source_rank,) * tp_size
    ar_rank = wire_matched_allreduce_rank(source_ranks)
    lr_factors = fit_strong_lr_allreduce_rank_bank(
        target_weight,
        fit_covariance,
        heldout_covariance,
        ranks=(ar_rank,),
        covariance_damping=0.0,
        work_dtype=torch.float64,
        factor_dtype=torch.float64,
    )[0]
    lr_encoders = tuple(lr_factors.input_factor.split(source_width, dim=0))

    fit_target = sum(
        activation @ coefficient
        for activation, coefficient in zip(fit_activations, coefficients)
    )
    heldout_target = sum(
        activation @ coefficient
        for activation, coefficient in zip(heldout_activations, coefficients)
    )
    c1_fit = simulate_c1_allgather(
        fit_activations,
        c1_encoders,
        c1_decoders,
    )
    c1_heldout = simulate_c1_allgather(
        heldout_activations,
        c1_encoders,
        c1_decoders,
    )
    lr_fit = simulate_lr_allreduce(
        fit_activations,
        lr_encoders,
        lr_factors.decoder,
    )
    lr_heldout = simulate_lr_allreduce(
        heldout_activations,
        lr_encoders,
        lr_factors.decoder,
    )

    pairwise = pairwise_decoder_subspace_metrics(c1_decoders)
    subspace = _aggregate_pairwise(pairwise)
    union = decoder_union_energy_ranks(c1_decoders)
    c1_fit_error = _relative_tensor_mse(fit_target, c1_fit)
    c1_heldout_error = _relative_tensor_mse(heldout_target, c1_heldout)
    lr_fit_error = _relative_tensor_mse(fit_target, lr_fit)
    lr_heldout_error = _relative_tensor_mse(heldout_target, lr_heldout)
    analytic_lr_error = 0.5 * (
        1.0 - math.cos(math.radians(angle_degrees))
    )
    ag_wire = ring_allgather_bytes_per_rank(
        rows=1,
        source_ranks=source_ranks,
        dtype_bytes=wire_dtype_bytes,
    )
    ar_wire = ring_allreduce_bytes_per_rank(
        rows=1,
        rank=ar_rank,
        tp_size=tp_size,
        dtype_bytes=wire_dtype_bytes,
    )
    covariance_identity_error = max(
        float(
            (fit_covariance - torch.eye(fit_covariance.shape[0])).abs().max()
        ),
        float(
            (
                heldout_covariance
                - torch.eye(heldout_covariance.shape[0])
            ).abs().max()
        ),
    )
    angle_error = max(
        abs(float(row[key]) - angle_degrees)
        for row in pairwise
        for key in (
            "minimum_angle_degrees",
            "median_angle_degrees",
            "mean_angle_degrees",
            "maximum_angle_degrees",
        )
    )
    gates = {
        "equal_ideal_ring_wire": math.isclose(ag_wire, ar_wire, abs_tol=0.0),
        "fit_and_heldout_covariance_identity": covariance_identity_error
        <= 5.0e-14,
        "principal_angle_matches_target": angle_error <= 1.0e-5,
        "c1_exact": max(c1_fit_error, c1_heldout_error) <= 1.0e-24,
        "lr_matches_analytic_optimum": max(
            abs(lr_fit_error - analytic_lr_error),
            abs(lr_heldout_error - analytic_lr_error),
        )
        <= 2.0e-12,
    }
    return {
        "angle_degrees": angle_degrees,
        "seed": seed,
        "status": "pass" if all(gates.values()) else "fail",
        "geometry": {
            "tp_size": tp_size,
            "source_width": source_width,
            "source_rank": source_rank,
            "c1_total_rank": sum(source_ranks),
            "lr_allreduce_rank": ar_rank,
            "output_width": output_width,
            "rows": rows,
        },
        "budget": {
            "mode": "equal_ideal_ring_wire",
            "wire_dtype_bytes": wire_dtype_bytes,
            "c1_allgather_bytes_per_rank_per_row": ag_wire,
            "lr_allreduce_bytes_per_rank_per_row": ar_wire,
            "c1_decoder_rows": sum(source_ranks),
            "lr_decoder_rows": ar_rank,
            "c1_decoder_flops_per_row": 2 * sum(source_ranks) * output_width,
            "lr_decoder_flops_per_row": 2 * ar_rank * output_width,
        },
        "c1_allgather": {
            "solver": "independent_exact_fp64_truncated_svd",
            "fit_seconds": c1_fit_seconds,
            "fit_summed_output_relative_mse": c1_fit_error,
            "heldout_summed_output_relative_mse": c1_heldout_error,
            "fit_operator_relative_mse": relative_output_mse(
                target_weight,
                c1_weight,
                fit_covariance,
            ),
            "heldout_operator_relative_mse": relative_output_mse(
                target_weight,
                c1_weight,
                heldout_covariance,
            ),
        },
        "lr_allreduce": {
            "solver": lr_factors.metrics["solver_status"],
            "fit_summed_output_relative_mse": lr_fit_error,
            "heldout_summed_output_relative_mse": lr_heldout_error,
            "fit_operator_relative_mse": relative_output_mse(
                target_weight,
                lr_factors.reconstructed_weight(),
                fit_covariance,
            ),
            "heldout_operator_relative_mse": relative_output_mse(
                target_weight,
                lr_factors.reconstructed_weight(),
                heldout_covariance,
            ),
            "analytic_relative_mse": analytic_lr_error,
            "relative_eigen_residual": lr_factors.metrics[
                "relative_eigen_residual"
            ],
            "shared_decomposition_seconds": lr_factors.metrics[
                "shared_decomposition_seconds"
            ],
        },
        "c1_advantage_heldout_relative_mse": (
            lr_heldout_error - c1_heldout_error
        ),
        "subspace": {
            "pairwise": list(pairwise),
            "aggregate": subspace,
            "union": union,
        },
        "diagnostics": {
            "maximum_covariance_identity_error": covariance_identity_error,
            "maximum_principal_angle_error_degrees": angle_error,
        },
        "gates": gates,
    }


def _mean(rows: Sequence[Mapping[str, Any]], path: Sequence[str]) -> float:
    values = []
    for row in rows:
        current: Any = row
        for key in path:
            current = current[key]
        values.append(float(current))
    return statistics.fmean(values)


def _aggregate_trials(
    trials: Sequence[Mapping[str, Any]],
    angles: Sequence[float],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    angle_rows = []
    for angle in angles:
        selected = [
            trial
            for trial in trials
            if math.isclose(float(trial["angle_degrees"]), angle, abs_tol=1.0e-12)
        ]
        union_95 = [
            int(trial["subspace"]["union"]["energy_ranks"]["95%"])  # type: ignore[index]
            for trial in selected
        ]
        angle_rows.append(
            {
                "angle_degrees": angle,
                "trials": len(selected),
                "measured_mean_angle_degrees": _mean(
                    selected,
                    ("subspace", "aggregate", "mean_angle_degrees"),
                ),
                "mean_projection_overlap": _mean(
                    selected,
                    ("subspace", "aggregate", "mean_projection_overlap"),
                ),
                "mean_normalized_chordal_distance_squared": _mean(
                    selected,
                    (
                        "subspace",
                        "aggregate",
                        "mean_normalized_chordal_distance_squared",
                    ),
                ),
                "union_energy_rank_95_min": min(union_95),
                "union_energy_rank_95_max": max(union_95),
                "mean_c1_heldout_relative_mse": _mean(
                    selected,
                    ("c1_allgather", "heldout_summed_output_relative_mse"),
                ),
                "mean_lr_heldout_relative_mse": _mean(
                    selected,
                    ("lr_allreduce", "heldout_summed_output_relative_mse"),
                ),
                "analytic_lr_relative_mse": _mean(
                    selected,
                    ("lr_allreduce", "analytic_relative_mse"),
                ),
                "mean_c1_advantage_heldout_relative_mse": _mean(
                    selected,
                    ("c1_advantage_heldout_relative_mse",),
                ),
                "all_trials_passed": all(
                    trial["status"] == "pass" for trial in selected
                ),
            }
        )

    diversity = [
        float(row["mean_normalized_chordal_distance_squared"])
        for row in angle_rows
    ]
    advantage = [
        float(row["mean_c1_advantage_heldout_relative_mse"])
        for row in angle_rows
    ]
    pearson = scipy.stats.pearsonr(diversity, advantage)
    spearman = scipy.stats.spearmanr(diversity, advantage)
    lr_errors = [float(row["mean_lr_heldout_relative_mse"]) for row in angle_rows]
    gates = {
        "all_trials_passed": all(row["all_trials_passed"] for row in angle_rows),
        "lr_error_monotonic_non_decreasing": all(
            following + 2.0e-12 >= previous
            for previous, following in zip(lr_errors, lr_errors[1:])
        ),
        "shared_endpoint_exact": lr_errors[0] <= 1.0e-24,
        "orthogonal_endpoint_is_half_error": abs(lr_errors[-1] - 0.5)
        <= 2.0e-12,
        "diversity_advantage_spearman_positive": float(spearman.statistic)
        >= 0.999999,
    }
    summary = {
        "angles_completed": len(angle_rows),
        "trials_completed": len(trials),
        "pearson_diversity_vs_c1_advantage": {
            "statistic": float(pearson.statistic),
            "pvalue": float(pearson.pvalue),
        },
        "spearman_diversity_vs_c1_advantage": {
            "statistic": float(spearman.statistic),
            "pvalue": float(spearman.pvalue),
        },
        "gates": gates,
        "passed": all(gates.values()),
    }
    return angle_rows, summary


def _energy_rank_cell(row: Mapping[str, Any]) -> str:
    minimum = int(row["union_energy_rank_95_min"])
    maximum = int(row["union_energy_rank_95_max"])
    return str(minimum) if minimum == maximum else f"{minimum}--{maximum}"


def _summary_markdown(payload: Mapping[str, Any]) -> str:
    geometry = payload["geometry"]
    budget = payload["budget"]
    summary = payload["summary"]
    lines = [
        "# C1 controlled source-subspace angles",
        "",
        (
            "This deterministic CPU/FP64 experiment is a scaled Qwen3-8B TP4 "
            "representation sanity check. It isolates source output-subspace "
            "angle from calibration noise and model effects."
        ),
        "",
        "## Geometry and budget",
        "",
        f"- TP sources: `{geometry['tp_size']}`",
        f"- Per-source C1 rank: `{geometry['source_rank']}`",
        f"- Total C1 AllGather rank: `{geometry['c1_total_rank']}`",
        f"- Equal-wire LR-AllReduce rank: `{geometry['lr_allreduce_rank']}`",
        f"- Fit and heldout rows: `{geometry['rows']}` each",
        "- Work/factor dtype: `float64`",
        (
            "- Ideal ring traffic per rank per row: "
            f"C1 `{budget['c1_allgather_bytes_per_rank_per_row']:.0f}` bytes, "
            f"LR `{budget['lr_allreduce_bytes_per_rank_per_row']:.0f}` bytes"
        ),
        "",
        "## Results",
        "",
        "| Target angle | Measured angle | Projection overlap | Normalized chordal d^2 | Union r95 | C1 heldout MSE | LR heldout MSE | Analytic LR MSE |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["angles"]:
        lines.append(
            f"| {row['angle_degrees']:.0f}° | "
            f"{row['measured_mean_angle_degrees']:.6f}° | "
            f"{row['mean_projection_overlap']:.6f} | "
            f"{row['mean_normalized_chordal_distance_squared']:.6f} | "
            f"{_energy_rank_cell(row)} | "
            f"{row['mean_c1_heldout_relative_mse']:.6g} | "
            f"{row['mean_lr_heldout_relative_mse']:.6g} | "
            f"{row['analytic_lr_relative_mse']:.6g} |"
        )
    lines.extend(
        [
            "",
            "## Validation",
            "",
            (
                "- Pearson diversity/C1-advantage correlation: "
                f"`{summary['pearson_diversity_vs_c1_advantage']['statistic']:.6f}`"
            ),
            (
                "- Spearman diversity/C1-advantage correlation: "
                f"`{summary['spearman_diversity_vs_c1_advantage']['statistic']:.6f}`"
            ),
        ]
    )
    for name, passed in summary["gates"].items():
        lines.append(f"- `{name}`: `{'pass' if passed else 'FAIL'}`")
    lines.extend(
        [
            "",
            "The synthetic result validates the conceptual prediction only. "
            "It does not establish that pretrained-model TP sources have "
            "diverse subspaces; that is the next real-model analysis.",
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


def _parse_csv(raw: str, *, converter: type[int] | type[float]) -> tuple[Any, ...]:
    values = tuple(converter(piece.strip()) for piece in raw.split(",") if piece.strip())
    if not values:
        raise ValueError("comma-separated option must not be empty")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--source-width", type=int, default=16)
    parser.add_argument("--source-rank", type=int, default=4)
    parser.add_argument("--output-width", type=int, default=32)
    parser.add_argument("--rows", type=int, default=128)
    parser.add_argument("--angles", default="0,15,30,45,60,75,90")
    parser.add_argument("--seeds", default="20260828,20260829,20260830")
    parser.add_argument("--wire-dtype-bytes", type=int, default=2)
    parser.add_argument("--torch-num-threads", type=int, default=1)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.torch_num_threads)
    if args.tp_size != 4:
        raise ValueError("this experiment is fixed to the Qwen3-8B TP4 hypothesis")
    angles = tuple(
        sorted(set(_parse_csv(args.angles, converter=float)))
    )
    seeds = tuple(_parse_csv(args.seeds, converter=int))
    if angles != (0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0):
        raise ValueError("the canonical angle grid is fixed by the experiment")
    if len(set(seeds)) != len(seeds):
        raise ValueError("synthetic seeds must be unique")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_json = output_dir / "results.json"
    output_markdown = output_dir / "summary.md"

    started = time.perf_counter()
    trials = []
    for angle in angles:
        for seed in seeds:
            trial = run_trial(
                angle_degrees=angle,
                seed=seed,
                tp_size=args.tp_size,
                source_width=args.source_width,
                source_rank=args.source_rank,
                output_width=args.output_width,
                rows=args.rows,
                wire_dtype_bytes=args.wire_dtype_bytes,
            )
            trials.append(trial)
            print(
                json.dumps(
                    {
                        "event": "trial_complete",
                        "angle_degrees": angle,
                        "seed": seed,
                        "status": trial["status"],
                        "c1_heldout_relative_mse": trial["c1_allgather"][
                            "heldout_summed_output_relative_mse"
                        ],
                        "lr_heldout_relative_mse": trial["lr_allreduce"][
                            "heldout_summed_output_relative_mse"
                        ],
                    }
                ),
                flush=True,
            )
    angle_rows, summary = _aggregate_trials(trials, angles)
    first = trials[0]
    payload = {
        "format": FORMAT,
        "schema_version": 1,
        "status": "complete" if summary["passed"] else "failed_validation",
        "command": shlex.join((sys.executable, *sys.argv)),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "elapsed_seconds": time.perf_counter() - started,
        "environment": {
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "device": "cpu",
            "torch": torch.__version__,
            "work_dtype": "float64",
            "factor_dtype": "float64",
            "torch_num_threads": args.torch_num_threads,
        },
        "model_context": {
            "model": "Qwen3-8B-Base",
            "scope": "model-independent scaled TP4 representation analogue",
            "real_c1_source_rank": 512,
            "real_c1_total_rank": 2048,
            "real_equal_wire_lr_rank": 1024,
            "synthetic_rank_scale": 128,
        },
        "geometry": first["geometry"],
        "budget": first["budget"],
        "angles_degrees": list(angles),
        "seeds": list(seeds),
        "angles": angle_rows,
        "summary": summary,
        "trials": trials,
    }
    _atomic_text(
        output_json,
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )
    _atomic_text(output_markdown, _summary_markdown(payload))
    if payload["status"] != "complete":
        raise RuntimeError(f"synthetic validation failed; inspect {output_json}")
    print(
        json.dumps(
            {
                "event": "result_written",
                "results": str(output_json),
                "results_sha256": _sha256(output_json),
                "summary": str(output_markdown),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
