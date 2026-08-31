#!/usr/bin/env python3
"""Benchmark Qwen3-8B Wo decoder GEMMs for the C1 uniform-rank sweep."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import sys
from typing import Any, Callable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402


FORMAT = "basisserve.qwen3_8b.wo_decoder_gemm_microbenchmark.v1"
TP_SIZE = 4
HIDDEN_SIZE = 4096
SOURCE_WIDTH = HIDDEN_SIZE // TP_SIZE
DTYPE = torch.bfloat16
DTYPE_BYTES = 2
DEFAULT_LOCAL_RANKS = "512,640,768,896,1024"
DEFAULT_ROWS = "1,8,32,64,128,256"


def parse_positive_csv(raw: str, *, label: str) -> tuple[int, ...]:
    """Parse a nonempty, duplicate-free CSV of positive integers."""

    values = tuple(int(piece.strip()) for piece in raw.split(",") if piece.strip())
    if not values or len(values) != len(set(values)):
        raise ValueError(f"{label} must be nonempty and unique")
    if any(value <= 0 for value in values):
        raise ValueError(f"{label} must be positive")
    return values


def decoder_geometry(*, local_rank: int, rows: int) -> dict[str, Any]:
    """Return decoder widths and exact GEMM work for one TP4 C1 rank."""

    selected_rank = int(local_rank)
    selected_rows = int(rows)
    if not 0 < selected_rank <= SOURCE_WIDTH:
        raise ValueError(f"local rank must lie in [1, {SOURCE_WIDTH}]")
    if selected_rows <= 0:
        raise ValueError("rows must be positive")
    c1_rank = TP_SIZE * selected_rank
    wire_lr_rank = 2 * selected_rank
    return {
        "rows": selected_rows,
        "local_c1_rank": selected_rank,
        "c1_total_decoder_rank": c1_rank,
        "wire_matched_lr_decoder_rank": wire_lr_rank,
        "dense_local_reduction_width": SOURCE_WIDTH,
        "output_width": HIDDEN_SIZE,
        "c1_decoder_flop_ratio_vs_dense_local_o_proj": c1_rank / SOURCE_WIDTH,
        "wire_lr_decoder_flop_ratio_vs_dense_local_o_proj": (
            wire_lr_rank / SOURCE_WIDTH
        ),
    }


def gemm_work(
    *,
    rows: int,
    reduction_width: int,
    output_width: int = HIDDEN_SIZE,
    dtype_bytes: int = DTYPE_BYTES,
) -> dict[str, float | int]:
    """Return GEMM FLOPs and the minimum compulsory tensor byte count."""

    selected_rows = int(rows)
    selected_reduction = int(reduction_width)
    selected_output = int(output_width)
    selected_bytes = int(dtype_bytes)
    if min(selected_rows, selected_reduction, selected_output, selected_bytes) <= 0:
        raise ValueError("GEMM dimensions and dtype bytes must be positive")
    flops = 2 * selected_rows * selected_reduction * selected_output
    minimum_bytes = selected_bytes * (
        selected_rows * selected_reduction
        + selected_reduction * selected_output
        + selected_rows * selected_output
    )
    return {
        "flops": flops,
        "minimum_algorithmic_bytes": minimum_bytes,
        "arithmetic_intensity_flops_per_minimum_byte": flops / minimum_bytes,
    }


def performance_metrics(
    *,
    rows: int,
    reduction_width: int,
    p50_ms: float,
    output_width: int = HIDDEN_SIZE,
) -> dict[str, float | int]:
    """Attach p50 throughput metrics to the exact GEMM work accounting."""

    selected_ms = float(p50_ms)
    if not math.isfinite(selected_ms) or selected_ms <= 0.0:
        raise ValueError("p50 latency must be finite and positive")
    work = gemm_work(
        rows=rows,
        reduction_width=reduction_width,
        output_width=output_width,
    )
    seconds = selected_ms / 1000.0
    return {
        **work,
        "tflops_at_p50": float(work["flops"]) / seconds / 1.0e12,
        "effective_minimum_bandwidth_gbps_at_p50": (
            float(work["minimum_algorithmic_bytes"]) / seconds / 1.0e9
        ),
    }


def comparison_metrics(arms: Mapping[str, Mapping[str, Any]]) -> dict[str, float]:
    """Compare decoder-only p50 latency without mixing in collectives."""

    def p50(name: str) -> float:
        value = float(arms[name]["timing"]["p50_ms"])
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"arm {name} has invalid p50 latency")
        return value

    dense = p50("dense_local_o_proj_gemm")
    c1 = p50("c1_feature_major_decoder_gemm")
    wire_lr = p50("wire_lr_token_major_decoder_gemm")
    return {
        "c1_latency_change_vs_dense_local_o_proj": c1 / dense - 1.0,
        "wire_lr_latency_change_vs_dense_local_o_proj": wire_lr / dense - 1.0,
        "c1_latency_change_vs_wire_lr": c1 / wire_lr - 1.0,
    }


def _quantile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(map(float, values))
    if not ordered:
        raise ValueError("cannot summarize empty timings")
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def _summary(values: Sequence[float]) -> dict[str, float]:
    selected = tuple(map(float, values))
    if not selected:
        raise ValueError("cannot summarize empty timings")
    return {
        "mean_ms": statistics.fmean(selected),
        "minimum_ms": min(selected),
        "p50_ms": statistics.median(selected),
        "p95_ms": _quantile(selected, 0.95),
        "maximum_ms": max(selected),
    }


def _cuda_timings(
    function: Callable[[], torch.Tensor],
    *,
    warmup: int,
    iterations: int,
    device: torch.device,
) -> list[float]:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize(device)
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    for start, end in zip(starts, ends, strict=True):
        start.record()
        function()
        end.record()
    torch.cuda.synchronize(device)
    return [
        float(start.elapsed_time(end))
        for start, end in zip(starts, ends, strict=True)
    ]


def _mm(
    left: torch.Tensor,
    right: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor:
    torch.mm(left, right, out=output)
    return output


def _arm_record(
    *,
    layout: str,
    reduction_width: int,
    rows: int,
    timings_ms: Sequence[float],
) -> dict[str, Any]:
    timing = _summary(timings_ms)
    return {
        "operation": f"BF16 GEMM [{rows},{reduction_width}]x[{reduction_width},{HIDDEN_SIZE}]",
        "input_layout": layout,
        "allocation_scope": "preallocated output; GEMM kernel only",
        "reduction_width": int(reduction_width),
        "output_width": HIDDEN_SIZE,
        "timing": timing,
        "performance": performance_metrics(
            rows=rows,
            reduction_width=reduction_width,
            p50_ms=timing["p50_ms"],
        ),
    }


@torch.inference_mode()
def _correctness_gate(
    local_ranks: Sequence[int],
    *,
    device: torch.device,
) -> dict[str, Any]:
    generator = torch.Generator(device=device).manual_seed(20260829)
    records = []
    maximum_relative_l2 = 0.0
    maximum_absolute = 0.0
    for local_rank in local_ranks:
        total_rank = TP_SIZE * local_rank
        logical = torch.randn(
            (7, total_rank),
            generator=generator,
            device=device,
            dtype=DTYPE,
        )
        feature_major = logical.transpose(0, 1).contiguous()
        decoder = torch.randn(
            (total_rank, HIDDEN_SIZE),
            generator=generator,
            device=device,
            dtype=DTYPE,
        )
        reference = torch.mm(logical, decoder)
        observed = torch.mm(feature_major.transpose(0, 1), decoder)
        difference = observed.float() - reference.float()
        relative_l2 = float(
            torch.linalg.vector_norm(difference)
            / torch.linalg.vector_norm(reference.float()).clamp_min(1.0e-12)
        )
        absolute = float(difference.abs().amax())
        maximum_relative_l2 = max(maximum_relative_l2, relative_l2)
        maximum_absolute = max(maximum_absolute, absolute)
        records.append(
            {
                "local_c1_rank": local_rank,
                "total_decoder_rank": total_rank,
                "rows": 7,
                "relative_l2": relative_l2,
                "maximum_absolute": absolute,
            }
        )
    if maximum_relative_l2 > 5.0e-3:
        raise AssertionError(
            "feature-major decoder correctness gate failed: "
            f"relative_l2={maximum_relative_l2}"
        )
    return {
        "status": "passed",
        "reference": "logically identical token-major torch.mm",
        "maximum_relative_l2": maximum_relative_l2,
        "maximum_absolute": maximum_absolute,
        "rank_records": records,
    }


@torch.inference_mode()
def _benchmark(
    local_ranks: Sequence[int],
    rows_grid: Sequence[int],
    *,
    warmup: int,
    iterations: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    generator = torch.Generator(device=device).manual_seed(20260830)
    dense_weight = torch.randn(
        (SOURCE_WIDTH, HIDDEN_SIZE),
        generator=generator,
        device=device,
        dtype=DTYPE,
    )
    dense_arms: dict[int, dict[str, Any]] = {}
    for rows in rows_grid:
        dense_input = torch.randn(
            (rows, SOURCE_WIDTH),
            generator=generator,
            device=device,
            dtype=DTYPE,
        )
        dense_output = torch.empty(
            (rows, HIDDEN_SIZE),
            device=device,
            dtype=DTYPE,
        )
        timings = _cuda_timings(
            lambda lhs=dense_input, rhs=dense_weight, out=dense_output: _mm(
                lhs, rhs, out
            ),
            warmup=warmup,
            iterations=iterations,
            device=device,
        )
        dense_arms[rows] = _arm_record(
            layout="token-major contiguous",
            reduction_width=SOURCE_WIDTH,
            rows=rows,
            timings_ms=timings,
        )

    records = []
    for local_rank in local_ranks:
        c1_rank = TP_SIZE * local_rank
        wire_lr_rank = 2 * local_rank
        c1_weight = torch.randn(
            (c1_rank, HIDDEN_SIZE),
            generator=generator,
            device=device,
            dtype=DTYPE,
        )
        wire_lr_weight = torch.randn(
            (wire_lr_rank, HIDDEN_SIZE),
            generator=generator,
            device=device,
            dtype=DTYPE,
        )
        for rows in rows_grid:
            c1_token_major = torch.randn(
                (rows, c1_rank),
                generator=generator,
                device=device,
                dtype=DTYPE,
            )
            c1_feature_major = c1_token_major.transpose(0, 1).contiguous()
            c1_output = torch.empty(
                (rows, HIDDEN_SIZE),
                device=device,
                dtype=DTYPE,
            )
            wire_lr_input = torch.randn(
                (rows, wire_lr_rank),
                generator=generator,
                device=device,
                dtype=DTYPE,
            )
            wire_lr_output = torch.empty_like(c1_output)

            c1_timings = _cuda_timings(
                lambda lhs=c1_feature_major.transpose(0, 1), rhs=c1_weight, out=c1_output: _mm(
                    lhs, rhs, out
                ),
                warmup=warmup,
                iterations=iterations,
                device=device,
            )
            wire_lr_timings = _cuda_timings(
                lambda lhs=wire_lr_input, rhs=wire_lr_weight, out=wire_lr_output: _mm(
                    lhs, rhs, out
                ),
                warmup=warmup,
                iterations=iterations,
                device=device,
            )
            arms = {
                "dense_local_o_proj_gemm": dense_arms[rows],
                "c1_feature_major_decoder_gemm": _arm_record(
                    layout=(
                        "feature-major contiguous [R,B], transposed view [B,R]; "
                        "current production C1 layout"
                    ),
                    reduction_width=c1_rank,
                    rows=rows,
                    timings_ms=c1_timings,
                ),
                "wire_lr_token_major_decoder_gemm": _arm_record(
                    layout="token-major contiguous after latent AllReduce",
                    reduction_width=wire_lr_rank,
                    rows=rows,
                    timings_ms=wire_lr_timings,
                ),
            }
            record = {
                "local_c1_rank": local_rank,
                "rows": rows,
                "geometry": decoder_geometry(local_rank=local_rank, rows=rows),
                "arms": arms,
                "comparisons": comparison_metrics(arms),
            }
            records.append(record)
            print(
                json.dumps(
                    {
                        "event": "configuration_complete",
                        "local_c1_rank": local_rank,
                        "rows": rows,
                        "dense_p50_ms": arms["dense_local_o_proj_gemm"]["timing"][
                            "p50_ms"
                        ],
                        "c1_p50_ms": arms["c1_feature_major_decoder_gemm"][
                            "timing"
                        ]["p50_ms"],
                        "wire_lr_p50_ms": arms[
                            "wire_lr_token_major_decoder_gemm"
                        ]["timing"]["p50_ms"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    return records


def _command_output(command: Sequence[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"command": list(command), "error": str(error)}
    return {
        "command": list(command),
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _git_commit() -> str | None:
    result = _command_output(("git", "rev-parse", "HEAD"))
    if result.get("returncode") != 0:
        return None
    return str(result["stdout"]).strip() or None


def _atomic_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _summary_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Qwen3-8B Wo decoder GEMM microbenchmark",
        "",
        (
            "BF16 on one GPU. Timings isolate the preallocated-output GEMM kernel; "
            "they exclude encoders, collectives, allocation, and the rest of attention."
        ),
        "",
        "| Local C1 rank | Rows | Dense K | C1 K | Wire-LR K | Dense ms | C1 ms | LR ms | C1 TFLOP/s | LR TFLOP/s | C1 vs LR |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in payload["records"]:
        geometry = record["geometry"]
        arms = record["arms"]
        comparisons = record["comparisons"]
        dense = arms["dense_local_o_proj_gemm"]
        c1 = arms["c1_feature_major_decoder_gemm"]
        wire_lr = arms["wire_lr_token_major_decoder_gemm"]
        lines.append(
            f"| {record['local_c1_rank']} | {record['rows']} | "
            f"{geometry['dense_local_reduction_width']} | "
            f"{geometry['c1_total_decoder_rank']} | "
            f"{geometry['wire_matched_lr_decoder_rank']} | "
            f"{dense['timing']['p50_ms']:.6f} | "
            f"{c1['timing']['p50_ms']:.6f} | "
            f"{wire_lr['timing']['p50_ms']:.6f} | "
            f"{c1['performance']['tflops_at_p50']:.3f} | "
            f"{wire_lr['performance']['tflops_at_p50']:.3f} | "
            f"{100.0 * comparisons['c1_latency_change_vs_wire_lr']:+.2f}% |"
        )
    lines.extend(
        [
            "",
            "## Protocol",
            "",
            f"- Correctness gate: `{payload['correctness_gate']['status']}`",
            f"- Warmup calls: `{payload['protocol']['warmup']}`",
            f"- Measured iterations: `{payload['protocol']['iterations']}`",
            "- Timing: CUDA events on one GPU",
            "- Occupancy and physical HBM counters: not collected; these require Nsight Compute",
            "- Reported effective bandwidth uses minimum algorithmic tensor bytes and is not a hardware-counter measurement",
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
    parser.add_argument("--local-ranks", default=DEFAULT_LOCAL_RANKS)
    parser.add_argument("--rows", default=DEFAULT_ROWS)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--torch-num-threads", type=int, default=2)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    local_ranks = parse_positive_csv(args.local_ranks, label="local ranks")
    rows_grid = parse_positive_csv(args.rows, label="rows")
    if max(local_ranks) > SOURCE_WIDTH:
        raise ValueError(f"local ranks cannot exceed {SOURCE_WIDTH}")
    if min(args.warmup, args.iterations, args.torch_num_threads) <= 0:
        raise ValueError("warmup, iterations, and thread count must be positive")
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if not torch.cuda.is_available():
        raise RuntimeError("decoder GEMM microbenchmark requires CUDA")

    torch.set_num_threads(args.torch_num_threads)
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    correctness = _correctness_gate(local_ranks, device=device)
    records = _benchmark(
        local_ranks,
        rows_grid,
        warmup=args.warmup,
        iterations=args.iterations,
        device=device,
    )
    payload = {
        "format": FORMAT,
        "schema_version": 1,
        "status": "complete",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join((sys.executable, *sys.argv)),
        "git_commit": _git_commit(),
        "protocol": {
            "model_geometry": "Qwen3-8B-Base Wo-only TP4",
            "tp_size": TP_SIZE,
            "hidden_size": HIDDEN_SIZE,
            "source_width": SOURCE_WIDTH,
            "local_c1_ranks": list(local_ranks),
            "c1_total_decoder_ranks": [TP_SIZE * rank for rank in local_ranks],
            "wire_matched_lr_decoder_ranks": [2 * rank for rank in local_ranks],
            "rows": list(rows_grid),
            "dtype": "bfloat16",
            "dtype_bytes": DTYPE_BYTES,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "timing_scope": "preallocated-output GEMM kernel only",
            "excluded": [
                "local encoder GEMM",
                "collective",
                "output allocation",
                "attention",
                "MLP",
            ],
        },
        "correctness_gate": correctness,
        "records": records,
        "environment": {
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "gpu": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "matmul_precision": torch.get_float32_matmul_precision(),
            "occupancy_and_hardware_counters": "not collected",
            "nvidia_smi": _command_output(
                (
                    "nvidia-smi",
                    "--query-gpu=name,driver_version,memory.total",
                    "--format=csv,noheader",
                )
            ),
        },
    }
    output_dir.mkdir(parents=True)
    _atomic_text(
        output_dir / "results.json",
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )
    _atomic_text(output_dir / "summary.md", _summary_markdown(payload))


if __name__ == "__main__":
    main()
