#!/usr/bin/env python3
"""Benchmark the isolated collectives in the Qwen3-8B Wo-only TP4 design."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any, Callable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

from basisserve.analysis.c1_rigorous import (  # noqa: E402
    ring_allgather_bytes_per_rank,
    ring_allreduce_bytes_per_rank,
    wire_matched_allreduce_rank,
)
from basisserve.kernels.feature_ragged_allgather import (  # noqa: E402
    FeatureRaggedCommunicator,
)
from basisserve.kernels.ragged_allgather import StaticRaggedPlan  # noqa: E402
from evaluation.benchmark_tp4_interconnect import (  # noqa: E402
    _critical_cuda_timings,
    _summary,
)


FORMAT = "basisserve.qwen3_8b.wo_collective_microbenchmark.tp4.v1"
TP_SIZE = 4
HIDDEN_SIZE = 4096
SOURCE_WIDTH = HIDDEN_SIZE // TP_SIZE
DTYPE = torch.bfloat16
DTYPE_BYTES = 2
DEFAULT_SOURCE_RANKS = "512,640,768,896,1024"
DEFAULT_TOKEN_ROWS = "1,8,32,64,512,2048,8192,32768"


def parse_positive_csv(raw: str, *, label: str) -> tuple[int, ...]:
    """Parse a nonempty, duplicate-free CSV of positive integers."""

    values = tuple(int(piece.strip()) for piece in raw.split(",") if piece.strip())
    if not values or len(values) != len(set(values)):
        raise ValueError(f"{label} must be nonempty and unique")
    if any(value <= 0 for value in values):
        raise ValueError(f"{label} must be positive")
    return values


def collective_geometry(*, source_rank: int, rows: int) -> dict[str, Any]:
    """Return the exact TP4 equal-wire geometry for one rank and row count."""

    selected_rank = int(source_rank)
    selected_rows = int(rows)
    if not 0 < selected_rank <= SOURCE_WIDTH:
        raise ValueError(f"source rank must lie in [1, {SOURCE_WIDTH}]")
    if selected_rows <= 0:
        raise ValueError("token rows must be positive")
    source_ranks = (selected_rank,) * TP_SIZE
    lr_rank = wire_matched_allreduce_rank(source_ranks)
    c1_bytes = ring_allgather_bytes_per_rank(
        rows=selected_rows,
        source_ranks=source_ranks,
        dtype_bytes=DTYPE_BYTES,
    )
    lr_bytes = ring_allreduce_bytes_per_rank(
        rows=selected_rows,
        rank=lr_rank,
        tp_size=TP_SIZE,
        dtype_bytes=DTYPE_BYTES,
    )
    dense_bytes = ring_allreduce_bytes_per_rank(
        rows=selected_rows,
        rank=HIDDEN_SIZE,
        tp_size=TP_SIZE,
        dtype_bytes=DTYPE_BYTES,
    )
    if c1_bytes != lr_bytes:
        raise AssertionError("C1-AllGather and LR-AllReduce are not wire matched")
    return {
        "source_rank": selected_rank,
        "retained_source_ratio": selected_rank / SOURCE_WIDTH,
        "wire_matched_lr_rank": lr_rank,
        "token_rows": selected_rows,
        "c1_allgather_local_payload_bytes": selected_rows * selected_rank * DTYPE_BYTES,
        "wire_lr_allreduce_payload_bytes": selected_rows * lr_rank * DTYPE_BYTES,
        "dense_allreduce_payload_bytes": selected_rows * HIDDEN_SIZE * DTYPE_BYTES,
        "c1_and_wire_lr_ideal_ring_bytes_per_rank": c1_bytes,
        "dense_ideal_ring_bytes_per_rank": dense_bytes,
        "ideal_ring_communication_reduction_vs_dense": 1.0 - c1_bytes / dense_bytes,
        "wire_matching_passed": True,
    }


def comparison_metrics(arms: Mapping[str, Mapping[str, Any]]) -> dict[str, float]:
    """Compare p50 critical-path latency for the four measured timing scopes."""

    def p50(arm: str) -> float:
        value = float(arms[arm]["timing"]["p50_ms"])
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"arm {arm} has invalid p50 latency")
        return value

    dense = p50("dense_allreduce")
    collective = p50("c1_allgather_collective_only")
    packed = p50("c1_packing_plus_allgather")
    lr = p50("wire_lr_allreduce")
    return {
        "c1_collective_latency_change_vs_wire_lr": collective / lr - 1.0,
        "c1_packed_latency_change_vs_wire_lr": packed / lr - 1.0,
        "c1_packing_overhead_vs_collective_only": packed / collective - 1.0,
        "c1_collective_latency_reduction_vs_dense": 1.0 - collective / dense,
        "c1_packed_latency_reduction_vs_dense": 1.0 - packed / dense,
        "wire_lr_latency_reduction_vs_dense": 1.0 - lr / dense,
    }


def _atomic_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


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


def _global_max(value: float, *, device: torch.device) -> float:
    tensor = torch.tensor(float(value), dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor)


def _constant_error(tensor: torch.Tensor, expected: float) -> float:
    minimum, maximum = torch.aminmax(tensor.float())
    return max(abs(float(minimum) - expected), abs(float(maximum) - expected))


@torch.inference_mode()
def _correctness_gates(
    communicator: FeatureRaggedCommunicator,
    *,
    source_ranks: Sequence[int],
    device: torch.device,
) -> dict[str, Any]:
    """Validate dense AR, packed C1 AG, and equal-wire LR AR geometries."""

    rank = dist.get_rank()
    expected_sum = float(sum(range(1, TP_SIZE + 1)))
    dense = torch.full(
        (3, HIDDEN_SIZE),
        rank + 1,
        dtype=DTYPE,
        device=device,
    )
    dist.all_reduce(dense)
    dense_error = _global_max(
        _constant_error(dense, expected_sum),
        device=device,
    )

    records = []
    maximum_error = dense_error
    for source_rank in source_ranks:
        geometry = collective_geometry(source_rank=source_rank, rows=3)
        plan = StaticRaggedPlan.from_source_widths((source_rank,) * TP_SIZE)
        prepared = communicator.prepare_uniform(
            plan,
            tokens=3,
            dtype=DTYPE,
            backend="uniform_nccl",
        )
        local = torch.full(
            (3, source_rank),
            rank + 1,
            dtype=DTYPE,
            device=device,
        )
        arena = prepared.gather(local, local_is_feature_major=False)
        c1_error = 0.0
        for source in range(TP_SIZE):
            block = arena.narrow(0, source * source_rank, source_rank)
            c1_error = max(c1_error, _constant_error(block, float(source + 1)))
        c1_error = _global_max(c1_error, device=device)

        lr_rank = int(geometry["wire_matched_lr_rank"])
        lr = torch.full(
            (3, lr_rank),
            rank + 1,
            dtype=DTYPE,
            device=device,
        )
        dist.all_reduce(lr)
        lr_error = _global_max(
            _constant_error(lr, expected_sum),
            device=device,
        )
        maximum_error = max(maximum_error, c1_error, lr_error)
        records.append(
            {
                "source_rank": source_rank,
                "wire_matched_lr_rank": lr_rank,
                "rows": 3,
                "c1_source_order_maximum_absolute_error": c1_error,
                "wire_lr_maximum_absolute_error": lr_error,
            }
        )
    if maximum_error != 0.0:
        raise AssertionError(f"collective correctness gate failed: {maximum_error}")
    return {
        "status": "passed",
        "dense_allreduce_maximum_absolute_error": dense_error,
        "maximum_absolute_error": maximum_error,
        "rank_records": records,
    }


def _arm_record(
    *,
    operation: str,
    timing_scope: str,
    payload_bytes: int,
    traffic_bytes_per_rank: float,
    timings_ms: Sequence[float],
) -> dict[str, Any]:
    timing = _summary(timings_ms)
    return {
        "operation": operation,
        "timing_scope": timing_scope,
        "local_payload_bytes": int(payload_bytes),
        "ideal_ring_traffic_bytes_per_rank": float(traffic_bytes_per_rank),
        "timing": timing,
        "effective_ideal_ring_traffic_gbps_at_p50": (
            traffic_bytes_per_rank / (timing["p50_ms"] / 1000.0) / 1.0e9
        ),
    }


def _time(
    function: Callable[[], Any],
    *,
    warmup: int,
    iterations: int,
    device: torch.device,
) -> list[float]:
    return _critical_cuda_timings(
        function,
        warmup=warmup,
        iterations=iterations,
        device=device,
    )


@torch.inference_mode()
def _benchmark(
    communicator: FeatureRaggedCommunicator,
    *,
    source_ranks: Sequence[int],
    token_rows: Sequence[int],
    warmup: int,
    iterations: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    dense_timings: dict[int, list[float]] = {}
    for rows in token_rows:
        dense = torch.zeros((rows, HIDDEN_SIZE), dtype=DTYPE, device=device)
        dense_timings[rows] = _time(
            lambda dense=dense: dist.all_reduce(dense),
            warmup=warmup,
            iterations=iterations,
            device=device,
        )

    records = []
    for source_rank in source_ranks:
        plan = StaticRaggedPlan.from_source_widths((source_rank,) * TP_SIZE)
        for rows in token_rows:
            geometry = collective_geometry(source_rank=source_rank, rows=rows)
            prepared = communicator.prepare_uniform(
                plan,
                tokens=rows,
                dtype=DTYPE,
                backend="uniform_nccl",
            )
            token_major = torch.zeros(
                (rows, source_rank),
                dtype=DTYPE,
                device=device,
            )
            prepared.local_feature_major_view().zero_()
            lr_rank = int(geometry["wire_matched_lr_rank"])
            lr = torch.zeros((rows, lr_rank), dtype=DTYPE, device=device)

            collective_only = _time(
                prepared.gather_inplace_fast,
                warmup=warmup,
                iterations=iterations,
                device=device,
            )
            packing_plus_collective = _time(
                lambda prepared=prepared, local=token_major: prepared.gather(
                    local,
                    local_is_feature_major=False,
                ),
                warmup=warmup,
                iterations=iterations,
                device=device,
            )
            lr_timings = _time(
                lambda lr=lr: dist.all_reduce(lr),
                warmup=warmup,
                iterations=iterations,
                device=device,
            )

            dense_traffic = float(geometry["dense_ideal_ring_bytes_per_rank"])
            matched_traffic = float(
                geometry["c1_and_wire_lr_ideal_ring_bytes_per_rank"]
            )
            arms = {
                "dense_allreduce": _arm_record(
                    operation="NCCL AllReduce",
                    timing_scope="collective only",
                    payload_bytes=int(geometry["dense_allreduce_payload_bytes"]),
                    traffic_bytes_per_rank=dense_traffic,
                    timings_ms=dense_timings[rows],
                ),
                "c1_allgather_collective_only": _arm_record(
                    operation="packed feature-major NCCL AllGather",
                    timing_scope="collective only; source slot already populated",
                    payload_bytes=int(geometry["c1_allgather_local_payload_bytes"]),
                    traffic_bytes_per_rank=matched_traffic,
                    timings_ms=collective_only,
                ),
                "c1_packing_plus_allgather": _arm_record(
                    operation="token-major pack + packed feature-major NCCL AllGather",
                    timing_scope="layout packing plus collective",
                    payload_bytes=int(geometry["c1_allgather_local_payload_bytes"]),
                    traffic_bytes_per_rank=matched_traffic,
                    timings_ms=packing_plus_collective,
                ),
                "wire_lr_allreduce": _arm_record(
                    operation="NCCL AllReduce",
                    timing_scope="collective only",
                    payload_bytes=int(geometry["wire_lr_allreduce_payload_bytes"]),
                    traffic_bytes_per_rank=matched_traffic,
                    timings_ms=lr_timings,
                ),
            }
            record = {
                "source_rank": source_rank,
                "wire_matched_lr_rank": lr_rank,
                "token_rows": rows,
                "geometry": geometry,
                "arms": arms,
                "comparisons": comparison_metrics(arms),
            }
            records.append(record)
            if dist.get_rank() == 0:
                print(
                    json.dumps(
                        {
                            "event": "configuration_complete",
                            "source_rank": source_rank,
                            "wire_lr_rank": lr_rank,
                            "token_rows": rows,
                            "c1_collective_p50_ms": arms[
                                "c1_allgather_collective_only"
                            ]["timing"]["p50_ms"],
                            "c1_packed_p50_ms": arms["c1_packing_plus_allgather"][
                                "timing"
                            ]["p50_ms"],
                            "wire_lr_p50_ms": arms["wire_lr_allreduce"]["timing"][
                                "p50_ms"
                            ],
                        }
                    ),
                    flush=True,
                )
    return records


def _summary_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Qwen3-8B Wo-only TP4 collective microbenchmark",
        "",
        (
            "BF16, four GPUs. C1-AllGather and LR-AllReduce use exactly equal "
            "ideal ring bytes for every row. Lower latency is better."
        ),
        "",
        "| C1 rank | LR rank | Rows | Dense AR ms | C1 collective ms | C1 packed ms | LR AR ms | C1 packed vs LR | Packing overhead |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in payload["records"]:
        arms = record["arms"]
        comparisons = record["comparisons"]
        lines.append(
            f"| {record['source_rank']} | {record['wire_matched_lr_rank']} | "
            f"{record['token_rows']} | "
            f"{arms['dense_allreduce']['timing']['p50_ms']:.6f} | "
            f"{arms['c1_allgather_collective_only']['timing']['p50_ms']:.6f} | "
            f"{arms['c1_packing_plus_allgather']['timing']['p50_ms']:.6f} | "
            f"{arms['wire_lr_allreduce']['timing']['p50_ms']:.6f} | "
            f"{100.0 * comparisons['c1_packed_latency_change_vs_wire_lr']:+.2f}% | "
            f"{100.0 * comparisons['c1_packing_overhead_vs_collective_only']:+.2f}% |"
        )
    lines.extend(
        [
            "",
            "## Protocol",
            "",
            f"- Correctness gate: `{payload['correctness_gate']['status']}`",
            f"- Warmup calls per arm: `{payload['protocol']['warmup']}`",
            f"- Measured iterations per arm: `{payload['protocol']['iterations']}`",
            "- Timing: maximum CUDA-event elapsed time across TP ranks",
            "- Dense V/KV cache: unchanged; this benchmark isolates only the Wo collective",
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
    parser.add_argument("--source-ranks", default=DEFAULT_SOURCE_RANKS)
    parser.add_argument("--token-rows", default=DEFAULT_TOKEN_ROWS)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--torch-num-threads", type=int, default=2)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    source_ranks = parse_positive_csv(args.source_ranks, label="source ranks")
    token_rows = parse_positive_csv(args.token_rows, label="token rows")
    if max(source_ranks) > SOURCE_WIDTH:
        raise ValueError(f"source ranks cannot exceed {SOURCE_WIDTH}")
    if min(args.warmup, args.iterations, args.torch_num_threads) <= 0:
        raise ValueError("warmup, iterations, and thread count must be positive")

    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    torch.set_num_threads(args.torch_num_threads)
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    if dist.get_world_size() != TP_SIZE:
        raise RuntimeError(f"collective benchmark requires exactly TP{TP_SIZE}")

    communicator: FeatureRaggedCommunicator | None = None
    try:
        communicator = FeatureRaggedCommunicator.from_distributed(device=device)
        communicator.configure_direct_workspace(
            tokens=max(token_rows),
            max_total_width=TP_SIZE * max(source_ranks),
            dtype=DTYPE,
        )
        correctness = _correctness_gates(
            communicator,
            source_ranks=source_ranks,
            device=device,
        )
        records = _benchmark(
            communicator,
            source_ranks=source_ranks,
            token_rows=token_rows,
            warmup=args.warmup,
            iterations=args.iterations,
            device=device,
        )
        if dist.get_rank() == 0:
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
                    "source_ranks": list(source_ranks),
                    "wire_matched_lr_ranks": [
                        wire_matched_allreduce_rank((rank,) * TP_SIZE)
                        for rank in source_ranks
                    ],
                    "token_rows": list(token_rows),
                    "dtype": "bfloat16",
                    "dtype_bytes": DTYPE_BYTES,
                    "warmup": args.warmup,
                    "iterations": args.iterations,
                    "timing": (
                        "maximum CUDA-event elapsed time across TP ranks per call"
                    ),
                    "dense_qkv_and_kv_cache": True,
                },
                "correctness_gate": correctness,
                "records": records,
                "environment": {
                    "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
                    "gpu": torch.cuda.get_device_name(device),
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                    "nccl": ".".join(map(str, torch.cuda.nccl.version())),
                    "nccl_environment": {
                        key: value
                        for key, value in os.environ.items()
                        if key.startswith("NCCL_")
                    },
                    "nvidia_smi_topology": _command_output(
                        ("nvidia-smi", "topo", "-m")
                    ),
                },
            }
            output_dir.mkdir(parents=True)
            _atomic_text(
                output_dir / "results.json",
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
            )
            _atomic_text(output_dir / "summary.md", _summary_markdown(payload))
            print(
                json.dumps(
                    {
                        "event": "result_written",
                        "output_dir": str(output_dir),
                    }
                ),
                flush=True,
            )
        dist.barrier()
    finally:
        if communicator is not None:
            communicator.close()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
