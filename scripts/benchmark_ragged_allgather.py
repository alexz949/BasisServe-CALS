#!/usr/bin/env python3
"""Benchmark the compiled ragged NCCL AllGather on a static TP schedule.

The comparison is deliberately end-to-end at the output-collective boundary:

* ``ragged`` communicates each source's exact width, runs the CUDA layout
  kernel for batches above one, and issues one compact decoder GEMM;
* ``padded`` uses the existing equal-width ``all_gather_into_tensor`` shape,
  rank-major-to-token-major packing, and one zero-padded decoder GEMM.

Both paths reconstruct the same output.  Reported model times are sums of
per-layer maximum-over-rank medians for the supplied 32-layer schedule.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys
from typing import Any, Callable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.distributed as dist
from torch import Tensor

from basisserve.kernels.ragged_allgather import (
    RaggedNcclCommunicator,
    StaticRaggedPlan,
    load_ragged_allgather_extension,
)


FORMAT = "basisserve.ragged_allgather_benchmark.v1"


def _dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def _load_schedule(path: Path, *, heads_per_source: int) -> tuple[StaticRaggedPlan, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    selection = payload.get("selection")
    schedule = payload.get("selected_schedule")
    if schedule is None and isinstance(selection, dict):
        schedule = selection.get("selected_schedule")
    if not isinstance(schedule, list) or not schedule:
        raise ValueError(
            f"selected_schedule is absent from both the top level and selection in {path}"
        )
    plans = tuple(
        StaticRaggedPlan.from_head_ranks(
            tuple(int(rank) for rank in layer),
            heads_per_source=heads_per_source,
        )
        for layer in schedule
    )
    world_size = len(plans[0].source_widths)
    if any(len(plan.source_widths) != world_size for plan in plans):
        raise ValueError("schedule world size changes between layers")
    return plans


def _quantile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    index = min(int(fraction * len(ordered)), len(ordered) - 1)
    return ordered[index]


def _measure(
    function: Callable[[], Tensor],
    *,
    warmup: int,
    iterations: int,
    device: torch.device,
) -> dict[str, float]:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize(device)
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    result: Tensor | None = None
    for start, end in zip(starts, ends):
        start.record()
        result = function()
        end.record()
    torch.cuda.synchronize(device)
    if result is None:
        raise AssertionError("benchmark did not execute")
    local = [float(start.elapsed_time(end)) for start, end in zip(starts, ends)]
    metrics = torch.tensor(
        [min(local), statistics.median(local), _quantile(local, 0.9), max(local)],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(metrics, op=dist.ReduceOp.MAX)
    return {
        "minimum_ms": float(metrics[0]),
        "median_ms": float(metrics[1]),
        "p90_ms": float(metrics[2]),
        "maximum_ms": float(metrics[3]),
    }


def _padded_gather_rank_major(local_padded: Tensor) -> Tensor:
    world_size = dist.get_world_size()
    output = torch.empty(
        world_size * local_padded.shape[0],
        local_padded.shape[1],
        dtype=local_padded.dtype,
        device=local_padded.device,
    )
    dist.all_gather_into_tensor(output, local_padded)
    return output


def _padded_pack(rank_major: Tensor, *, batch: int, world_size: int) -> Tensor:
    maximum_width = int(rank_major.shape[1])
    return (
        rank_major.reshape(world_size, batch, maximum_width)
        .permute(1, 0, 2)
        .contiguous()
        .reshape(batch, world_size * maximum_width)
    )


def _compact_from_padded(
    rank_major: Tensor,
    plan: StaticRaggedPlan,
    *,
    batch: int,
) -> Tensor:
    world_size = len(plan.source_widths)
    maximum_width = int(rank_major.shape[1])
    by_source = rank_major.reshape(world_size, batch, maximum_width)
    return torch.cat(
        tuple(
            by_source[source, :, :width]
            for source, width in enumerate(plan.source_widths)
        ),
        dim=1,
    )


def _padded_decoder(compact: Tensor, plan: StaticRaggedPlan) -> Tensor:
    maximum_width = max(plan.source_widths)
    output = torch.zeros(
        len(plan.source_widths) * maximum_width,
        compact.shape[1],
        dtype=compact.dtype,
        device=compact.device,
    )
    for source, width in enumerate(plan.source_widths):
        compact_start = plan.offsets[source]
        padded_start = source * maximum_width
        output[padded_start : padded_start + width].copy_(
            compact[compact_start : compact_start + width]
        )
    return output


def _summary_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Static-ragged NCCL AllGather benchmark",
        "",
        f"TP size: `{payload['metadata']['world_size']}`; "
        f"layers: `{payload['metadata']['layers']}`; "
        f"GPU: `{payload['metadata']['device']}`.",
        "",
        "| Batch | Ragged gather+decode (ms/model) | Padded gather+decode (ms/model) | Speedup | Ragged gather (ms/model) | Padded gather+pack (ms/model) |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["aggregate"]:
        lines.append(
            f"| {row['batch']} | {row['ragged_gather_decode_ms']:.6g} | "
            f"{row['padded_gather_decode_ms']:.6g} | {row['speedup']:.3f}x | "
            f"{row['ragged_gather_ms']:.6g} | {row['padded_gather_pack_ms']:.6g} |"
        )
    accounting = payload["accounting"]
    lines.extend(
        [
            "",
            f"Exact compact width over all layers: `{accounting['compact_total_width']}`; "
            f"padded width: `{accounting['padded_total_width']}` "
            f"(`+{100.0 * accounting['padding_overhead']:.2f}%`).",
            "",
            "The ragged backend is a dedicated NCCL communicator using grouped "
            "exact-count send/recv, a CUDA layout kernel for batch > 1, and one "
            "compact GEMM. It does not call Python distributed P2P in the timed path.",
        ]
    )
    return "\n".join(lines) + "\n"


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--heads-per-source", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--batches", default="1,4,32,128")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--correctness-batch", type=int, default=4)
    args = parser.parse_args()

    local_rank = int(__import__("os").environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    dtype = _dtype(args.dtype)
    batches = tuple(int(value) for value in args.batches.split(","))
    if args.warmup <= 0 or args.iterations <= 0:
        raise ValueError("warmup and iterations must be positive")

    schedule_path = Path(args.schedule_json).resolve()
    plans = _load_schedule(schedule_path, heads_per_source=args.heads_per_source)
    if any(len(plan.source_widths) != world_size for plan in plans):
        raise ValueError(f"schedule is not compatible with TP{world_size}")

    extension = load_ragged_allgather_extension()
    communicator = RaggedNcclCommunicator.from_process_group(device=device)
    records: list[dict[str, Any]] = []
    correctness: list[dict[str, float | int]] = []
    try:
        for batch in batches:
            for layer, plan in enumerate(plans):
                generator = torch.Generator(device=device).manual_seed(20260819 + layer)
                local_generator = torch.Generator(device=device).manual_seed(
                    2026081900 + 1000 * layer + rank
                )
                local = torch.randn(
                    batch,
                    plan.source_widths[rank],
                    dtype=dtype,
                    device=device,
                    generator=local_generator,
                )
                compact_decoder = torch.randn(
                    plan.total_width,
                    args.hidden_size,
                    dtype=dtype,
                    device=device,
                    generator=generator,
                ) / plan.total_width**0.5
                maximum_width = max(plan.source_widths)
                local_padded = torch.zeros(
                    batch,
                    maximum_width,
                    dtype=dtype,
                    device=device,
                )
                local_padded[:, : local.shape[1]].copy_(local)
                padded_decoder = _padded_decoder(compact_decoder, plan)

                def ragged_gather() -> Tensor:
                    return communicator.all_gather(local, plan)

                def ragged_gather_decode() -> Tensor:
                    return communicator.all_gather_decode(
                        local,
                        compact_decoder,
                        plan,
                    )

                def padded_gather_pack() -> Tensor:
                    return _padded_pack(
                        _padded_gather_rank_major(local_padded),
                        batch=batch,
                        world_size=world_size,
                    )

                def padded_gather_decode() -> Tensor:
                    return padded_gather_pack() @ padded_decoder

                if batch == args.correctness_batch:
                    observed_gather = ragged_gather()
                    padded_rank_major = _padded_gather_rank_major(local_padded)
                    reference_gather = _compact_from_padded(
                        padded_rank_major,
                        plan,
                        batch=batch,
                    )
                    torch.testing.assert_close(
                        observed_gather,
                        reference_gather,
                        rtol=0,
                        atol=0,
                    )
                    observed = ragged_gather_decode().float()
                    reference = (reference_gather @ compact_decoder).float()
                    absolute = (observed - reference).abs().max()
                    relative = torch.linalg.vector_norm(observed - reference) / (
                        torch.linalg.vector_norm(reference).clamp_min(1e-12)
                    )
                    metrics = torch.stack((absolute, relative)).to(torch.float64)
                    dist.all_reduce(metrics, op=dist.ReduceOp.MAX)
                    tolerance = 2e-2 if dtype == torch.bfloat16 else 5e-3
                    if float(metrics[1]) > tolerance:
                        raise AssertionError(
                            f"layer {layer} relative error {float(metrics[1]):.6e}"
                        )
                    correctness.append(
                        {
                            "layer": layer,
                            "maximum_absolute_error": float(metrics[0]),
                            "maximum_relative_l2_error": float(metrics[1]),
                        }
                    )

                functions = (
                    ("ragged_gather", ragged_gather),
                    ("ragged_gather_decode", ragged_gather_decode),
                    ("padded_gather_pack", padded_gather_pack),
                    ("padded_gather_decode", padded_gather_decode),
                )
                for name, function in functions:
                    dist.barrier()
                    timing = _measure(
                        function,
                        warmup=args.warmup,
                        iterations=args.iterations,
                        device=device,
                    )
                    records.append(
                        {
                            "batch": batch,
                            "layer": layer,
                            "name": name,
                            "source_widths": list(plan.source_widths),
                            "compact_width": plan.total_width,
                            "padded_width": plan.padded_total_width,
                            "timing": timing,
                        }
                    )
                if rank == 0:
                    print(
                        f"[Layer] batch={batch} layer={layer} "
                        f"compact={plan.total_width} padded={plan.padded_total_width}",
                        flush=True,
                    )
                del local, compact_decoder, local_padded, padded_decoder

        aggregate: list[dict[str, float | int]] = []
        for batch in batches:
            by_name = {
                name: sum(
                    float(row["timing"]["median_ms"])
                    for row in records
                    if int(row["batch"]) == batch and row["name"] == name
                )
                for name in (
                    "ragged_gather",
                    "ragged_gather_decode",
                    "padded_gather_pack",
                    "padded_gather_decode",
                )
            }
            aggregate.append(
                {
                    "batch": batch,
                    "ragged_gather_ms": by_name["ragged_gather"],
                    "ragged_gather_decode_ms": by_name["ragged_gather_decode"],
                    "padded_gather_pack_ms": by_name["padded_gather_pack"],
                    "padded_gather_decode_ms": by_name["padded_gather_decode"],
                    "speedup": (
                        by_name["padded_gather_decode"]
                        / by_name["ragged_gather_decode"]
                    ),
                }
            )

        compact_total = sum(plan.total_width for plan in plans)
        padded_total = sum(plan.padded_total_width for plan in plans)
        payload = {
            "format": FORMAT,
            "status": "complete",
            "metadata": {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "command": " ".join(sys.argv),
                "world_size": world_size,
                "layers": len(plans),
                "device": torch.cuda.get_device_name(device),
                "dtype": args.dtype,
                "hidden_size": args.hidden_size,
                "batches": list(batches),
                "warmup": args.warmup,
                "iterations": args.iterations,
                "torch_version": torch.__version__,
                "torch_cuda_version": torch.version.cuda,
                "nccl_version": int(extension.nccl_version()),
                "schedule_json": str(schedule_path),
            },
            "accounting": {
                "compact_total_width": compact_total,
                "padded_total_width": padded_total,
                "padding_overhead": padded_total / compact_total - 1.0,
            },
            "correctness": correctness,
            "aggregate": aggregate,
            "records": records,
        }
        if rank == 0:
            output = Path(args.output_dir).resolve()
            if output.exists():
                raise FileExistsError(f"refusing to overwrite {output}")
            output.mkdir(parents=True, exist_ok=False)
            (output / "results.json").write_text(
                json.dumps(payload, indent=2) + "\n",
                encoding="utf-8",
            )
            (output / "summary.md").write_text(
                _summary_markdown(payload),
                encoding="utf-8",
            )
            print(json.dumps(payload["aggregate"], indent=2), flush=True)
            print(f"[Done] output={output}", flush=True)
    finally:
        dist.barrier()
        communicator.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
