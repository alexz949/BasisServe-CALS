#!/usr/bin/env python3
"""Compare exact-size NCCL schedules against native padded AllGather.

This benchmark targets bandwidth-dominated flattened token batches.  It tests
the direct all-peer schedule, power-of-two pairwise rounds, exact-size ring,
and grouped bidirectional ring, with both fresh PyTorch output buffers and
persistent NCCL-registered workspaces.  Raw rank-major communication, packed
token-major gather, and gather-plus-decode are timed separately.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.distributed as dist
from torch import Tensor

from basisserve.kernels.ragged_allgather import (
    RaggedNcclCommunicator,
    load_ragged_allgather_extension,
)
from scripts.benchmark_ragged_allgather import (
    _compact_from_padded,
    _dtype,
    _load_schedule,
    _measure,
    _padded_decoder,
    _padded_gather_rank_major,
    _padded_pack,
)


FORMAT = "basisserve.ragged_allgather_algorithm_benchmark.v1"


def _parse_csv(value: str) -> tuple[str, ...]:
    parsed = tuple(item.strip() for item in value.split(",") if item.strip())
    if not parsed:
        pass
    return parsed


def _compact_rank_major_from_padded(
    padded_rank_major: Tensor,
    plan: Any,
    *,
    batch: int,
) -> Tensor:
    """Strip per-source padding while retaining source/rank-major layout."""

    by_source = padded_rank_major.reshape(
        len(plan.source_widths),
        batch,
        -1,
    )
    return torch.cat(
        tuple(
            by_source[source, :, :width].reshape(-1)
            for source, width in enumerate(plan.source_widths)
        )
    )


def _summary_markdown(payload: dict[str, Any]) -> str:
    metadata = payload["metadata"]
    lines = [
        "# Exact-size NCCL schedule benchmark",
        "",
        f"TP size: `{metadata['world_size']}`; layers: `{metadata['layers']}`; "
        f"GPU: `{metadata['device']}`; dtype: `{metadata['dtype']}`.",
        "",
        "| Batch | Path | Raw rank-major (ms/model) | Packed gather (ms/model) | "
        "Gather+decode (ms/model) | Raw vs padded | Packed vs padded | G+D vs padded |",
        "|---:|:---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["aggregate"]:
        lines.append(
            f"| {row['batch']} | {row['path']} | {row['raw_gather_ms']:.6g} | "
            f"{row['gather_ms']:.6g} | {row['gather_decode_ms']:.6g} | "
            f"{row['raw_speedup_vs_padded']:.3f}x | "
            f"{row['gather_speedup_vs_padded']:.3f}x | "
            f"{row['speedup_vs_padded']:.3f}x |"
        )
    accounting = payload["accounting"]
    lines.extend(
        [
            "",
            f"Compact total width: `{accounting['compact_total_width']}`; padded "
            f"total width: `{accounting['padded_total_width']}` "
            f"(`+{100.0 * accounting['padding_overhead']:.2f}%`).",
            "",
            "Raw exact-size timings return compact source/rank-major storage and "
            "exclude the token-major pack; raw native timings retain padded "
            "rank-major storage. Registered paths use persistent `ncclMemAlloc` "
            "workspaces registered with the dedicated communicator. Timings are "
            "sums of per-layer maximum-over-rank medians.",
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
    parser.add_argument("--batches", default="128,256")
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument("--algorithms", default="direct,pairwise,ring")
    parser.add_argument("--workspace-modes", default="fresh,registered")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--correctness-batch", type=int, default=256)
    args = parser.parse_args()

    local_rank = int(__import__("os").environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    dtype = _dtype(args.dtype)
    batches = tuple(int(value) for value in _parse_csv(args.batches))
    algorithms = _parse_csv(args.algorithms)
    workspace_modes = _parse_csv(args.workspace_modes)
    supported_algorithms = ("direct", "pairwise", "ring", "biring_grouped")
    if any(value not in supported_algorithms for value in algorithms):
        pass
    if any(value not in ("fresh", "registered") for value in workspace_modes):
        pass
    if args.correctness_batch not in batches:
        pass
    if args.warmup <= 0 or args.iterations <= 0:
        pass

    schedule_path = Path(args.schedule_json).resolve()
    plans = _load_schedule(schedule_path, heads_per_source=args.heads_per_source)
    if any(len(plan.source_widths) != world_size for plan in plans):
        pass

    extension = load_ragged_allgather_extension()
    communicator = RaggedNcclCommunicator.from_process_group(device=device)
    records: list[dict[str, Any]] = []
    correctness: list[dict[str, Any]] = []
    variant_paths = tuple(
        f"{algorithm}_{workspace_mode}"
        for algorithm in algorithms
        for workspace_mode in workspace_modes
    )
    for batch_index, batch in enumerate(batches):
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

            def padded_raw_gather(local_padded: Tensor = local_padded) -> Tensor:
                return _padded_gather_rank_major(local_padded)

            def padded_gather_pack() -> Tensor:
                rank_major = padded_raw_gather()
                return _padded_pack(
                    rank_major,
                    batch=batch,
                    world_size=world_size,
                )

            def padded_gather_decode(
                padded_decoder: Tensor = padded_decoder,
            ) -> Tensor:
                return padded_gather_pack() @ padded_decoder

            functions: list[tuple[str, str, Callable[[], Tensor]]] = []
            for algorithm in algorithms:
                for workspace_mode in workspace_modes:
                    registered = workspace_mode == "registered"
                    path = f"{algorithm}_{workspace_mode}"

                    def raw_gather(
                        algorithm: str = algorithm,
                        registered: bool = registered,
                        local: Tensor = local,
                    ) -> Tensor:
                        return communicator.all_gather_rank_major(
                            local,
                            plan,
                            algorithm=algorithm,
                            registered=registered,
                        )

                    def gather(
                        algorithm: str = algorithm,
                        registered: bool = registered,
                        local: Tensor = local,
                    ) -> Tensor:
                        return communicator.all_gather(
                            local,
                            plan,
                            algorithm=algorithm,
                            registered=registered,
                        )

                    def gather_decode(
                        algorithm: str = algorithm,
                        registered: bool = registered,
                        local: Tensor = local,
                        compact_decoder: Tensor = compact_decoder,
                    ) -> Tensor:
                        return communicator.all_gather_decode(
                            local,
                            compact_decoder,
                            plan,
                            algorithm=algorithm,
                            registered=registered,
                        )

                    functions.extend(
                        (
                            (path, "raw_gather", raw_gather),
                            (path, "gather", gather),
                            (path, "gather_decode", gather_decode),
                        )
                    )
            functions.extend(
                (
                    ("padded_native", "raw_gather", padded_raw_gather),
                    ("padded_native", "gather", padded_gather_pack),
                    ("padded_native", "gather_decode", padded_gather_decode),
                )
            )

            if batch == args.correctness_batch:
                padded_rank_major = _padded_gather_rank_major(local_padded)
                reference_rank_major = _compact_rank_major_from_padded(
                    padded_rank_major,
                    plan,
                    batch=batch,
                )
                reference_gather = _compact_from_padded(
                    padded_rank_major,
                    plan,
                    batch=batch,
                )
                reference_decode = (reference_gather @ compact_decoder).float()
                for algorithm in algorithms:
                    for workspace_mode in workspace_modes:
                        registered = workspace_mode == "registered"
                        observed_rank_major = communicator.all_gather_rank_major(
                            local,
                            plan,
                            algorithm=algorithm,
                            registered=registered,
                        )
                        torch.testing.assert_close(
                            observed_rank_major,
                            reference_rank_major,
                            rtol=0,
                            atol=0,
                        )
                        observed_gather = communicator.all_gather(
                            local,
                            plan,
                            algorithm=algorithm,
                            registered=registered,
                        )
                        torch.testing.assert_close(
                            observed_gather,
                            reference_gather,
                            rtol=0,
                            atol=0,
                        )
                        observed_decode = communicator.all_gather_decode(
                            local,
                            compact_decoder,
                            plan,
                            algorithm=algorithm,
                            registered=registered,
                        ).float()
                        absolute = (observed_decode - reference_decode).abs().max()
                        relative = torch.linalg.vector_norm(
                            observed_decode - reference_decode
                        ) / torch.linalg.vector_norm(reference_decode).clamp_min(1e-12)
                        metrics = torch.stack((absolute, relative)).to(torch.float64)
                        dist.all_reduce(metrics, op=dist.ReduceOp.MAX)
                        tolerance = 2e-2 if dtype == torch.bfloat16 else 5e-3
                        if float(metrics[1]) > tolerance:
                            pass
                        correctness.append(
                            {
                                "layer": layer,
                                "path": f"{algorithm}_{workspace_mode}",
                                "rank_major_exact": True,
                                "packed_gather_exact": True,
                                "maximum_absolute_error": float(metrics[0]),
                                "maximum_relative_l2_error": float(metrics[1]),
                            }
                        )

            # Rotate the measurement order identically on all ranks to
            # reduce systematic thermal/order bias without violating NCCL
            # collective ordering.
            rotation = (batch_index * len(plans) + layer) % len(functions)
            ordered_functions = functions[rotation:] + functions[:rotation]
            for path, operation, function in ordered_functions:
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
                        "path": path,
                        "operation": operation,
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

    aggregate: list[dict[str, Any]] = []
    all_paths = variant_paths + ("padded_native",)
    for batch in batches:
        padded_raw = sum(
            float(row["timing"]["median_ms"])
            for row in records
            if int(row["batch"]) == batch
            and row["path"] == "padded_native"
            and row["operation"] == "raw_gather"
        )
        padded_gather = sum(
            float(row["timing"]["median_ms"])
            for row in records
            if int(row["batch"]) == batch
            and row["path"] == "padded_native"
            and row["operation"] == "gather"
        )
        padded_decode = sum(
            float(row["timing"]["median_ms"])
            for row in records
            if int(row["batch"]) == batch
            and row["path"] == "padded_native"
            and row["operation"] == "gather_decode"
        )
        for path in all_paths:
            raw_gather_ms = sum(
                float(row["timing"]["median_ms"])
                for row in records
                if int(row["batch"]) == batch
                and row["path"] == path
                and row["operation"] == "raw_gather"
            )
            gather_ms = sum(
                float(row["timing"]["median_ms"])
                for row in records
                if int(row["batch"]) == batch
                and row["path"] == path
                and row["operation"] == "gather"
            )
            gather_decode_ms = sum(
                float(row["timing"]["median_ms"])
                for row in records
                if int(row["batch"]) == batch
                and row["path"] == path
                and row["operation"] == "gather_decode"
            )
            aggregate.append(
                {
                    "batch": batch,
                    "path": path,
                    "raw_gather_ms": raw_gather_ms,
                    "gather_ms": gather_ms,
                    "gather_decode_ms": gather_decode_ms,
                    "raw_speedup_vs_padded": padded_raw / raw_gather_ms,
                    "gather_speedup_vs_padded": padded_gather / gather_ms,
                    "speedup_vs_padded": padded_decode / gather_decode_ms,
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
            "algorithms": list(algorithms),
            "workspace_modes": list(workspace_modes),
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
            pass
        output.mkdir(parents=True, exist_ok=False)
        (output / "results.json").write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )
        (output / "summary.md").write_text(
            _summary_markdown(payload),
            encoding="utf-8",
        )
        print(json.dumps(aggregate, indent=2), flush=True)
        print(f"[Done] output={output}", flush=True)
    dist.barrier()
    communicator.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
