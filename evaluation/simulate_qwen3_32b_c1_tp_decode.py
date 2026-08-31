#!/usr/bin/env python3
"""Single-GPU semantic simulation of the Qwen3-32B C1 TP8 decode boundary.

Eight virtual ranks use the real per-source C1 factors, but execute
sequentially on one GPU.  The simulator proves tensor ownership and decoder
ordering without requiring an eight-GPU allocation.  It reports:

* measured sequential single-GPU latency;
* an isolated parallel-compute floor built from the slowest virtual rank;
* feature-major arena construction and one-big-GEMM equivalence;
* exact logical collective byte counts for TP8.

It never models or reports NCCL latency, and its rows/s are not full-model
tokens/s or measured TP8 throughput.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shlex
import statistics
import sys
import time
from typing import Any, Callable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
from torch import Tensor  # noqa: E402
from torch.nn import functional as F  # noqa: E402

from basisserve.core.c1_tp_decode import (  # noqa: E402
    C1TPFactorLoader,
    PackedC1TPLayer,
)
from basisserve.kernels.feature_ragged_allgather import (  # noqa: E402
    decode_feature_major,
)
from evaluation.benchmark_qwen3_32b_c1_tp_decode import (  # noqa: E402
    _dense_local_decoder,
    _dtype,
    _padded_decoder,
    _parse_layers,
    _parse_positive_ints,
)


FORMAT = "basisserve.qwen3_32b.c1_tp_decode_single_gpu_simulation.v2"


def _quantile(values: Sequence[float], fraction: float) -> float:
    if not values:
        raise ValueError("cannot summarize empty timings")
    ordered = sorted(map(float, values))
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def _measure(
    function: Callable[[], Tensor],
    *,
    warmup: int,
    iterations: int,
    device: torch.device,
) -> dict[str, float]:
    result: Tensor | None = None
    for _ in range(warmup):
        result = function()
    torch.cuda.synchronize(device)
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    for start, end in zip(starts, ends):
        start.record()
        result = function()
        end.record()
    torch.cuda.synchronize(device)
    if result is None:
        raise AssertionError("simulation function did not execute")
    timings = [start.elapsed_time(end) for start, end in zip(starts, ends)]
    return {
        "minimum_ms": min(timings),
        "p50_ms": statistics.median(timings),
        "p95_ms": _quantile(timings, 0.95),
        "maximum_ms": max(timings),
    }


def _sum_outputs(outputs: Sequence[Tensor]) -> Tensor:
    if not outputs:
        raise ValueError("virtual TP output list cannot be empty")
    result = outputs[0]
    for value in outputs[1:]:
        result = result + value
    return result


def _compact_latent(
    packed: Sequence[PackedC1TPLayer],
    attention: Sequence[Tensor],
) -> Tensor:
    return torch.cat(
        tuple(
            source.encode_local_attention(local_attention)
            for source, local_attention in zip(packed, attention)
        ),
        dim=1,
    )


def _padded_latent(
    packed: Sequence[PackedC1TPLayer],
    attention: Sequence[Tensor],
) -> Tensor:
    maximum_width = max(packed[0].plan.source_widths)
    blocks = []
    for source, local_attention in zip(packed, attention):
        latent = source.encode_local_attention(local_attention)
        blocks.append(F.pad(latent, (0, maximum_width - latent.shape[1])))
    return torch.cat(blocks, dim=1)


def _write_feature_major_arena(
    arena: Tensor,
    packed: Sequence[PackedC1TPLayer],
    attention: Sequence[Tensor],
) -> Tensor:
    """Encode virtual sources directly into a reusable ``[K, rows]`` arena."""

    if not packed or len(packed) != len(attention):
        raise ValueError("packed factors and attention must define the same sources")
    plan = packed[0].plan
    rows = int(attention[0].shape[0])
    expected_shape = (plan.total_width, rows)
    if tuple(arena.shape) != expected_shape:
        raise ValueError(
            f"feature-major arena must have shape {expected_shape}, "
            f"got {tuple(arena.shape)}"
        )
    for source, local_attention in zip(packed, attention):
        local = source.encode_local_attention(local_attention)
        start = plan.offsets[source.process_rank]
        width = plan.source_widths[source.process_rank]
        arena.narrow(0, start, width).copy_(local.transpose(0, 1))
    return arena


def _relative_error(observed: Tensor, reference: Tensor) -> dict[str, float]:
    difference = observed.float() - reference.float()
    return {
        "maximum_absolute_error": float(difference.abs().max()),
        "relative_l2_error": float(
            torch.linalg.vector_norm(difference)
            / torch.linalg.vector_norm(reference.float()).clamp_min(1e-30)
        ),
        "nonfinite": float(not torch.isfinite(observed).all()),
    }


def _assert_correct(
    *,
    layer: int,
    path: str,
    metrics: dict[str, float],
    relative_tolerance: float,
) -> None:
    if metrics["nonfinite"] or metrics["relative_l2_error"] > relative_tolerance:
        raise AssertionError(
            f"{path} failed at layer {layer}: {metrics}, "
            f"relative_tolerance={relative_tolerance}"
        )


def _communication_bytes(
    packed: Sequence[PackedC1TPLayer],
    *,
    batch: int,
) -> dict[str, dict[str, int | str]]:
    plan = packed[0].plan
    world_size = len(plan.source_widths)
    element_bytes = packed[0].global_decoder.element_size()
    maximum_width = max(plan.source_widths)
    hidden_size = packed[0].hidden_size
    ragged_total = (
        (world_size - 1) * batch * plan.total_width * element_bytes
    )
    padded_total = (
        (world_size - 1)
        * world_size
        * batch
        * maximum_width
        * element_bytes
    )
    ring_allreduce_total = (
        2 * (world_size - 1) * batch * hidden_size * element_bytes
    )
    return {
        "dense_projected_allreduce": {
            "model": "ring-equivalent AllReduce wire volume",
            "total_wire_bytes_all_ranks": ring_allreduce_total,
            "per_rank_sent_bytes": (
                2
                * (world_size - 1)
                * batch
                * hidden_size
                * element_bytes
                // world_size
            ),
        },
        "local_c1_allreduce": {
            "model": "ring-equivalent AllReduce wire volume",
            "total_wire_bytes_all_ranks": ring_allreduce_total,
            "per_rank_sent_bytes": (
                2
                * (world_size - 1)
                * batch
                * hidden_size
                * element_bytes
                // world_size
            ),
        },
        "compact_ragged_allgather": {
            "model": "exact payload replicated to every other rank",
            "total_wire_bytes_all_ranks": ragged_total,
            "maximum_rank_received_bytes": max(
                batch * (plan.total_width - width) * element_bytes
                for width in plan.source_widths
            ),
        },
        "padded_allgather": {
            "model": "equal-width payload replicated to every other rank",
            "total_wire_bytes_all_ranks": padded_total,
            "per_rank_received_bytes": (
                (world_size - 1) * batch * maximum_width * element_bytes
            ),
        },
    }


def _path_record(
    *,
    layer: int,
    batch: int,
    path: str,
    sequential: dict[str, float],
    parallel_compute_floor_ms: float,
    components: dict[str, Any],
    communication: dict[str, int | str],
    packed: Sequence[PackedC1TPLayer],
) -> dict[str, Any]:
    return {
        "layer": layer,
        "batch": batch,
        "path": path,
        "source_ranks": list(packed[0].source_ranks),
        "source_widths": list(packed[0].plan.source_widths),
        "sequential_single_gpu": sequential,
        "parallel_compute_floor_ms": parallel_compute_floor_ms,
        "components": components,
        "communication_accounting": communication,
    }


def _aggregate(
    records: Sequence[dict[str, Any]],
    *,
    layer_count: int,
) -> list[dict[str, Any]]:
    keys = sorted({(int(row["batch"]), str(row["path"])) for row in records})
    output: list[dict[str, Any]] = []
    for batch, path in keys:
        selected = [
            row
            for row in records
            if int(row["batch"]) == batch and str(row["path"]) == path
        ]
        if len(selected) != layer_count:
            raise AssertionError(f"incomplete layer records for {path}, batch {batch}")
        sequential_p50_ms = sum(
            float(row["sequential_single_gpu"]["p50_ms"]) for row in selected
        )
        sequential_p95_ms = sum(
            float(row["sequential_single_gpu"]["p95_ms"]) for row in selected
        )
        compute_floor_ms = sum(
            float(row["parallel_compute_floor_ms"]) for row in selected
        )
        wire_bytes = sum(
            int(row["communication_accounting"]["total_wire_bytes_all_ranks"])
            for row in selected
        )
        output.append(
            {
                "batch": batch,
                "path": path,
                "sum_layer_sequential_p50_ms": sequential_p50_ms,
                "sum_layer_sequential_p95_ms": sequential_p95_ms,
                "sum_layer_parallel_compute_floor_ms": compute_floor_ms,
                "sequential_single_gpu_rows_per_second": (
                    1000.0 * batch / sequential_p50_ms
                ),
                "parallel_compute_floor_rows_per_second_excluding_communication": (
                    1000.0 * batch / compute_floor_ms
                ),
                "logical_collective_wire_bytes_all_ranks": wire_bytes,
            }
        )
    return output


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _summary_markdown(payload: dict[str, Any]) -> str:
    metadata = payload["metadata"]
    lines = [
        "# Qwen3-32B C1 virtual-TP8 decode simulation",
        "",
        f"GPU: `{metadata['device']}`; layers: `{metadata['layer_count']}`; "
        f"dtype: `{metadata['dtype']}`.",
        "",
        "| Rows | Path | Sequential p50 (ms/model) | Parallel compute floor "
        "(ms/model) | Logical wire MiB/model |",
        "|---:|:---|---:|---:|---:|",
    ]
    for row in payload["aggregate"]:
        lines.append(
            f"| {row['batch']} | {row['path']} | "
            f"{row['sum_layer_sequential_p50_ms']:.6g} | "
            f"{row['sum_layer_parallel_compute_floor_ms']:.6g} | "
            f"{row['logical_collective_wire_bytes_all_ranks'] / (1 << 20):.6g} |"
        )
    lines.extend(
        [
            "",
            "The parallel-compute floor excludes all communication. Logical wire "
            "bytes are accounting values, not NCCL latency measurements. Neither "
            "column is measured TP8 throughput or full-model tokens/s. The "
            "feature-major path writes a reusable arena and decodes its transposed "
            "view with one GEMM.",
            "",
            "Layers were streamed one at a time; maximum measured per-layer CUDA "
            f"allocation was `{metadata['maximum_layer_peak_allocated_bytes']}` bytes.",
        ]
    )
    return "\n".join(lines) + "\n"


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factor-dir", required=True)
    parser.add_argument("--model", required=True, help="model directory or config.json")
    parser.add_argument("--expected-result-sha256")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-markdown")
    parser.add_argument("--layers", default="all", help="all or comma-separated indices")
    parser.add_argument("--batches", default="1,4,16,64")
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument("--correctness-batch", type=int, default=4)
    parser.add_argument("--relative-tolerance", type=float, default=0.05)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    if args.warmup <= 0 or args.iterations <= 0:
        raise ValueError("warmup and iterations must be positive")
    if args.relative_tolerance <= 0:
        raise ValueError("relative tolerance must be positive")
    batches = _parse_positive_ints(args.batches)
    if args.correctness_batch not in batches:
        raise ValueError("correctness batch must be present in --batches")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("C1 TP simulation requires one CUDA GPU")
    torch.cuda.set_device(device)
    dtype = _dtype(args.dtype)
    loader = C1TPFactorLoader(
        args.factor_dir,
        model_config=args.model,
        tp_size=8,
        expected_result_sha256=args.expected_result_sha256,
    )
    layers = _parse_layers(args.layers, num_layers=loader.geometry.num_layers)
    records: list[dict[str, Any]] = []
    correctness: list[dict[str, Any]] = []
    artifact_hashes: dict[str, str] = {}
    layer_memory: dict[str, dict[str, int]] = {}

    print(
        json.dumps(
            {
                "event": "simulation_start",
                "device": torch.cuda.get_device_name(device),
                "layers": list(layers),
                "batches": list(batches),
                "dtype": str(dtype),
            }
        ),
        flush=True,
    )

    for layer in layers:
        layer_started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats(device)
        print(
            json.dumps({"event": "layer_start", "layer": layer}),
            flush=True,
        )
        packed = loader.load_virtual_tp_layer(layer, device=device, dtype=dtype)
        artifact_hashes[str(layer)] = packed[0].artifact_sha256
        dense_decoders = tuple(_dense_local_decoder(source) for source in packed)
        padded_decoder = _padded_decoder(packed[0].global_decoder, packed[0].plan)
        for batch in batches:
            attention = tuple(
                torch.randn(
                    batch,
                    source.ownership.query_head_count,
                    loader.geometry.head_dim,
                    generator=torch.Generator(device=device).manual_seed(
                        args.seed + 1000003 * layer + 1009 * batch + source.process_rank
                    ),
                    dtype=dtype,
                    device=device,
                )
                for source in packed
            )
            compact_latent = _compact_latent(packed, attention)
            padded_latent = _padded_latent(packed, attention)
            feature_major_arena = torch.empty(
                packed[0].plan.total_width,
                batch,
                dtype=dtype,
                device=device,
            )
            _write_feature_major_arena(feature_major_arena, packed, attention)
            reference = compact_latent @ packed[0].global_decoder

            if batch == args.correctness_batch:
                candidates = {
                    "dense_projected_allreduce": _sum_outputs(
                        tuple(
                            local_attention.flatten(start_dim=1) @ dense_decoder
                            for local_attention, dense_decoder in zip(
                                attention,
                                dense_decoders,
                            )
                        )
                    ),
                    "local_c1_allreduce": _sum_outputs(
                        tuple(
                            source.decode_local_attention(local_attention)
                            for source, local_attention in zip(packed, attention)
                        )
                    ),
                    "feature_major_ragged_allgather": decode_feature_major(
                        feature_major_arena,
                        packed[0].global_decoder,
                    ),
                    "padded_allgather": padded_latent @ padded_decoder,
                }
                for path, observed in candidates.items():
                    metrics = _relative_error(observed, reference)
                    _assert_correct(
                        layer=layer,
                        path=path,
                        metrics=metrics,
                        relative_tolerance=args.relative_tolerance,
                    )
                    correctness.append({"layer": layer, "path": path, **metrics})
                del candidates, observed

            source_encode = tuple(
                _measure(
                    lambda source=source, local_attention=local_attention: (
                        source.encode_local_attention(local_attention)
                    ),
                    warmup=args.warmup,
                    iterations=args.iterations,
                    device=device,
                )
                for source, local_attention in zip(packed, attention)
            )
            source_local_c1 = tuple(
                _measure(
                    lambda source=source, local_attention=local_attention: (
                        source.decode_local_attention(local_attention)
                    ),
                    warmup=args.warmup,
                    iterations=args.iterations,
                    device=device,
                )
                for source, local_attention in zip(packed, attention)
            )
            source_dense = tuple(
                _measure(
                    lambda local_attention=local_attention, decoder=decoder: (
                        local_attention.flatten(start_dim=1) @ decoder
                    ),
                    warmup=args.warmup,
                    iterations=args.iterations,
                    device=device,
                )
                for local_attention, decoder in zip(attention, dense_decoders)
            )
            compact_decode = _measure(
                lambda: compact_latent @ packed[0].global_decoder,
                warmup=args.warmup,
                iterations=args.iterations,
                device=device,
            )
            feature_major_decode = _measure(
                lambda: decode_feature_major(
                    feature_major_arena,
                    packed[0].global_decoder,
                ),
                warmup=args.warmup,
                iterations=args.iterations,
                device=device,
            )
            padded_decode = _measure(
                lambda: padded_latent @ padded_decoder,
                warmup=args.warmup,
                iterations=args.iterations,
                device=device,
            )

            def dense_sequential() -> Tensor:
                return _sum_outputs(
                    tuple(
                        local_attention.flatten(start_dim=1) @ decoder
                        for local_attention, decoder in zip(attention, dense_decoders)
                    )
                )

            def local_c1_sequential() -> Tensor:
                return _sum_outputs(
                    tuple(
                        source.decode_local_attention(local_attention)
                        for source, local_attention in zip(packed, attention)
                    )
                )

            def compact_sequential() -> Tensor:
                return _compact_latent(packed, attention) @ packed[0].global_decoder

            def feature_major_sequential() -> Tensor:
                _write_feature_major_arena(feature_major_arena, packed, attention)
                return decode_feature_major(
                    feature_major_arena,
                    packed[0].global_decoder,
                )

            def padded_sequential() -> Tensor:
                return _padded_latent(packed, attention) @ padded_decoder

            communication = _communication_bytes(packed, batch=batch)
            max_encode_p50 = max(item["p50_ms"] for item in source_encode)
            records.extend(
                (
                    _path_record(
                        layer=layer,
                        batch=batch,
                        path="dense_projected_allreduce",
                        sequential=_measure(
                            dense_sequential,
                            warmup=args.warmup,
                            iterations=args.iterations,
                            device=device,
                        ),
                        parallel_compute_floor_ms=max(
                            item["p50_ms"] for item in source_dense
                        ),
                        components={"per_virtual_rank_dense_project": source_dense},
                        communication=communication["dense_projected_allreduce"],
                        packed=packed,
                    ),
                    _path_record(
                        layer=layer,
                        batch=batch,
                        path="local_c1_allreduce",
                        sequential=_measure(
                            local_c1_sequential,
                            warmup=args.warmup,
                            iterations=args.iterations,
                            device=device,
                        ),
                        parallel_compute_floor_ms=max(
                            item["p50_ms"] for item in source_local_c1
                        ),
                        components={"per_virtual_rank_encode_decode": source_local_c1},
                        communication=communication["local_c1_allreduce"],
                        packed=packed,
                    ),
                    _path_record(
                        layer=layer,
                        batch=batch,
                        path="compact_ragged_allgather",
                        sequential=_measure(
                            compact_sequential,
                            warmup=args.warmup,
                            iterations=args.iterations,
                            device=device,
                        ),
                        parallel_compute_floor_ms=(
                            max_encode_p50 + compact_decode["p50_ms"]
                        ),
                        components={
                            "per_virtual_rank_encode": source_encode,
                            "replicated_compact_decode": compact_decode,
                        },
                        communication=communication["compact_ragged_allgather"],
                        packed=packed,
                    ),
                    _path_record(
                        layer=layer,
                        batch=batch,
                        path="feature_major_ragged_allgather",
                        sequential=_measure(
                            feature_major_sequential,
                            warmup=args.warmup,
                            iterations=args.iterations,
                            device=device,
                        ),
                        parallel_compute_floor_ms=(
                            max_encode_p50 + feature_major_decode["p50_ms"]
                        ),
                        components={
                            "per_virtual_rank_encode": source_encode,
                            "replicated_feature_major_decode": feature_major_decode,
                            "sequential_arena_fill_included": True,
                            "parallel_floor_assumes_direct_attention_output": True,
                        },
                        communication=communication["compact_ragged_allgather"],
                        packed=packed,
                    ),
                    _path_record(
                        layer=layer,
                        batch=batch,
                        path="padded_allgather",
                        sequential=_measure(
                            padded_sequential,
                            warmup=args.warmup,
                            iterations=args.iterations,
                            device=device,
                        ),
                        parallel_compute_floor_ms=(
                            max_encode_p50 + padded_decode["p50_ms"]
                        ),
                        components={
                            "per_virtual_rank_encode": source_encode,
                            "replicated_padded_decode": padded_decode,
                        },
                        communication=communication["padded_allgather"],
                        packed=packed,
                    ),
                )
            )
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        # Clear closure cells before emptying the allocator cache. This makes
        # layer streaming observable rather than retaining the final batch's
        # tensors until the next loop iteration.
        attention = ()
        compact_latent = None
        padded_latent = None
        feature_major_arena = None
        reference = None
        packed = ()
        dense_decoders = ()
        padded_decoder = None
        torch.cuda.empty_cache()
        allocated_after_cleanup = int(torch.cuda.memory_allocated(device))
        layer_memory[str(layer)] = {
            "peak_allocated_bytes": peak_allocated,
            "allocated_after_cleanup_bytes": allocated_after_cleanup,
        }
        print(
            json.dumps(
                {
                    "event": "layer_complete",
                    "layer": layer,
                    "peak_allocated_bytes": peak_allocated,
                    "allocated_after_cleanup_bytes": allocated_after_cleanup,
                    "elapsed_seconds": time.perf_counter() - layer_started,
                }
            ),
            flush=True,
        )

    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": FORMAT,
        "status": "complete",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(sys.argv),
        "metadata": {
            "factor_dir": str(loader.factor_dir),
            "model_config": str(loader.model_config_path),
            "result_sha256": loader.result_sha256,
            "schedule_sha256": loader.schedule_sha256,
            "artifact_sha256": artifact_hashes,
            "virtual_tp_size": 8,
            "layers": list(layers),
            "layer_count": len(layers),
            "batches": list(batches),
            "dtype": str(dtype),
            "device": torch.cuda.get_device_name(device),
            "warmup": args.warmup,
            "iterations": args.iterations,
            "correctness_batch": args.correctness_batch,
            "relative_tolerance": args.relative_tolerance,
            "layer_loading": "one C1 factor artifact at a time",
            "maximum_layer_peak_allocated_bytes": max(
                row["peak_allocated_bytes"] for row in layer_memory.values()
            ),
            "simulation_scope": "eight virtual ranks executed sequentially on one GPU",
            "valid_claims": [
                "C1 TP8 semantic correctness",
                "single-GPU sequential compute latency",
                "isolated parallel-compute floor excluding communication",
                "logical collective byte accounting",
            ],
            "invalid_claims": [
                "measured TP8 collective latency",
                "measured TP8 throughput",
                "full-model tokens per second",
            ],
        },
        "correctness": correctness,
        "layer_memory": layer_memory,
        "records": records,
        "aggregate": _aggregate(records, layer_count=len(layers)),
    }
    _atomic_json(output_path, payload)
    if args.output_markdown:
        markdown_path = Path(args.output_markdown).expanduser().resolve()
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(_summary_markdown(payload), encoding="utf-8")
    print(json.dumps(payload["aggregate"], indent=2))


if __name__ == "__main__":
    main()
