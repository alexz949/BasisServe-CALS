#!/usr/bin/env python3
"""Benchmark the real Qwen3-32B C1 post-attention TP8 decode boundary.

This is intentionally a minimal serving harness, not a full transformer
benchmark.  Every path starts from rank-local dense attention output with
shape ``[rows, 8, 128]`` and reconstructs the same C1 layer output using the
checkpoint's real encoders and decoders:

* ``dense_projected_allreduce`` materializes each local ``A_s @ D_h`` map;
* ``local_c1_allreduce`` encodes, locally decodes, then AllReduces hidden size;
* ``padded_allgather`` uses native equal-width AllGather and a padded decoder;
* ``ragged_*`` communicates exact widths and applies the compact decoder.

Reported model-boundary latency is the sum of per-layer, maximum-over-rank
timings.  It excludes QK attention, KV-cache IO, MLP, sampling, and scheduler
cost, so its rows/s must not be presented as full-model tokens/s.
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
from typing import Any, Callable, Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.distributed as dist
from torch import Tensor

from basisserve.core.c1_tp_decode import (
    C1TPFactorLoader,
    PackedC1TPLayer,
    assert_distributed_loader_consensus,
)
from basisserve.kernels.ragged_allgather import (
    RaggedNcclCommunicator,
    StaticRaggedPlan,
    load_ragged_allgather_extension,
)


FORMAT = "basisserve.qwen3_32b.c1_tp_decode_boundary_benchmark.v1"
_ALGORITHMS = ("direct", "pairwise", "ring", "biring_grouped")
_WORKSPACE_MODES = ("fresh", "registered")


def _parse_csv(value: str) -> tuple[str, ...]:
    parsed = tuple(item.strip() for item in value.split(",") if item.strip())
    if not parsed:
        raise ValueError("comma-separated option cannot be empty")
    return parsed


def _parse_positive_ints(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item) for item in _parse_csv(value))
    if any(item <= 0 for item in parsed):
        raise ValueError(f"expected positive integers, got {parsed}")
    if len(set(parsed)) != len(parsed):
        raise ValueError(f"duplicate values are not allowed, got {parsed}")
    return parsed


def _parse_layers(value: str, *, num_layers: int) -> tuple[int, ...]:
    if value.strip().lower() == "all":
        return tuple(range(num_layers))
    layers = tuple(int(item) for item in _parse_csv(value))
    if len(set(layers)) != len(layers):
        raise ValueError(f"duplicate layers are not allowed, got {layers}")
    if any(not 0 <= layer < num_layers for layer in layers):
        raise ValueError(f"layers must lie in [0, {num_layers}), got {layers}")
    return layers


def _dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


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
    """Measure per-iteration critical-path latency across every TP rank."""

    dist.barrier()
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
        raise AssertionError("benchmark function did not execute")
    local = torch.tensor(
        [start.elapsed_time(end) for start, end in zip(starts, ends)],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(local, op=dist.ReduceOp.MAX)
    critical = local.cpu().tolist()
    return {
        "minimum_ms": min(critical),
        "p50_ms": statistics.median(critical),
        "p95_ms": _quantile(critical, 0.95),
        "maximum_ms": max(critical),
    }


def _dense_local_decoder(packed: PackedC1TPLayer) -> Tensor:
    heads = packed.ownership.query_head_count
    local_decoders = packed.local_decoder.reshape(
        heads,
        packed.local_rank,
        packed.hidden_size,
    )
    return torch.einsum(
        "dr,hrm->hdm",
        packed.local_encoder,
        local_decoders,
    ).reshape(
        heads * packed.local_encoder.shape[0],
        packed.hidden_size,
    ).contiguous()


def _padded_decoder(compact: Tensor, plan: StaticRaggedPlan) -> Tensor:
    maximum_width = max(plan.source_widths)
    padded = torch.zeros(
        len(plan.source_widths) * maximum_width,
        compact.shape[1],
        dtype=compact.dtype,
        device=compact.device,
    )
    for source, width in enumerate(plan.source_widths):
        compact_start = plan.offsets[source]
        padded_start = source * maximum_width
        padded[padded_start : padded_start + width].copy_(
            compact[compact_start : compact_start + width]
        )
    return padded


def _native_padded_gather(
    local_latent: Tensor,
    plan: StaticRaggedPlan,
) -> Tensor:
    world_size = dist.get_world_size()
    if world_size != len(plan.source_widths):
        raise ValueError("padded gather plan differs from distributed world")
    batch = int(local_latent.shape[0])
    maximum_width = max(plan.source_widths)
    local_padded = torch.zeros(
        batch,
        maximum_width,
        dtype=local_latent.dtype,
        device=local_latent.device,
    )
    local_padded[:, : local_latent.shape[1]].copy_(local_latent)
    rank_major = torch.empty(
        world_size * batch,
        maximum_width,
        dtype=local_latent.dtype,
        device=local_latent.device,
    )
    dist.all_gather_into_tensor(rank_major, local_padded)
    return (
        rank_major.reshape(world_size, batch, maximum_width)
        .permute(1, 0, 2)
        .contiguous()
        .reshape(batch, world_size * maximum_width)
    )


def _compact_from_padded(padded: Tensor, plan: StaticRaggedPlan) -> Tensor:
    maximum_width = max(plan.source_widths)
    by_source = padded.reshape(
        padded.shape[0],
        len(plan.source_widths),
        maximum_width,
    )
    return torch.cat(
        tuple(
            by_source[:, source, :width]
            for source, width in enumerate(plan.source_widths)
        ),
        dim=1,
    )


def _allreduce_copy(local_output: Tensor) -> Tensor:
    output = local_output.clone()
    dist.all_reduce(output)
    return output


def _error_metrics(observed: Tensor, reference: Tensor) -> dict[str, float]:
    difference = observed.float() - reference.float()
    values = torch.tensor(
        [
            float(difference.abs().max()),
            float(torch.linalg.vector_norm(difference)),
            float(torch.linalg.vector_norm(reference.float())),
            float(not torch.isfinite(observed).all()),
        ],
        dtype=torch.float64,
        device=observed.device,
    )
    dist.all_reduce(values, op=dist.ReduceOp.MAX)
    maximum_absolute, difference_l2, reference_l2, nonfinite = values.tolist()
    return {
        "maximum_absolute_error": maximum_absolute,
        "relative_l2_error": difference_l2 / max(reference_l2, 1e-30),
        "nonfinite": nonfinite,
    }


def _check_error(
    *,
    path: str,
    layer: int,
    metrics: dict[str, float],
    relative_tolerance: float,
) -> None:
    if metrics["nonfinite"] or metrics["relative_l2_error"] > relative_tolerance:
        raise AssertionError(
            f"{path} failed at layer {layer}: {metrics}, "
            f"relative_tolerance={relative_tolerance}"
        )


def _correctness(
    *,
    packed: PackedC1TPLayer,
    local_attention: Tensor,
    dense_decoder: Tensor,
    padded_decoder: Tensor,
    communicator: RaggedNcclCommunicator,
    variants: Iterable[tuple[str, str, bool]],
    relative_tolerance: float,
) -> list[dict[str, Any]]:
    local_latent = packed.encode_local_attention(local_attention)
    local_output = local_latent @ packed.local_decoder
    reference = _allreduce_copy(local_output)
    dense_output = _allreduce_copy(
        local_attention.flatten(start_dim=1) @ dense_decoder
    )
    padded_gather = _native_padded_gather(local_latent, packed.plan)
    padded_output = padded_gather @ padded_decoder
    results: list[dict[str, Any]] = []

    dense_metrics = _error_metrics(dense_output, reference)
    _check_error(
        path="dense_projected_allreduce",
        layer=packed.layer_index,
        metrics=dense_metrics,
        relative_tolerance=relative_tolerance,
    )
    results.append(
        {
            "layer": packed.layer_index,
            "path": "dense_projected_allreduce",
            **dense_metrics,
        }
    )
    padded_metrics = _error_metrics(padded_output, reference)
    _check_error(
        path="padded_allgather",
        layer=packed.layer_index,
        metrics=padded_metrics,
        relative_tolerance=relative_tolerance,
    )
    results.append(
        {
            "layer": packed.layer_index,
            "path": "padded_allgather",
            **padded_metrics,
        }
    )

    reference_compact = _compact_from_padded(padded_gather, packed.plan)
    for path, algorithm, registered in variants:
        observed_compact = communicator.all_gather(
            local_latent,
            packed.plan,
            algorithm=algorithm,
            registered=registered,
        ).clone()
        if not torch.equal(observed_compact, reference_compact):
            raise AssertionError(
                f"{path} changed gathered C1 coordinates at layer {packed.layer_index}"
            )
        observed = communicator.all_gather_decode(
            local_latent,
            packed.global_decoder,
            packed.plan,
            algorithm=algorithm,
            registered=registered,
        )
        metrics = _error_metrics(observed, reference)
        _check_error(
            path=path,
            layer=packed.layer_index,
            metrics=metrics,
            relative_tolerance=relative_tolerance,
        )
        results.append(
            {
                "layer": packed.layer_index,
                "path": path,
                **metrics,
            }
        )
    return results


def _path_record(
    *,
    layer: int,
    batch: int,
    path: str,
    packed: PackedC1TPLayer,
    full: dict[str, float],
    components: dict[str, dict[str, float]],
) -> dict[str, Any]:
    return {
        "layer": layer,
        "batch": batch,
        "path": path,
        "source_ranks": list(packed.source_ranks),
        "source_widths": list(packed.plan.source_widths),
        "compact_total_width": packed.plan.total_width,
        "padded_total_width": packed.plan.padded_total_width,
        "full": full,
        "components": components,
    }


def _aggregate(
    records: Sequence[dict[str, Any]],
    *,
    layer_count: int,
) -> list[dict[str, Any]]:
    keys = sorted({(int(row["batch"]), str(row["path"])) for row in records})
    aggregate: list[dict[str, Any]] = []
    for batch, path in keys:
        selected = [
            row
            for row in records
            if int(row["batch"]) == batch and str(row["path"]) == path
        ]
        if len(selected) != layer_count:
            raise AssertionError(
                f"{path} batch {batch} has {len(selected)} records for {layer_count} layers"
            )
        p50_ms = sum(float(row["full"]["p50_ms"]) for row in selected)
        p95_ms = sum(float(row["full"]["p95_ms"]) for row in selected)
        aggregate.append(
            {
                "batch": batch,
                "path": path,
                "sum_layer_p50_ms": p50_ms,
                "sum_layer_p95_ms": p95_ms,
                "boundary_rows_per_second_at_p50": 1000.0 * batch / p50_ms,
                "boundary_rows_per_second_at_p95": 1000.0 * batch / p95_ms,
            }
        )
    return aggregate


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
        "# Qwen3-32B C1 TP decode-boundary benchmark",
        "",
        f"TP size: `{metadata['world_size']}`; measured layers: "
        f"`{metadata['layer_count']}`; dtype: `{metadata['dtype']}`; "
        f"GPU: `{metadata['device']}`.",
        "",
        "| Rows | Path | Sum layer p50 (ms) | Sum layer p95 (ms) | "
        "Boundary rows/s (p50) | Boundary rows/s (p95) |",
        "|---:|:---|---:|---:|---:|---:|",
    ]
    for row in payload["aggregate"]:
        lines.append(
            f"| {row['batch']} | {row['path']} | "
            f"{row['sum_layer_p50_ms']:.6g} | {row['sum_layer_p95_ms']:.6g} | "
            f"{row['boundary_rows_per_second_at_p50']:.6g} | "
            f"{row['boundary_rows_per_second_at_p95']:.6g} |"
        )
    lines.extend(
        [
            "",
            "These are post-attention output-boundary rows/s, not full-model "
            "tokens/s. Dense attention, KV-cache IO, MLP, sampling, and serving "
            "scheduler costs are excluded.",
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
    parser.add_argument("--batches", default="1,4,8,16,32,64,128")
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument(
        "--algorithms",
        default="direct,pairwise,ring,biring_grouped",
    )
    parser.add_argument("--workspace-modes", default="fresh,registered")
    parser.add_argument("--correctness-batch", type=int, default=4)
    parser.add_argument("--relative-tolerance", type=float, default=0.05)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260822)
    args = parser.parse_args()

    if args.warmup <= 0 or args.iterations <= 0:
        raise ValueError("warmup and iterations must be positive")
    if args.relative_tolerance <= 0:
        raise ValueError("relative tolerance must be positive")
    batches = _parse_positive_ints(args.batches)
    if args.correctness_batch not in batches:
        raise ValueError("correctness batch must be present in --batches")
    algorithms = _parse_csv(args.algorithms)
    workspace_modes = _parse_csv(args.workspace_modes)
    if any(algorithm not in _ALGORITHMS for algorithm in algorithms):
        raise ValueError(f"algorithms must be drawn from {_ALGORITHMS}")
    if any(mode not in _WORKSPACE_MODES for mode in workspace_modes):
        raise ValueError(f"workspace modes must be drawn from {_WORKSPACE_MODES}")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    dtype = _dtype(args.dtype)
    loader = C1TPFactorLoader(
        args.factor_dir,
        model_config=args.model,
        tp_size=world_size,
        expected_result_sha256=args.expected_result_sha256,
    )
    if world_size != 8:
        raise ValueError(f"Qwen3-32B C1 harness requires TP8, got TP{world_size}")
    layers = _parse_layers(args.layers, num_layers=loader.geometry.num_layers)
    assert_distributed_loader_consensus(loader)
    load_ragged_allgather_extension()
    communicator = RaggedNcclCommunicator.from_process_group(device=device)
    variants = tuple(
        (
            f"ragged_{algorithm}_{workspace_mode}",
            algorithm,
            workspace_mode == "registered",
        )
        for algorithm in algorithms
        for workspace_mode in workspace_modes
    )
    records: list[dict[str, Any]] = []
    correctness: list[dict[str, Any]] = []
    artifact_hashes: dict[str, str] = {}
    try:
        for layer in layers:
            packed = loader.load_layer(
                layer,
                process_rank=rank,
                device=device,
                dtype=dtype,
            )
            artifact_hashes[str(layer)] = packed.artifact_sha256
            dense_decoder = _dense_local_decoder(packed)
            padded_decoder = _padded_decoder(packed.global_decoder, packed.plan)
            for batch in batches:
                generator = torch.Generator(device=device).manual_seed(
                    args.seed + 1000003 * layer + 1009 * batch + rank
                )
                local_attention = torch.randn(
                    batch,
                    packed.ownership.query_head_count,
                    loader.geometry.head_dim,
                    dtype=dtype,
                    device=device,
                    generator=generator,
                )
                local_latent = packed.encode_local_attention(local_attention)
                local_output = local_latent @ packed.local_decoder
                padded_gather = _native_padded_gather(local_latent, packed.plan)
                compact_gather = _compact_from_padded(padded_gather, packed.plan)

                if batch == args.correctness_batch:
                    correctness.extend(
                        _correctness(
                            packed=packed,
                            local_attention=local_attention,
                            dense_decoder=dense_decoder,
                            padded_decoder=padded_decoder,
                            communicator=communicator,
                            variants=variants,
                            relative_tolerance=args.relative_tolerance,
                        )
                    )

                encode_timing = _measure(
                    lambda: packed.encode_local_attention(local_attention),
                    warmup=args.warmup,
                    iterations=args.iterations,
                    device=device,
                )
                dense_project_timing = _measure(
                    lambda: local_attention.flatten(start_dim=1) @ dense_decoder,
                    warmup=args.warmup,
                    iterations=args.iterations,
                    device=device,
                )
                local_decode_timing = _measure(
                    lambda: local_latent @ packed.local_decoder,
                    warmup=args.warmup,
                    iterations=args.iterations,
                    device=device,
                )
                allreduce_timing = _measure(
                    lambda: _allreduce_copy(local_output),
                    warmup=args.warmup,
                    iterations=args.iterations,
                    device=device,
                )
                padded_gather_timing = _measure(
                    lambda: _native_padded_gather(local_latent, packed.plan),
                    warmup=args.warmup,
                    iterations=args.iterations,
                    device=device,
                )
                padded_decode_timing = _measure(
                    lambda: padded_gather @ padded_decoder,
                    warmup=args.warmup,
                    iterations=args.iterations,
                    device=device,
                )
                compact_decode_timing = _measure(
                    lambda: compact_gather @ packed.global_decoder,
                    warmup=args.warmup,
                    iterations=args.iterations,
                    device=device,
                )

                def dense_full() -> Tensor:
                    local = local_attention.flatten(start_dim=1) @ dense_decoder
                    dist.all_reduce(local)
                    return local

                def local_c1_full() -> Tensor:
                    latent = packed.encode_local_attention(local_attention)
                    local = latent @ packed.local_decoder
                    dist.all_reduce(local)
                    return local

                def padded_full() -> Tensor:
                    latent = packed.encode_local_attention(local_attention)
                    gathered = _native_padded_gather(latent, packed.plan)
                    return gathered @ padded_decoder

                records.append(
                    _path_record(
                        layer=layer,
                        batch=batch,
                        path="dense_projected_allreduce",
                        packed=packed,
                        full=_measure(
                            dense_full,
                            warmup=args.warmup,
                            iterations=args.iterations,
                            device=device,
                        ),
                        components={
                            "dense_project": dense_project_timing,
                            "allreduce_with_input_copy": allreduce_timing,
                        },
                    )
                )
                records.append(
                    _path_record(
                        layer=layer,
                        batch=batch,
                        path="local_c1_allreduce",
                        packed=packed,
                        full=_measure(
                            local_c1_full,
                            warmup=args.warmup,
                            iterations=args.iterations,
                            device=device,
                        ),
                        components={
                            "encode": encode_timing,
                            "local_decode": local_decode_timing,
                            "allreduce_with_input_copy": allreduce_timing,
                        },
                    )
                )
                records.append(
                    _path_record(
                        layer=layer,
                        batch=batch,
                        path="padded_allgather",
                        packed=packed,
                        full=_measure(
                            padded_full,
                            warmup=args.warmup,
                            iterations=args.iterations,
                            device=device,
                        ),
                        components={
                            "encode": encode_timing,
                            "gather_pack": padded_gather_timing,
                            "decode": padded_decode_timing,
                        },
                    )
                )
                for path, algorithm, registered in variants:
                    ragged_gather_timing = _measure(
                        lambda algorithm=algorithm, registered=registered: (
                            communicator.all_gather(
                                local_latent,
                                packed.plan,
                                algorithm=algorithm,
                                registered=registered,
                            )
                        ),
                        warmup=args.warmup,
                        iterations=args.iterations,
                        device=device,
                    )

                    def ragged_full(
                        algorithm: str = algorithm,
                        registered: bool = registered,
                    ) -> Tensor:
                        latent = packed.encode_local_attention(local_attention)
                        return communicator.all_gather_decode(
                            latent,
                            packed.global_decoder,
                            packed.plan,
                            algorithm=algorithm,
                            registered=registered,
                        )

                    records.append(
                        _path_record(
                            layer=layer,
                            batch=batch,
                            path=path,
                            packed=packed,
                            full=_measure(
                                ragged_full,
                                warmup=args.warmup,
                                iterations=args.iterations,
                                device=device,
                            ),
                            components={
                                "encode": encode_timing,
                                "gather_pack": ragged_gather_timing,
                                "decode": compact_decode_timing,
                            },
                        )
                    )
            del packed, dense_decoder, padded_decoder
            torch.cuda.empty_cache()
    finally:
        communicator.close()

    if rank == 0:
        selected_plans = [loader.schedule[layer] for layer in layers]
        heads_per_source = loader.geometry.query_heads_per_rank
        compact_width = sum(
            heads_per_source * sum(source_ranks)
            for source_ranks in selected_plans
        )
        padded_width = sum(
            world_size * heads_per_source * max(source_ranks)
            for source_ranks in selected_plans
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
                "distributed_identity_sha256": loader.distributed_identity_sha256,
                "artifact_sha256": artifact_hashes,
                "world_size": world_size,
                "layers": list(layers),
                "layer_count": len(layers),
                "batches": list(batches),
                "dtype": str(dtype),
                "device": torch.cuda.get_device_name(device),
                "algorithms": list(algorithms),
                "workspace_modes": list(workspace_modes),
                "warmup": args.warmup,
                "iterations": args.iterations,
                "correctness_batch": args.correctness_batch,
                "relative_tolerance": args.relative_tolerance,
                "measurement_boundary": "post-attention C1 TP output only",
                "excluded": [
                    "QK attention",
                    "KV-cache IO",
                    "MLP",
                    "sampling",
                    "serving scheduler",
                ],
            },
            "accounting": {
                "compact_width_sum_over_layers": compact_width,
                "padded_width_sum_over_layers": padded_width,
                "padding_overhead": padded_width / compact_width - 1.0,
            },
            "correctness": correctness,
            "records": records,
            "aggregate": _aggregate(records, layer_count=len(layers)),
        }
        _atomic_json(output_path, payload)
        if args.output_markdown:
            markdown_path = Path(args.output_markdown).expanduser().resolve()
            markdown_path.parent.mkdir(parents=True, exist_ok=True)
            markdown_path.write_text(_summary_markdown(payload), encoding="utf-8")
        print(json.dumps(payload["aggregate"], indent=2))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
