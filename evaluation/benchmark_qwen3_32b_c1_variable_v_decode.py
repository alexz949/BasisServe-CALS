#!/usr/bin/env python3
"""Profile the rank-local Qwen3-32B C1 variable-width V-cache decode path.

This is a single-GPU microbenchmark, not a full-model or measured TP8 run.
For each selected layer and virtual TP rank it streams the real dense
``v_proj`` slice and the real C1 encoder ``A_s``.  Identical Q/K/hidden inputs
then exercise two one-token decode paths:

* dense V projection/cache/attention followed by headwise C1 encoding;
* folded compact V projection, ``r_s``-wide cache, and compact-V attention.

The output of both paths is ``[batch, 1, owned_query_heads * r_s]``, the local
payload consumed by ``C1RaggedOutputTPDecode``.  Collective and decoder time
are deliberately excluded so cache/attention kernel requirements remain
visible.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shlex
import statistics
import sys
import time
from typing import Any, Callable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from safetensors import safe_open  # noqa: E402
import torch  # noqa: E402
from torch import Tensor, nn  # noqa: E402

from basisserve.core.c1_tp_decode import (  # noqa: E402
    C1TPFactorLoader,
    headwise_c1_encode,
)
from basisserve.core.c1_variable_v_attention import (  # noqa: E402
    C1RankLocalAttentionReference,
    C1StaticKVCache,
    reference_grouped_query_attention,
)
from evaluation.benchmark_qwen3_32b_c1_tp_decode import (  # noqa: E402
    _dtype,
    _parse_layers,
    _parse_positive_ints,
)


FORMAT = "basisserve.qwen3_32b.c1_variable_v_decode_benchmark.v1"


def _parse_ranks(value: str, *, tp_size: int) -> tuple[int, ...]:
    if value.strip().lower() == "all":
        return tuple(range(tp_size))
    ranks = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not ranks:
        raise ValueError("rank list cannot be empty")
    if len(set(ranks)) != len(ranks):
        raise ValueError(f"duplicate ranks are not allowed, got {ranks}")
    if any(not 0 <= rank < tp_size for rank in ranks):
        raise ValueError(f"ranks must lie in [0, {tp_size}), got {ranks}")
    return ranks


def _quantile(values: Sequence[float], fraction: float) -> float:
    if not values:
        raise ValueError("cannot summarize empty timings")
    ordered = sorted(map(float, values))
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def _measure(
    function: Callable[[], Any],
    *,
    warmup: int,
    iterations: int,
    device: torch.device,
) -> dict[str, float]:
    result: Any = None
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
        raise AssertionError("timed function did not return a result")
    timings = [float(start.elapsed_time(end)) for start, end in zip(starts, ends)]
    return {
        "minimum_ms": min(timings),
        "p50_ms": statistics.median(timings),
        "p95_ms": _quantile(timings, 0.95),
        "maximum_ms": max(timings),
    }


def _error_metrics(observed: Tensor, reference: Tensor) -> dict[str, float | int]:
    difference = observed.float() - reference.float()
    return {
        "maximum_absolute_error": float(difference.abs().max()),
        "relative_l2_error": float(
            torch.linalg.vector_norm(difference)
            / torch.linalg.vector_norm(reference.float()).clamp_min(1e-30)
        ),
        "nonfinite": int(not torch.isfinite(observed).all()),
    }


def _validate_error(
    metrics: Mapping[str, float | int],
    *,
    label: str,
    layer: int,
    process_rank: int,
    tolerance: float,
) -> None:
    if int(metrics["nonfinite"]) or float(metrics["relative_l2_error"]) > tolerance:
        raise AssertionError(
            f"{label} failed at layer {layer}, rank {process_rank}: "
            f"{dict(metrics)}, tolerance={tolerance}"
        )


def _load_weight_map(model_path: Path) -> dict[str, str]:
    index_path = model_path / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(index_path)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    raw_map = index.get("weight_map")
    if not isinstance(raw_map, dict) or not raw_map:
        raise ValueError("model safetensors index has no weight_map")
    weight_map = {str(name): str(shard) for name, shard in raw_map.items()}
    if any(Path(shard).name != shard for shard in weight_map.values()):
        raise ValueError("model weight_map contains a non-local shard path")
    return weight_map


def _load_indexed_tensor(
    model_path: Path,
    weight_map: Mapping[str, str],
    tensor_name: str,
) -> tuple[Tensor, str]:
    shard_name = weight_map.get(tensor_name)
    if shard_name is None:
        raise KeyError(f"model index has no tensor {tensor_name}")
    shard_path = model_path / shard_name
    if not shard_path.is_file():
        raise FileNotFoundError(shard_path)
    with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
        if tensor_name not in handle.keys():
            raise KeyError(f"{tensor_name} is absent from indexed shard {shard_name}")
        tensor = handle.get_tensor(tensor_name)
    return tensor, shard_name


def _load_layer_v_projection(
    model_path: Path,
    weight_map: Mapping[str, str],
    *,
    layer: int,
    num_kv_heads: int,
    head_dim: int,
    hidden_size: int,
) -> tuple[Tensor, Tensor | None, dict[str, str | None]]:
    prefix = f"model.layers.{layer}.self_attn.v_proj"
    weight_name = f"{prefix}.weight"
    weight, weight_shard = _load_indexed_tensor(model_path, weight_map, weight_name)
    expected_weight = (num_kv_heads * head_dim, hidden_size)
    if tuple(weight.shape) != expected_weight or not weight.is_floating_point():
        raise ValueError(
            f"{weight_name} must have shape {expected_weight}, got {tuple(weight.shape)}"
        )
    bias_name = f"{prefix}.bias"
    bias: Tensor | None = None
    bias_shard: str | None = None
    if bias_name in weight_map:
        bias, bias_shard = _load_indexed_tensor(model_path, weight_map, bias_name)
        expected_bias = (num_kv_heads * head_dim,)
        if tuple(bias.shape) != expected_bias or not bias.is_floating_point():
            raise ValueError(
                f"{bias_name} must have shape {expected_bias}, got {tuple(bias.shape)}"
            )
    return weight, bias, {
        "weight_name": weight_name,
        "weight_shard": weight_shard,
        "bias_name": bias_name if bias is not None else None,
        "bias_shard": bias_shard,
    }


def _make_dense_projection(
    weight: Tensor,
    bias: Tensor | None,
    *,
    hidden_size: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> nn.Linear:
    projection = nn.Linear(
        hidden_size,
        head_dim,
        bias=bias is not None,
        dtype=dtype,
        device=device,
    )
    projection.requires_grad_(False)
    projection.weight.copy_(weight.to(device=device, dtype=dtype))
    if projection.bias is not None and bias is not None:
        projection.bias.copy_(bias.to(device=device, dtype=dtype))
    return projection.eval()


def _critical_path_aggregate(
    records: Sequence[Mapping[str, Any]],
    *,
    layers: Sequence[int],
    ranks: Sequence[int],
) -> list[dict[str, float | int]]:
    configurations = sorted(
        {(int(row["batch"]), int(row["context_length"])) for row in records}
    )
    output: list[dict[str, float | int]] = []
    for batch, context_length in configurations:
        selected = [
            row
            for row in records
            if int(row["batch"]) == batch
            and int(row["context_length"]) == context_length
        ]
        if len(selected) != len(layers) * len(ranks):
            raise AssertionError(
                f"incomplete records for batch={batch}, context={context_length}"
            )
        dense_critical_ms = 0.0
        compact_critical_ms = 0.0
        for layer in layers:
            layer_rows = [row for row in selected if int(row["layer"]) == layer]
            dense_critical_ms += max(
                float(row["timings"]["dense"]["end_to_end"]["p50_ms"])
                for row in layer_rows
            )
            compact_critical_ms += max(
                float(row["timings"]["compact"]["end_to_end"]["p50_ms"])
                for row in layer_rows
            )
        per_rank_speedups = [
            float(row["speedups"]["end_to_end_dense_over_compact"])
            for row in selected
        ]
        output.append(
            {
                "batch": batch,
                "context_length": context_length,
                "record_count": len(selected),
                "sampled_layer_count": len(layers),
                "selected_rank_count": len(ranks),
                "dense_critical_path_p50_ms": dense_critical_ms,
                "compact_critical_path_p50_ms": compact_critical_ms,
                "critical_path_speedup": dense_critical_ms / compact_critical_ms,
                "median_per_rank_speedup": statistics.median(per_rank_speedups),
                "minimum_per_rank_speedup": min(per_rank_speedups),
                "maximum_per_rank_speedup": max(per_rank_speedups),
                "mean_value_cache_reduction_fraction": statistics.fmean(
                    float(row["memory"]["value_cache_reduction_fraction"])
                    for row in selected
                ),
            }
        )
    return output


def _width_scaling_summary(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, float | int]]:
    """Summarize rank-width scaling at the largest measured workload."""

    maximum_batch = max(int(row["batch"]) for row in records)
    maximum_context = max(int(row["context_length"]) for row in records)
    output: list[dict[str, float | int]] = []
    for source_rank in sorted({int(row["source_rank"]) for row in records}):
        selected = [
            row
            for row in records
            if int(row["source_rank"]) == source_rank
            and int(row["batch"]) == maximum_batch
            and int(row["context_length"]) == maximum_context
        ]
        output.append(
            {
                "source_rank": source_rank,
                "batch": maximum_batch,
                "context_length": maximum_context,
                "record_count": len(selected),
                "value_cache_reduction_fraction": float(
                    selected[0]["memory"]["value_cache_reduction_fraction"]
                ),
                "median_attention_speedup": statistics.median(
                    float(row["speedups"]["attention_dense_over_compact"])
                    for row in selected
                ),
                "median_end_to_end_speedup": statistics.median(
                    float(row["speedups"]["end_to_end_dense_over_compact"])
                    for row in selected
                ),
            }
        )
    return output


def _layer_straggler_summary(
    records: Sequence[Mapping[str, Any]],
    *,
    layers: Sequence[int],
) -> list[dict[str, float | int]]:
    """Expose maximum-rank TP critical paths at the largest workload."""

    maximum_batch = max(int(row["batch"]) for row in records)
    maximum_context = max(int(row["context_length"]) for row in records)
    output: list[dict[str, float | int]] = []
    for layer in layers:
        selected = [
            row
            for row in records
            if int(row["layer"]) == layer
            and int(row["batch"]) == maximum_batch
            and int(row["context_length"]) == maximum_context
        ]
        source_ranks = [int(row["source_rank"]) for row in selected]
        dense_ms = max(
            float(row["timings"]["dense"]["end_to_end"]["p50_ms"])
            for row in selected
        )
        compact_ms = max(
            float(row["timings"]["compact"]["end_to_end"]["p50_ms"])
            for row in selected
        )
        output.append(
            {
                "layer": int(layer),
                "batch": maximum_batch,
                "context_length": maximum_context,
                "mean_source_rank": statistics.fmean(source_ranks),
                "maximum_source_rank": max(source_ranks),
                "dense_critical_path_p50_ms": dense_ms,
                "compact_critical_path_p50_ms": compact_ms,
                "critical_path_speedup": dense_ms / compact_ms,
            }
        )
    return output


def _markdown(report: Mapping[str, Any]) -> str:
    run = report["run"]
    correctness = report["correctness_summary"]
    lines = [
        "# Qwen3-32B C1 variable-width V-cache decode microbenchmark",
        "",
        "This is a single-GPU rank-local PyTorch reference benchmark. It does not "
        "measure NCCL, the global decoder, full transformer layers, or serving tokens/s.",
        "",
        f"- Device: `{run['device']}`",
        f"- Dtype: `{run['dtype']}`",
        f"- Layers: `{run['layers']}`",
        f"- Virtual TP ranks: `{run['ranks']}`",
        f"- Warmup / iterations: `{run['warmup']} / {run['iterations']}`",
        f"- Maximum projection relative L2 error: "
        f"`{correctness['maximum_projection_relative_l2_error']:.6g}`",
        f"- Maximum attention relative L2 error: "
        f"`{correctness['maximum_attention_relative_l2_error']:.6g}`",
        "",
        "## Sampled-layer critical path",
        "",
        "For each sampled layer, the table takes the slowest selected virtual rank, "
        "then sums those layer latencies. Communication is excluded.",
        "",
        "| Batch | Context | Dense p50 ms | Compact p50 ms | Speedup | "
        "Median rank speedup | Mean V-cache reduction |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["critical_path"]:
        lines.append(
            f"| {row['batch']} | {row['context_length']} | "
            f"{row['dense_critical_path_p50_ms']:.6g} | "
            f"{row['compact_critical_path_p50_ms']:.6g} | "
            f"{row['critical_path_speedup']:.4g}x | "
            f"{row['median_per_rank_speedup']:.4g}x | "
            f"{100.0 * row['mean_value_cache_reduction_fraction']:.3f}% |"
        )
    width_scaling = report["width_scaling"]
    lines.extend(
        (
            "",
            "## Width scaling at the largest workload",
            "",
            f"Batch `{width_scaling[0]['batch']}`, context "
            f"`{width_scaling[0]['context_length']}`; medians are across every "
            "occurrence of that source rank in the sampled layers.",
            "",
            "| Source rank | V-cache reduction | Attention speedup | End-to-end speedup |",
            "|---:|---:|---:|---:|",
        )
    )
    for row in width_scaling:
        lines.append(
            f"| {row['source_rank']} | "
            f"{100.0 * row['value_cache_reduction_fraction']:.3f}% | "
            f"{row['median_attention_speedup']:.4g}x | "
            f"{row['median_end_to_end_speedup']:.4g}x |"
        )
    layer_stragglers = report["layer_stragglers"]
    lines.extend(
        (
            "",
            "## TP maximum-rank straggler at the largest workload",
            "",
            "The per-layer critical path is the slowest selected virtual rank. A low "
            "mean rank does not improve synchronized TP latency when the layer still "
            "contains an uncompressed rank-128 source.",
            "",
            "| Layer | Mean rank | Max rank | Dense p50 ms | Compact p50 ms | Speedup |",
            "|---:|---:|---:|---:|---:|---:|",
        )
    )
    for row in layer_stragglers:
        lines.append(
            f"| {row['layer']} | {row['mean_source_rank']:.3f} | "
            f"{row['maximum_source_rank']} | "
            f"{row['dense_critical_path_p50_ms']:.6g} | "
            f"{row['compact_critical_path_p50_ms']:.6g} | "
            f"{row['critical_path_speedup']:.4g}x |"
        )
    lines.extend(
        (
            "",
            "## Exact command",
            "",
            "```bash",
            str(run["command"]),
            "```",
            "",
        )
    )
    return "\n".join(lines)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factor-dir", required=True)
    parser.add_argument("--model", required=True, help="local model snapshot directory")
    parser.add_argument("--expected-result-sha256")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-markdown")
    parser.add_argument("--layers", default="0,20,48,63")
    parser.add_argument("--ranks", default="all")
    parser.add_argument("--batches", default="1,8,32")
    parser.add_argument("--contexts", default="128,2048")
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--relative-tolerance", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    if args.warmup <= 0 or args.iterations <= 0:
        raise ValueError("warmup and iterations must be positive")
    if args.relative_tolerance <= 0.0:
        raise ValueError("relative tolerance must be positive")
    model_path = Path(args.model).expanduser().resolve()
    if not model_path.is_dir() or not (model_path / "config.json").is_file():
        raise FileNotFoundError(f"invalid local model snapshot: {model_path}")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("variable-width V-cache benchmark requires one CUDA GPU")
    torch.cuda.set_device(device)
    dtype = _dtype(args.dtype)

    loader = C1TPFactorLoader(
        args.factor_dir,
        model_config=model_path,
        tp_size=8,
        expected_result_sha256=args.expected_result_sha256,
    )
    layers = _parse_layers(args.layers, num_layers=loader.geometry.num_layers)
    ranks = _parse_ranks(args.ranks, tp_size=loader.geometry.tp_size)
    batches = _parse_positive_ints(args.batches)
    contexts = _parse_positive_ints(args.contexts)
    weight_map = _load_weight_map(model_path)
    records: list[dict[str, Any]] = []
    layer_tensors: dict[str, dict[str, str | None]] = {}

    print(
        json.dumps(
            {
                "event": "benchmark_start",
                "device": torch.cuda.get_device_name(device),
                "layers": list(layers),
                "ranks": list(ranks),
                "batches": list(batches),
                "contexts": list(contexts),
                "dtype": str(dtype),
            }
        ),
        flush=True,
    )
    started = time.perf_counter()
    for layer in layers:
        layer_started = time.perf_counter()
        print(json.dumps({"event": "layer_start", "layer": layer}), flush=True)
        packed_layer = loader.load_virtual_tp_layer(layer, device=device, dtype=dtype)
        dense_v_weight, dense_v_bias, tensor_record = _load_layer_v_projection(
            model_path,
            weight_map,
            layer=layer,
            num_kv_heads=loader.geometry.num_kv_heads,
            head_dim=loader.geometry.head_dim,
            hidden_size=loader.geometry.hidden_size,
        )
        layer_tensors[str(layer)] = tensor_record
        for process_rank in ranks:
            factors = packed_layer[process_rank]
            dense_rows = slice(
                process_rank * loader.geometry.head_dim,
                (process_rank + 1) * loader.geometry.head_dim,
            )
            local_bias = None if dense_v_bias is None else dense_v_bias[dense_rows]
            dense_projection = _make_dense_projection(
                dense_v_weight[dense_rows].contiguous(),
                local_bias,
                hidden_size=loader.geometry.hidden_size,
                head_dim=loader.geometry.head_dim,
                dtype=dtype,
                device=device,
            )
            compact_attention = C1RankLocalAttentionReference.from_dense_projection(
                dense_projection,
                factors,
                output_dtype=dtype,
            ).eval()
            encoder = factors.local_encoder
            for batch in batches:
                for context_length in contexts:
                    configuration_started = time.perf_counter()
                    allocated_before = torch.cuda.memory_allocated(device)
                    torch.cuda.reset_peak_memory_stats(device)
                    generator = torch.Generator(device=device).manual_seed(
                        args.seed
                        + 1_000_003 * layer
                        + 10_007 * process_rank
                        + 101 * batch
                        + context_length
                    )
                    context_keys = torch.randn(
                        batch,
                        context_length,
                        loader.geometry.head_dim,
                        generator=generator,
                        dtype=dtype,
                        device=device,
                    )
                    dense_context_values = torch.randn(
                        batch,
                        context_length,
                        loader.geometry.head_dim,
                        generator=generator,
                        dtype=dtype,
                        device=device,
                    )
                    compact_context_values = dense_context_values @ encoder
                    dense_cache = C1StaticKVCache(
                        batch_size=batch,
                        capacity=context_length + 1,
                        head_dim=loader.geometry.head_dim,
                        value_head_dim=loader.geometry.head_dim,
                        dtype=dtype,
                        device=device,
                    )
                    compact_cache = C1StaticKVCache(
                        batch_size=batch,
                        capacity=context_length + 1,
                        head_dim=loader.geometry.head_dim,
                        value_head_dim=factors.local_rank,
                        dtype=dtype,
                        device=device,
                    )
                    dense_cache.append(context_keys, dense_context_values)
                    compact_cache.append(context_keys, compact_context_values)
                    hidden_states = torch.randn(
                        batch,
                        1,
                        loader.geometry.hidden_size,
                        generator=generator,
                        dtype=dtype,
                        device=device,
                    )
                    query_states = torch.randn(
                        batch,
                        factors.ownership.query_head_count,
                        1,
                        loader.geometry.head_dim,
                        generator=generator,
                        dtype=dtype,
                        device=device,
                    )
                    new_key = torch.randn(
                        batch,
                        1,
                        loader.geometry.head_dim,
                        generator=generator,
                        dtype=dtype,
                        device=device,
                    )
                    dense_new_value = dense_projection(hidden_states)
                    compact_new_value = compact_attention.project_values(hidden_states)
                    dense_cache.append(new_key, dense_new_value)
                    compact_cache.append(new_key, compact_new_value)
                    cached_dense_key, cached_dense_value = dense_cache.current()
                    cached_compact_key, cached_compact_value = compact_cache.current()
                    dense_head_output, dense_weights = reference_grouped_query_attention(
                        query_states,
                        cached_dense_key,
                        cached_dense_value,
                        is_causal=False,
                    )
                    compact_head_output, compact_weights = reference_grouped_query_attention(
                        query_states,
                        cached_compact_key,
                        cached_compact_value,
                        is_causal=False,
                    )
                    dense_coordinates = headwise_c1_encode(dense_head_output, encoder)
                    compact_coordinates = compact_head_output.reshape(
                        batch,
                        1,
                        factors.local_wire_width,
                    )
                    projection_error = _error_metrics(
                        compact_new_value,
                        dense_new_value @ encoder,
                    )
                    attention_error = _error_metrics(
                        compact_coordinates,
                        dense_coordinates,
                    )
                    weight_error = _error_metrics(compact_weights, dense_weights)
                    for label, metrics in (
                        ("folded_projection", projection_error),
                        ("compact_attention", attention_error),
                        ("attention_weights", weight_error),
                    ):
                        _validate_error(
                            metrics,
                            label=label,
                            layer=layer,
                            process_rank=process_rank,
                            tolerance=args.relative_tolerance,
                        )

                    def dense_cache_append() -> tuple[Tensor, Tensor]:
                        dense_cache.truncate(context_length)
                        return dense_cache.append(new_key, dense_new_value)

                    def compact_cache_append() -> tuple[Tensor, Tensor]:
                        compact_cache.truncate(context_length)
                        return compact_cache.append(new_key, compact_new_value)

                    def dense_attention_only() -> Tensor:
                        return reference_grouped_query_attention(
                            query_states,
                            cached_dense_key,
                            cached_dense_value,
                            is_causal=False,
                        )[0]

                    def compact_attention_only() -> Tensor:
                        return reference_grouped_query_attention(
                            query_states,
                            cached_compact_key,
                            cached_compact_value,
                            is_causal=False,
                        )[0]

                    def dense_end_to_end() -> Tensor:
                        dense_cache.truncate(context_length)
                        new_value = dense_projection(hidden_states)
                        keys, values = dense_cache.append(new_key, new_value)
                        head_output, _ = reference_grouped_query_attention(
                            query_states,
                            keys,
                            values,
                            is_causal=False,
                        )
                        return headwise_c1_encode(head_output, encoder)

                    def compact_end_to_end() -> Tensor:
                        compact_cache.truncate(context_length)
                        return compact_attention(
                            hidden_states,
                            query_states,
                            new_key,
                            compact_cache,
                            is_causal=False,
                        ).local_coordinates

                    dense_timings = {
                        "projection": _measure(
                            lambda: dense_projection(hidden_states),
                            warmup=args.warmup,
                            iterations=args.iterations,
                            device=device,
                        ),
                        "cache_append": _measure(
                            dense_cache_append,
                            warmup=args.warmup,
                            iterations=args.iterations,
                            device=device,
                        ),
                        "attention": _measure(
                            dense_attention_only,
                            warmup=args.warmup,
                            iterations=args.iterations,
                            device=device,
                        ),
                        "post_attention_encode": _measure(
                            lambda: headwise_c1_encode(dense_head_output, encoder),
                            warmup=args.warmup,
                            iterations=args.iterations,
                            device=device,
                        ),
                        "end_to_end": _measure(
                            dense_end_to_end,
                            warmup=args.warmup,
                            iterations=args.iterations,
                            device=device,
                        ),
                    }
                    compact_timings = {
                        "projection": _measure(
                            lambda: compact_attention.project_values(hidden_states),
                            warmup=args.warmup,
                            iterations=args.iterations,
                            device=device,
                        ),
                        "cache_append": _measure(
                            compact_cache_append,
                            warmup=args.warmup,
                            iterations=args.iterations,
                            device=device,
                        ),
                        "attention": _measure(
                            compact_attention_only,
                            warmup=args.warmup,
                            iterations=args.iterations,
                            device=device,
                        ),
                        "end_to_end": _measure(
                            compact_end_to_end,
                            warmup=args.warmup,
                            iterations=args.iterations,
                            device=device,
                        ),
                    }
                    element_size = torch.empty((), dtype=dtype).element_size()
                    dense_value_bytes = (
                        batch
                        * (context_length + 1)
                        * loader.geometry.head_dim
                        * element_size
                    )
                    compact_value_bytes = (
                        batch
                        * (context_length + 1)
                        * factors.local_rank
                        * element_size
                    )
                    record = {
                        "layer": layer,
                        "process_rank": process_rank,
                        "kv_head": factors.ownership.kv_head,
                        "query_head_range": [
                            factors.ownership.query_head_start,
                            factors.ownership.query_head_stop,
                        ],
                        "source_rank": factors.local_rank,
                        "local_wire_width": factors.local_wire_width,
                        "batch": batch,
                        "context_length": context_length,
                        "correctness": {
                            "folded_projection": projection_error,
                            "compact_attention": attention_error,
                            "attention_weights": weight_error,
                        },
                        "memory": {
                            "dense_value_cache_bytes": dense_value_bytes,
                            "compact_value_cache_bytes": compact_value_bytes,
                            "value_cache_reduction_fraction": 1.0
                            - compact_value_bytes / dense_value_bytes,
                            "dense_total_kv_cache_bytes": dense_cache.allocated_bytes,
                            "compact_total_kv_cache_bytes": compact_cache.allocated_bytes,
                            "incremental_peak_allocated_bytes": max(
                                0,
                                torch.cuda.max_memory_allocated(device)
                                - allocated_before,
                            ),
                        },
                        "timings": {
                            "dense": dense_timings,
                            "compact": compact_timings,
                        },
                        "speedups": {
                            "projection_dense_over_compact": dense_timings["projection"][
                                "p50_ms"
                            ]
                            / compact_timings["projection"]["p50_ms"],
                            "cache_append_dense_over_compact": dense_timings[
                                "cache_append"
                            ]["p50_ms"]
                            / compact_timings["cache_append"]["p50_ms"],
                            "attention_dense_over_compact": dense_timings["attention"][
                                "p50_ms"
                            ]
                            / compact_timings["attention"]["p50_ms"],
                            "end_to_end_dense_over_compact": dense_timings["end_to_end"][
                                "p50_ms"
                            ]
                            / compact_timings["end_to_end"]["p50_ms"],
                        },
                    }
                    records.append(record)
                    print(
                        json.dumps(
                            {
                                "event": "configuration_complete",
                                "layer": layer,
                                "rank": process_rank,
                                "source_rank": factors.local_rank,
                                "batch": batch,
                                "context": context_length,
                                "dense_p50_ms": dense_timings["end_to_end"]["p50_ms"],
                                "compact_p50_ms": compact_timings["end_to_end"]["p50_ms"],
                                "speedup": record["speedups"][
                                    "end_to_end_dense_over_compact"
                                ],
                                "elapsed_seconds": time.perf_counter()
                                - configuration_started,
                            }
                        ),
                        flush=True,
                    )
        torch.cuda.empty_cache()
        print(
            json.dumps(
                {
                    "event": "layer_complete",
                    "layer": layer,
                    "elapsed_seconds": time.perf_counter() - layer_started,
                }
            ),
            flush=True,
        )

    critical_path = _critical_path_aggregate(
        records,
        layers=layers,
        ranks=ranks,
    )
    width_scaling = _width_scaling_summary(records)
    layer_stragglers = _layer_straggler_summary(records, layers=layers)
    projection_errors = [
        float(row["correctness"]["folded_projection"]["relative_l2_error"])
        for row in records
    ]
    attention_errors = [
        float(row["correctness"]["compact_attention"]["relative_l2_error"])
        for row in records
    ]
    command = " ".join(shlex.quote(item) for item in [sys.executable, *sys.argv])
    report = {
        "format": FORMAT,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run": {
            "command": command,
            "environment": "lowrank",
            "device": torch.cuda.get_device_name(device),
            "device_capability": list(torch.cuda.get_device_capability(device)),
            "torch_version": torch.__version__,
            "dtype": str(dtype),
            "model": str(model_path),
            "factor_dir": str(Path(args.factor_dir).expanduser().resolve()),
            "result_sha256": loader.result_sha256,
            "schedule_sha256": loader.schedule_sha256,
            "layers": list(layers),
            "ranks": list(ranks),
            "batches": list(batches),
            "contexts": list(contexts),
            "warmup": args.warmup,
            "iterations": args.iterations,
            "relative_tolerance": args.relative_tolerance,
            "seed": args.seed,
            "elapsed_seconds": time.perf_counter() - started,
            "scope": "single-GPU sequential virtual ranks; one-token decode",
            "valid_claims": [
                "folded v_proj and compact-V attention correctness",
                "rank-local PyTorch projection/cache/attention latency",
                "variable-width V-cache allocation reduction",
                "sampled-layer compute critical path excluding communication",
            ],
            "invalid_claims": [
                "custom CUDA kernel performance",
                "measured TP8 collective latency",
                "full-model latency or tokens per second",
            ],
        },
        "model_tensors": layer_tensors,
        "correctness_summary": {
            "check_count": len(records) * 3,
            "maximum_projection_relative_l2_error": max(projection_errors),
            "maximum_attention_relative_l2_error": max(attention_errors),
            "nonfinite_count": sum(
                int(metrics["nonfinite"])
                for row in records
                for metrics in row["correctness"].values()
            ),
        },
        "critical_path": critical_path,
        "width_scaling": width_scaling,
        "layer_stragglers": layer_stragglers,
        "records": records,
    }
    output_json = Path(args.output_json).expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.output_markdown:
        output_markdown = Path(args.output_markdown).expanduser().resolve()
        output_markdown.parent.mkdir(parents=True, exist_ok=True)
        output_markdown.write_text(_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "event": "benchmark_complete",
                "records": len(records),
                "output_json": str(output_json),
                "elapsed_seconds": report["run"]["elapsed_seconds"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
