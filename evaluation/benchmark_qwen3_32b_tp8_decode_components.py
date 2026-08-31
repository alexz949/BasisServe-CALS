#!/usr/bin/env python3
"""Profile the unchanged Qwen3-32B TP8 decode components on one GPU.

This benchmark supplies the pieces surrounding C1 variable-width attention:

* input RMSNorm and rank-width-aware fused Q/K/compact-V projection;
* Q/K head RMSNorm plus one-token RoPE;
* attention residual/RMSNorm, TP-local SwiGLU MLP, and residual;
* final RMSNorm, TP-sharded LM head, distributed-greedy local selection;
* a fixed-batch scheduler metadata update and host-to-device dispatch;
* real per-layer compact decoder GEMMs for one, two, three, and eight waves.

The run uses exact Qwen3-32B tensor shapes and real C1 decoder factors.  The
unchanged dense weights are synthetic because GEMM latency depends on shape,
layout, dtype, and hardware rather than their values.  Attention and TP
collectives are intentionally measured by their dedicated benchmarks.
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
from typing import Any, Callable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
from torch import Tensor  # noqa: E402
from torch.nn import functional as F  # noqa: E402

from basisserve.core.c1_e2e_latency import readiness_waves  # noqa: E402
from basisserve.core.c1_tp_decode import C1TPFactorLoader  # noqa: E402
from evaluation.benchmark_qwen3_32b_c1_tp_decode import (  # noqa: E402
    _dtype,
    _parse_layers,
)


FORMAT = "basisserve.qwen3_32b.tp8_decode_component_profile.v1"


def _quantile(values: Sequence[float], fraction: float) -> float:
    if not values:
        raise ValueError("cannot summarize empty timings")
    ordered = sorted(map(float, values))
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def _summary(values: Sequence[float]) -> dict[str, float]:
    timings = tuple(map(float, values))
    if not timings:
        raise ValueError("cannot summarize empty timings")
    return {
        "minimum_ms": min(timings),
        "p50_ms": statistics.median(timings),
        "p95_ms": _quantile(timings, 0.95),
        "maximum_ms": max(timings),
    }


def _measure_cuda(
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
        raise AssertionError("timed CUDA function returned no result")
    return _summary(tuple(start.elapsed_time(end) for start, end in zip(starts, ends)))


def _measure_host(
    function: Callable[[], Any],
    *,
    warmup: int,
    iterations: int,
) -> dict[str, float]:
    for _ in range(warmup):
        function()
    timings = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        function()
        timings.append((time.perf_counter_ns() - started) / 1e6)
    return _summary(timings)


def _rms_norm(hidden: Tensor, weight: Tensor, eps: float) -> Tensor:
    return F.rms_norm(hidden, (hidden.shape[-1],), weight, eps)


def _head_rms_norm(hidden: Tensor, weight: Tensor, eps: float) -> Tensor:
    input_dtype = hidden.dtype
    work = hidden.float()
    normalized = work * torch.rsqrt(work.square().mean(dim=-1, keepdim=True) + eps)
    return normalized.to(input_dtype) * weight


def _rotate_half(hidden: Tensor) -> Tensor:
    first, second = hidden.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _qk_norm_rope(
    query: Tensor,
    key: Tensor,
    q_weight: Tensor,
    k_weight: Tensor,
    cos: Tensor,
    sin: Tensor,
    eps: float,
) -> tuple[Tensor, Tensor]:
    query = _head_rms_norm(query, q_weight, eps)
    key = _head_rms_norm(key, k_weight, eps)
    broadcast_cos = cos.unsqueeze(1)
    broadcast_sin = sin.unsqueeze(1)
    return (
        query * broadcast_cos + _rotate_half(query) * broadcast_sin,
        key * broadcast_cos + _rotate_half(key) * broadcast_sin,
    )


class _StaticDecodeScheduler:
    """Minimal fixed-batch scheduler state used by the paper harness."""

    def __init__(
        self,
        *,
        batch: int,
        context_length: int,
        slot_stride: int,
    ) -> None:
        self.sequence_lengths = torch.empty(
            batch,
            dtype=torch.int64,
            pin_memory=True,
        ).fill_(context_length)
        self.positions = torch.empty(
            batch,
            dtype=torch.int64,
            pin_memory=True,
        )
        self.slot_mapping = torch.empty(
            batch,
            dtype=torch.int64,
            pin_memory=True,
        )
        self.slot_bases = torch.arange(batch, dtype=torch.int64) * slot_stride

    def step(self) -> tuple[Tensor, Tensor]:
        self.positions.copy_(self.sequence_lengths)
        torch.add(
            self.slot_bases,
            self.sequence_lengths,
            out=self.slot_mapping,
        )
        self.sequence_lengths.add_(1)
        return self.positions, self.slot_mapping


def _relative_error(observed: Tensor, reference: Tensor) -> dict[str, float | int]:
    difference = observed.float() - reference.float()
    return {
        "maximum_absolute_error": float(difference.abs().max()),
        "relative_l2_error": float(
            torch.linalg.vector_norm(difference)
            / torch.linalg.vector_norm(reference.float()).clamp_min(1e-30)
        ),
        "nonfinite": int(not torch.isfinite(observed).all()),
    }


def _decoder_path(
    compact_latent: Tensor,
    global_decoder: Tensor,
    source_widths: Sequence[int],
    source_groups: Sequence[Sequence[int]],
    *,
    warmup: int,
    iterations: int,
    device: torch.device,
) -> tuple[dict[str, Any], Tensor]:
    offsets = []
    cursor = 0
    for width in source_widths:
        offsets.append(cursor)
        cursor += int(width)
    wave_tensors: list[tuple[tuple[int, ...], Tensor, Tensor]] = []
    for raw_sources in source_groups:
        sources = tuple(map(int, raw_sources))
        inputs = torch.cat(
            tuple(
                compact_latent[
                    :, offsets[source] : offsets[source] + source_widths[source]
                ]
                for source in sources
            ),
            dim=1,
        ).contiguous()
        decoder = torch.cat(
            tuple(
                global_decoder[
                    offsets[source] : offsets[source] + source_widths[source]
                ]
                for source in sources
            ),
            dim=0,
        ).contiguous()
        wave_tensors.append((sources, inputs, decoder))

    output = torch.empty(
        compact_latent.shape[0],
        global_decoder.shape[1],
        dtype=compact_latent.dtype,
        device=device,
    )
    waves = []
    for sources, inputs, decoder in wave_tensors:
        timing = _measure_cuda(
            lambda inputs=inputs, decoder=decoder: torch.addmm(
                output,
                inputs,
                decoder,
                beta=0.0,
                out=output,
            ),
            warmup=warmup,
            iterations=iterations,
            device=device,
        )
        waves.append(
            {
                "sources": list(sources),
                "input_width": int(inputs.shape[1]),
                "timing": timing,
            }
        )

    def full_path() -> Tensor:
        for index, (_, inputs, decoder) in enumerate(wave_tensors):
            torch.addmm(
                output,
                inputs,
                decoder,
                beta=0.0 if index == 0 else 1.0,
                out=output,
            )
        return output

    full = _measure_cuda(
        full_path,
        warmup=warmup,
        iterations=iterations,
        device=device,
    )
    observed = full_path().clone()
    return {"full": full, "waves": waves}, observed


def _markdown(payload: Mapping[str, Any]) -> str:
    metadata = payload["metadata"]
    baseline = payload["baseline"]
    lines = [
        "# Qwen3-32B TP8 full-decode component profile",
        "",
        f"GPU: `{metadata['device']}`; batch: `{metadata['batch']}`; context: "
        f"`{metadata['context_length']}`; dtype: `{metadata['dtype']}`.",
        "",
        "| Unchanged component | p50 ms | p95 ms |",
        "|:---|---:|---:|",
    ]
    for name, timing in baseline["rank_invariant"].items():
        lines.append(f"| {name} | {timing['p50_ms']:.6g} | {timing['p95_ms']:.6g} |")
    lines.extend(
        [
            "",
            "| Source rank | Fused Q/K/compact-V p50 ms |",
            "|---:|---:|",
        ]
    )
    for rank, timing in baseline["fused_qkv_by_source_rank"].items():
        lines.append(f"| {rank} | {timing['p50_ms']:.6g} |")
    lines.extend(
        [
            "",
            "Decoder entries use the checkpoint's real compact decoder blocks. "
            "Attention, ragged AllGather, MLP AllReduce, and distributed-greedy "
            "communication are outside this single-GPU profile.",
        ]
    )
    return "\n".join(lines) + "\n"


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factor-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--expected-result-sha256")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-markdown")
    parser.add_argument("--layers", default="all")
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--scheduler-iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    if (
        min(
            args.batch,
            args.context_length,
            args.warmup,
            args.iterations,
            args.scheduler_iterations,
        )
        <= 0
    ):
        raise ValueError(
            "batch, context, warmup, and iteration counts must be positive"
        )
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("decode component profiling requires one CUDA GPU")
    torch.cuda.set_device(device)
    dtype = _dtype(args.dtype)

    model_path = Path(args.model).expanduser().resolve()
    config_path = model_path / "config.json" if model_path.is_dir() else model_path
    config = json.loads(config_path.read_text(encoding="utf-8"))
    loader = C1TPFactorLoader(
        args.factor_dir,
        model_config=config_path,
        tp_size=8,
        expected_result_sha256=args.expected_result_sha256,
    )
    layers = _parse_layers(args.layers, num_layers=loader.geometry.num_layers)
    hidden_size = int(config["hidden_size"])
    intermediate_size = int(config["intermediate_size"])
    vocab_size = int(config["vocab_size"])
    head_dim = int(config["head_dim"])
    local_query_heads = int(config["num_attention_heads"]) // 8
    local_intermediate = intermediate_size // 8
    local_vocab = vocab_size // 8
    eps = float(config["rms_norm_eps"])
    if intermediate_size % 8 or vocab_size % 8:
        raise ValueError("Qwen3-32B intermediate and vocabulary widths must divide TP8")

    print(
        json.dumps(
            {
                "event": "component_profile_start",
                "device": torch.cuda.get_device_name(device),
                "batch": args.batch,
                "context_length": args.context_length,
                "layers": list(layers),
            }
        ),
        flush=True,
    )
    started = time.perf_counter()
    generator = torch.Generator(device=device).manual_seed(args.seed)
    hidden = torch.randn(
        args.batch,
        hidden_size,
        dtype=dtype,
        device=device,
        generator=generator,
    )
    residual = torch.randn(
        hidden.shape,
        dtype=dtype,
        device=device,
        generator=generator,
    )
    norm_weight = torch.ones(hidden_size, dtype=dtype, device=device)
    normalized = _rms_norm(hidden, norm_weight, eps)

    rank_invariant: dict[str, dict[str, float]] = {}
    rank_invariant["input_rmsnorm"] = _measure_cuda(
        lambda: _rms_norm(hidden, norm_weight, eps),
        warmup=args.warmup,
        iterations=args.iterations,
        device=device,
    )

    source_ranks = sorted(
        {rank for layer in loader.schedule for rank in layer} | {head_dim}
    )
    fused_qkv_by_rank: dict[str, dict[str, float]] = {}
    for source_rank in source_ranks:
        qkv_width = local_query_heads * head_dim + head_dim + source_rank
        qkv_weight = (
            torch.randn(
                qkv_width,
                hidden_size,
                dtype=dtype,
                device=device,
                generator=generator,
            )
            / hidden_size**0.5
        )
        fused_qkv_by_rank[str(source_rank)] = _measure_cuda(
            lambda qkv_weight=qkv_weight: F.linear(normalized, qkv_weight),
            warmup=args.warmup,
            iterations=args.iterations,
            device=device,
        )
        del qkv_weight

    query = torch.randn(
        args.batch,
        local_query_heads,
        1,
        head_dim,
        dtype=dtype,
        device=device,
        generator=generator,
    )
    key = torch.randn(
        args.batch,
        1,
        1,
        head_dim,
        dtype=dtype,
        device=device,
        generator=generator,
    )
    q_norm_weight = torch.ones(head_dim, dtype=dtype, device=device)
    k_norm_weight = torch.ones(head_dim, dtype=dtype, device=device)
    angles = torch.randn(
        args.batch,
        1,
        head_dim,
        dtype=torch.float32,
        device=device,
        generator=generator,
    )
    cos = angles.cos().to(dtype)
    sin = angles.sin().to(dtype)
    rank_invariant["qk_headnorm_rope"] = _measure_cuda(
        lambda: _qk_norm_rope(
            query,
            key,
            q_norm_weight,
            k_norm_weight,
            cos,
            sin,
            eps,
        ),
        warmup=args.warmup,
        iterations=args.iterations,
        device=device,
    )

    rank_invariant["post_attention_residual_rmsnorm"] = _measure_cuda(
        lambda: _rms_norm(hidden + residual, norm_weight, eps),
        warmup=args.warmup,
        iterations=args.iterations,
        device=device,
    )
    gate_up_weight = (
        torch.randn(
            2 * local_intermediate,
            hidden_size,
            dtype=dtype,
            device=device,
            generator=generator,
        )
        / hidden_size**0.5
    )
    gate_up = F.linear(normalized, gate_up_weight)
    gate, up = gate_up.chunk(2, dim=-1)
    activated = F.silu(gate) * up
    down_weight = (
        torch.randn(
            hidden_size,
            local_intermediate,
            dtype=dtype,
            device=device,
            generator=generator,
        )
        / local_intermediate**0.5
    )
    rank_invariant["mlp_gate_up"] = _measure_cuda(
        lambda: F.linear(normalized, gate_up_weight),
        warmup=args.warmup,
        iterations=args.iterations,
        device=device,
    )
    rank_invariant["mlp_silu_multiply"] = _measure_cuda(
        lambda: F.silu(gate) * up,
        warmup=args.warmup,
        iterations=args.iterations,
        device=device,
    )
    rank_invariant["mlp_down"] = _measure_cuda(
        lambda: F.linear(activated, down_weight),
        warmup=args.warmup,
        iterations=args.iterations,
        device=device,
    )

    def mlp_full() -> Tensor:
        packed_gate_up = F.linear(normalized, gate_up_weight)
        local_gate, local_up = packed_gate_up.chunk(2, dim=-1)
        return F.linear(F.silu(local_gate) * local_up, down_weight)

    rank_invariant["mlp_full"] = _measure_cuda(
        mlp_full,
        warmup=args.warmup,
        iterations=args.iterations,
        device=device,
    )
    mlp_output = mlp_full()
    rank_invariant["mlp_residual"] = _measure_cuda(
        lambda: residual + mlp_output,
        warmup=args.warmup,
        iterations=args.iterations,
        device=device,
    )
    rank_invariant["final_rmsnorm"] = _measure_cuda(
        lambda: _rms_norm(hidden, norm_weight, eps),
        warmup=args.warmup,
        iterations=args.iterations,
        device=device,
    )

    dense_attention_output = torch.randn(
        args.batch,
        local_query_heads * head_dim,
        dtype=dtype,
        device=device,
        generator=generator,
    )
    dense_o_proj_weight = (
        torch.randn(
            hidden_size,
            local_query_heads * head_dim,
            dtype=dtype,
            device=device,
            generator=generator,
        )
        / (local_query_heads * head_dim) ** 0.5
    )
    rank_invariant["dense_o_proj_local"] = _measure_cuda(
        lambda: F.linear(dense_attention_output, dense_o_proj_weight),
        warmup=args.warmup,
        iterations=args.iterations,
        device=device,
    )

    lm_head_weight = (
        torch.randn(
            local_vocab,
            hidden_size,
            dtype=dtype,
            device=device,
            generator=generator,
        )
        / hidden_size**0.5
    )
    local_logits = F.linear(normalized, lm_head_weight)
    rank_invariant["lm_head_local"] = _measure_cuda(
        lambda: F.linear(normalized, lm_head_weight),
        warmup=args.warmup,
        iterations=args.iterations,
        device=device,
    )
    rank_invariant["greedy_local_argmax"] = _measure_cuda(
        lambda: torch.max(local_logits.float(), dim=-1),
        warmup=args.warmup,
        iterations=args.iterations,
        device=device,
    )

    scheduler = _StaticDecodeScheduler(
        batch=args.batch,
        context_length=args.context_length,
        slot_stride=int(config["max_position_embeddings"]),
    )
    scheduler_host = _measure_host(
        scheduler.step,
        warmup=args.warmup,
        iterations=args.scheduler_iterations,
    )
    positions_device = torch.empty(args.batch, dtype=torch.int64, device=device)
    slots_device = torch.empty(args.batch, dtype=torch.int64, device=device)

    def scheduler_h2d() -> tuple[Tensor, Tensor]:
        positions_device.copy_(scheduler.positions, non_blocking=True)
        slots_device.copy_(scheduler.slot_mapping, non_blocking=True)
        return positions_device, slots_device

    scheduler_dispatch = _measure_cuda(
        scheduler_h2d,
        warmup=args.warmup,
        iterations=args.iterations,
        device=device,
    )

    decoder_layers: dict[str, Any] = {}
    correctness = []
    for layer in layers:
        layer_started = time.perf_counter()
        packed = loader.load_virtual_tp_layer(layer, device=device, dtype=dtype)
        source_widths = packed[0].plan.source_widths
        compact_latent = torch.randn(
            args.batch,
            packed[0].plan.total_width,
            dtype=dtype,
            device=device,
            generator=generator,
        )
        reference = compact_latent @ packed[0].global_decoder
        paths = {}
        groupings = {
            "big": (tuple(range(8)),),
            "wave_2": readiness_waves(packed[0].source_ranks, 2),
            "wave_3": readiness_waves(packed[0].source_ranks, 3),
            "partial_8": readiness_waves(packed[0].source_ranks, 8),
        }
        for name, groups in groupings.items():
            record, observed = _decoder_path(
                compact_latent,
                packed[0].global_decoder,
                source_widths,
                groups,
                warmup=args.warmup,
                iterations=args.iterations,
                device=device,
            )
            metrics = _relative_error(observed, reference)
            if int(metrics["nonfinite"]) or float(metrics["relative_l2_error"]) > 0.03:
                raise AssertionError(
                    f"decoder {name} failed at layer {layer}: {metrics}"
                )
            paths[name] = record
            correctness.append({"layer": layer, "path": name, **metrics})
        decoder_layers[str(layer)] = {
            "source_ranks": list(packed[0].source_ranks),
            "source_widths": list(source_widths),
            "compact_total_width": packed[0].plan.total_width,
            "paths": paths,
        }
        print(
            json.dumps(
                {
                    "event": "decoder_layer_complete",
                    "layer": layer,
                    "source_ranks": list(packed[0].source_ranks),
                    "big_p50_ms": paths["big"]["full"]["p50_ms"],
                    "partial_8_p50_ms": paths["partial_8"]["full"]["p50_ms"],
                    "wave_2_p50_ms": paths["wave_2"]["full"]["p50_ms"],
                    "wave_3_p50_ms": paths["wave_3"]["full"]["p50_ms"],
                    "elapsed_seconds": time.perf_counter() - layer_started,
                }
            ),
            flush=True,
        )
        del packed, compact_latent, reference
        torch.cuda.empty_cache()

    payload = {
        "format": FORMAT,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metadata": {
            "command": shlex.join([sys.executable, *sys.argv]),
            "environment": "lowrank",
            "device": torch.cuda.get_device_name(device),
            "device_capability": list(torch.cuda.get_device_capability(device)),
            "torch_version": torch.__version__,
            "dtype": str(dtype),
            "batch": args.batch,
            "context_length": args.context_length,
            "tp_size": 8,
            "layers": list(layers),
            "model": str(model_path),
            "factor_dir": str(loader.factor_dir),
            "result_sha256": loader.result_sha256,
            "schedule_sha256": loader.schedule_sha256,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "scheduler_iterations": args.scheduler_iterations,
            "elapsed_seconds": time.perf_counter() - started,
            "scheduler_scope": "fixed batch with no admission, eviction, or preemption",
            "sampling_scope": "TP-sharded distributed greedy selection",
            "valid_claims": [
                "single-A100 exact-shape unchanged-component GPU latency",
                "real-checkpoint compact decoder GEMM latency",
                "fixed-batch scheduler metadata latency",
            ],
            "invalid_claims": [
                "measured TP8 collective latency",
                "optimized variable-width attention latency",
                "vLLM continuous-batching scheduler latency",
                "measured full-model serving throughput",
            ],
        },
        "geometry": {
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "local_intermediate_size": local_intermediate,
            "vocab_size": vocab_size,
            "local_vocab_size": local_vocab,
            "head_dim": head_dim,
            "local_query_heads": local_query_heads,
        },
        "baseline": {
            "rank_invariant": rank_invariant,
            "fused_qkv_by_source_rank": fused_qkv_by_rank,
            "scheduler_host": scheduler_host,
            "scheduler_h2d": scheduler_dispatch,
        },
        "decoder_layers": decoder_layers,
        "correctness": correctness,
    }
    output_json = Path(args.output_json).expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_json.with_suffix(output_json.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output_json)
    if args.output_markdown:
        output_markdown = Path(args.output_markdown).expanduser().resolve()
        output_markdown.parent.mkdir(parents=True, exist_ok=True)
        output_markdown.write_text(_markdown(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "event": "component_profile_complete",
                "output_json": str(output_json),
                "decoder_layers": len(decoder_layers),
                "elapsed_seconds": payload["metadata"]["elapsed_seconds"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
