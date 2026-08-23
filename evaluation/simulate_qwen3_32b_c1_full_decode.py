#!/usr/bin/env python3
"""Compose a full Qwen3-32B TP8 single-token decode latency trace.

The simulator joins three independently auditable inputs:

* rank-local dense/compact attention measurements at one batch/context;
* unchanged Qwen3-32B component and real C1 decoder-GEMM measurements;
* the checkpoint's exact 64-layer TP8 rank schedule.

TP communication uses an explicit effective alpha-beta model supplied on the
command line.  Results therefore remain simulations, but every GPU component
is measured and every unmeasured communication assumption is recorded.
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
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basisserve.core.c1_e2e_latency import (  # noqa: E402
    DecoderWave,
    EffectiveNetwork,
    barrier_c1_boundary,
    pipelined_c1_boundary,
    ragged_maximum_received_bytes,
    ring_allreduce_bytes_per_rank,
)
from basisserve.core.c1_tp_decode import C1TPFactorLoader  # noqa: E402
from evaluation.benchmark_qwen3_32b_c1_variable_v_decode import (  # noqa: E402
    FORMAT as ATTENTION_FORMAT,
)
from evaluation.benchmark_qwen3_32b_tp8_decode_components import (  # noqa: E402
    FORMAT as COMPONENT_FORMAT,
)


FORMAT = "basisserve.qwen3_32b.c1_full_decode_latency_simulation.v1"
PERCENTILES = ("p50_ms", "p95_ms")


def _load_json(path: str | Path) -> tuple[Path, dict[str, Any]]:
    resolved = Path(path).expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON input must be an object: {resolved}")
    return resolved, payload


def _timing(mapping: Mapping[str, Any], percentile: str) -> float:
    value = float(mapping[percentile])
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"invalid {percentile} timing: {value}")
    return value


def _attention_curves(
    payload: Mapping[str, Any],
    *,
    batch: int,
    context_length: int,
    required_source_ranks: Sequence[int],
) -> dict[str, Any]:
    if payload.get("format") != ATTENTION_FORMAT:
        raise ValueError("attention profile has an incompatible format")
    selected = [
        row
        for row in payload.get("records", [])
        if int(row["batch"]) == batch and int(row["context_length"]) == context_length
    ]
    if not selected:
        raise ValueError(
            f"attention profile has no batch={batch}, context={context_length} records"
        )

    curves: dict[str, dict[str, dict[str, float]]] = {
        percentile: {} for percentile in PERCENTILES
    }
    for percentile in PERCENTILES:
        for source_rank in required_source_ranks:
            rows = [row for row in selected if int(row["source_rank"]) == source_rank]
            if not rows:
                raise ValueError(f"attention profile has no source rank {source_rank}")
            curves[percentile][str(source_rank)] = {
                "compact_cache_append_ms": statistics.median(
                    _timing(row["timings"]["compact"]["cache_append"], percentile)
                    for row in rows
                ),
                "compact_attention_ms": statistics.median(
                    _timing(row["timings"]["compact"]["attention"], percentile)
                    for row in rows
                ),
            }
        curves[percentile]["dense"] = {
            "dense_cache_append_ms": statistics.median(
                _timing(row["timings"]["dense"]["cache_append"], percentile)
                for row in selected
            ),
            "dense_attention_ms": statistics.median(
                _timing(row["timings"]["dense"]["attention"], percentile)
                for row in selected
            ),
        }
    return {
        "sampled_layers": sorted({int(row["layer"]) for row in selected}),
        "record_count": len(selected),
        "curves": curves,
    }


def _scaled_decoder_waves(
    path: Mapping[str, Any],
    *,
    percentile: str,
) -> tuple[DecoderWave, ...]:
    raw = [
        DecoderWave(
            sources=tuple(map(int, wave["sources"])),
            duration_ms=_timing(wave["timing"], percentile),
        )
        for wave in path["waves"]
    ]
    raw_sum = sum(wave.duration_ms for wave in raw)
    full = _timing(path["full"], percentile)
    if raw_sum <= 0.0:
        if full != 0.0:
            raise ValueError("decoder wave timings are zero but full timing is nonzero")
        return tuple(raw)
    scale = full / raw_sum
    return tuple(DecoderWave(wave.sources, wave.duration_ms * scale) for wave in raw)


def _validate_inputs(
    attention: Mapping[str, Any],
    component: Mapping[str, Any],
    loader: C1TPFactorLoader,
    *,
    batch: int,
    context_length: int,
) -> None:
    if component.get("format") != COMPONENT_FORMAT:
        raise ValueError("component profile has an incompatible format")
    metadata = component.get("metadata", {})
    attention_run = attention.get("run", {})
    expected = {
        "batch": batch,
        "context_length": context_length,
        "tp_size": 8,
        "result_sha256": loader.result_sha256,
        "schedule_sha256": loader.schedule_sha256,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(
                f"component profile {key} mismatch: {metadata.get(key)!r} != {value!r}"
            )
    attention_expected = {
        "result_sha256": loader.result_sha256,
        "schedule_sha256": loader.schedule_sha256,
        "device": metadata.get("device"),
        "dtype": metadata.get("dtype"),
    }
    for key, value in attention_expected.items():
        if attention_run.get(key) != value:
            raise ValueError(
                f"attention profile {key} mismatch: "
                f"{attention_run.get(key)!r} != {value!r}"
            )
    layer_keys = {int(value) for value in component.get("decoder_layers", {})}
    required_layers = set(range(loader.geometry.num_layers))
    if layer_keys != required_layers:
        raise ValueError(
            "component profile must contain every model layer: "
            f"missing={sorted(required_layers - layer_keys)}, "
            f"extra={sorted(layer_keys - required_layers)}"
        )


def _full_step_for_percentile(
    *,
    percentile: str,
    loader: C1TPFactorLoader,
    attention: Mapping[str, Any],
    component: Mapping[str, Any],
    network: EffectiveNetwork,
    batch: int,
    element_bytes: int,
) -> dict[str, Any]:
    invariant = component["baseline"]["rank_invariant"]
    qkv_by_rank = component["baseline"]["fused_qkv_by_source_rank"]
    curves = attention["curves"][percentile]
    hidden_size = loader.geometry.hidden_size
    world_size = loader.geometry.tp_size
    query_heads_per_rank = loader.geometry.query_heads_per_rank

    input_norm_ms = _timing(invariant["input_rmsnorm"], percentile)
    qk_norm_rope_ms = _timing(invariant["qk_headnorm_rope"], percentile)
    post_attention_ms = _timing(
        invariant["post_attention_residual_rmsnorm"], percentile
    )
    mlp_ms = _timing(invariant["mlp_full"], percentile)
    mlp_residual_ms = _timing(invariant["mlp_residual"], percentile)
    dense_o_proj_ms = _timing(invariant["dense_o_proj_local"], percentile)
    allreduce_bytes = ring_allreduce_bytes_per_rank(
        batch=batch,
        width=hidden_size,
        world_size=world_size,
        element_bytes=element_bytes,
    )
    allreduce_ms = network.transfer_ms(allreduce_bytes)
    unchanged_after_attention_ms = (
        post_attention_ms + mlp_ms + allreduce_ms + mlp_residual_ms
    )

    dense_curve = curves["dense"]
    dense_source_ready_ms = (
        input_norm_ms
        + _timing(qkv_by_rank[str(loader.geometry.head_dim)], percentile)
        + qk_norm_rope_ms
        + float(dense_curve["dense_cache_append_ms"])
        + float(dense_curve["dense_attention_ms"])
    )
    dense_layer_ms = (
        dense_source_ready_ms
        + dense_o_proj_ms
        + allreduce_ms
        + unchanged_after_attention_ms
    )

    path_names = (
        "dense_tp8",
        "barrier_big",
        "barrier_wave_2",
        "barrier_wave_3",
        "barrier_partial_8",
        "comm_overlap_wave_2",
        "comm_overlap_wave_3",
        "comm_overlap_partial_8",
        "ideal_overlap_wave_2",
        "ideal_overlap_wave_3",
        "ideal_overlap_partial_8",
    )
    layer_sums = {path: 0.0 for path in path_names}
    layer_records = []
    for layer, source_ranks in enumerate(loader.schedule):
        source_widths = tuple(query_heads_per_rank * rank for rank in source_ranks)
        source_ready = []
        for source_rank in source_ranks:
            curve = curves[str(source_rank)]
            source_ready.append(
                input_norm_ms
                + _timing(qkv_by_rank[str(source_rank)], percentile)
                + qk_norm_rope_ms
                + float(curve["compact_cache_append_ms"])
                + float(curve["compact_attention_ms"])
            )

        decoder_layer = component["decoder_layers"][str(layer)]
        if tuple(map(int, decoder_layer["source_ranks"])) != tuple(source_ranks):
            raise ValueError(f"decoder profile rank schedule mismatch at layer {layer}")
        if tuple(map(int, decoder_layer["source_widths"])) != source_widths:
            raise ValueError(f"decoder profile source widths mismatch at layer {layer}")
        decoder_paths = decoder_layer["paths"]
        big_ms = _timing(decoder_paths["big"]["full"], percentile)
        barrier_big = barrier_c1_boundary(
            source_ready,
            source_widths,
            batch=batch,
            element_bytes=element_bytes,
            decoder_ms=big_ms,
            network=network,
        )
        c1_boundaries: dict[str, float] = {"barrier_big": barrier_big.completion_ms}
        receiver_finishes: dict[str, list[float]] = {}
        for decoder_name in ("wave_2", "wave_3", "partial_8"):
            suffix = decoder_name
            decoder_full_ms = _timing(decoder_paths[decoder_name]["full"], percentile)
            c1_boundaries[f"barrier_{suffix}"] = (
                max(source_ready)
                + network.transfer_ms(barrier_big.maximum_received_bytes)
                + decoder_full_ms
            )
            waves = _scaled_decoder_waves(
                decoder_paths[decoder_name],
                percentile=percentile,
            )
            communication_overlap = pipelined_c1_boundary(
                source_ready,
                source_widths,
                waves,
                batch=batch,
                element_bytes=element_bytes,
                network=network,
                overlap_decoder_with_local_attention=False,
            )
            ideal_overlap = pipelined_c1_boundary(
                source_ready,
                source_widths,
                waves,
                batch=batch,
                element_bytes=element_bytes,
                network=network,
                overlap_decoder_with_local_attention=True,
            )
            c1_boundaries[f"comm_overlap_{suffix}"] = (
                communication_overlap.completion_ms
            )
            c1_boundaries[f"ideal_overlap_{suffix}"] = ideal_overlap.completion_ms
            receiver_finishes[f"comm_overlap_{suffix}"] = list(
                communication_overlap.receiver_completion_ms
            )
            receiver_finishes[f"ideal_overlap_{suffix}"] = list(
                ideal_overlap.receiver_completion_ms
            )

        layer_sums["dense_tp8"] += dense_layer_ms
        layer_paths: dict[str, float] = {"dense_tp8": dense_layer_ms}
        for path, boundary_ms in c1_boundaries.items():
            layer_ms = boundary_ms + unchanged_after_attention_ms
            layer_paths[path] = layer_ms
            layer_sums[path] += layer_ms
        layer_records.append(
            {
                "layer": layer,
                "source_ranks": list(source_ranks),
                "source_widths": list(source_widths),
                "source_ready_ms": source_ready,
                "maximum_source_ready_ms": max(source_ready),
                "ragged_maximum_received_bytes": ragged_maximum_received_bytes(
                    source_widths,
                    batch=batch,
                    element_bytes=element_bytes,
                ),
                "mlp_allreduce_bytes_per_rank": allreduce_bytes,
                "mlp_allreduce_ms": allreduce_ms,
                "unchanged_after_attention_ms": unchanged_after_attention_ms,
                "paths_ms": layer_paths,
                "pipeline_receiver_completion_ms": receiver_finishes,
            }
        )

    scheduler_host_ms = _timing(component["baseline"]["scheduler_host"], percentile)
    scheduler_h2d_ms = _timing(component["baseline"]["scheduler_h2d"], percentile)
    final_norm_ms = _timing(invariant["final_rmsnorm"], percentile)
    lm_head_ms = _timing(invariant["lm_head_local"], percentile)
    local_argmax_ms = _timing(invariant["greedy_local_argmax"], percentile)
    sampling_received_bytes = (world_size - 1) * batch * 8
    sampling_collective_ms = network.transfer_ms(sampling_received_bytes)
    model_tail_ms = (
        scheduler_host_ms
        + scheduler_h2d_ms
        + final_norm_ms
        + lm_head_ms
        + local_argmax_ms
        + sampling_collective_ms
    )
    aggregates = []
    dense_total_ms = layer_sums["dense_tp8"] + model_tail_ms
    for path in path_names:
        total_ms = layer_sums[path] + model_tail_ms
        aggregates.append(
            {
                "path": path,
                "sum_layer_ms": layer_sums[path],
                "model_tail_ms": model_tail_ms,
                "decode_step_ms": total_ms,
                "simulated_tokens_per_second": 1000.0 * batch / total_ms,
                "speedup_vs_dense_tp8": dense_total_ms / total_ms,
            }
        )
    return {
        "percentile": percentile,
        "dense_source_ready_ms": dense_source_ready_ms,
        "dense_layer_ms": dense_layer_ms,
        "allreduce_bytes_per_rank": allreduce_bytes,
        "allreduce_ms": allreduce_ms,
        "model_tail": {
            "scheduler_host_ms": scheduler_host_ms,
            "scheduler_h2d_ms": scheduler_h2d_ms,
            "final_rmsnorm_ms": final_norm_ms,
            "lm_head_local_ms": lm_head_ms,
            "greedy_local_argmax_ms": local_argmax_ms,
            "sampling_collective_received_bytes": sampling_received_bytes,
            "sampling_collective_ms": sampling_collective_ms,
            "total_ms": model_tail_ms,
        },
        "aggregate": aggregates,
        "layers": layer_records,
    }


def _markdown(payload: Mapping[str, Any]) -> str:
    metadata = payload["metadata"]
    by_percentile = {
        row["percentile"]: {item["path"]: item for item in row["aggregate"]}
        for row in payload["simulations"]
    }
    paths = tuple(by_percentile["p50_ms"])
    lines = [
        "# Qwen3-32B C1 full-decode latency simulation",
        "",
        f"Batch `{metadata['batch']}`, context `{metadata['context_length']}`, "
        f"TP`{metadata['tp_size']}`. Effective network: "
        f"`alpha={metadata['network']['latency_us']} us`, "
        f"`beta={metadata['network']['bandwidth_gbps']} GB/s`.",
        "",
        "| Path | p50 step ms | p95-component-sum ms | p50 simulated tok/s | Speedup vs dense |",
        "|:---|---:|---:|---:|---:|",
    ]
    for path in paths:
        p50 = by_percentile["p50_ms"][path]
        p95 = by_percentile["p95_ms"][path]
        lines.append(
            f"| {path} | {p50['decode_step_ms']:.6g} | "
            f"{p95['decode_step_ms']:.6g} | "
            f"{p50['simulated_tokens_per_second']:.6g} | "
            f"{p50['speedup_vs_dense_tp8']:.4g}x |"
        )
    lines.extend(
        [
            "",
            "GPU operator terms are measured; TP communication is alpha-beta "
            "modeled. P95 is a sum of component p95 values, not an empirical "
            "end-to-end latency quantile. Pipeline paths assume independent "
            "source progress; `ideal_overlap` additionally allows decoder GEMMs "
            "to overlap the receiver's local attention.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factor-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--expected-result-sha256")
    parser.add_argument("--attention-profile", required=True)
    parser.add_argument("--component-profile", required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument("--network-latency-us", type=float, required=True)
    parser.add_argument("--network-bandwidth-gbps", type=float, required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-markdown")
    args = parser.parse_args()

    if args.batch <= 0 or args.context_length <= 0:
        raise ValueError("batch and context length must be positive")
    loader = C1TPFactorLoader(
        args.factor_dir,
        model_config=args.model,
        tp_size=8,
        expected_result_sha256=args.expected_result_sha256,
    )
    attention_path, attention_payload = _load_json(args.attention_profile)
    component_path, component_payload = _load_json(args.component_profile)
    _validate_inputs(
        attention_payload,
        component_payload,
        loader,
        batch=args.batch,
        context_length=args.context_length,
    )
    required_source_ranks = sorted(
        {rank for layer in loader.schedule for rank in layer}
    )
    attention = _attention_curves(
        attention_payload,
        batch=args.batch,
        context_length=args.context_length,
        required_source_ranks=required_source_ranks,
    )
    dtype = str(component_payload["metadata"]["dtype"])
    if dtype not in ("torch.float16", "torch.bfloat16"):
        raise ValueError(f"unsupported component profile dtype: {dtype}")
    element_bytes = 2
    network = EffectiveNetwork(
        latency_us=args.network_latency_us,
        bandwidth_gbps=args.network_bandwidth_gbps,
    )
    simulations = [
        _full_step_for_percentile(
            percentile=percentile,
            loader=loader,
            attention=attention,
            component=component_payload,
            network=network,
            batch=args.batch,
            element_bytes=element_bytes,
        )
        for percentile in PERCENTILES
    ]
    payload = {
        "format": FORMAT,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metadata": {
            "command": shlex.join([sys.executable, *sys.argv]),
            "environment": "lowrank",
            "batch": args.batch,
            "context_length": args.context_length,
            "tp_size": 8,
            "dtype": dtype,
            "model": str(Path(args.model).expanduser().resolve()),
            "factor_dir": str(loader.factor_dir),
            "result_sha256": loader.result_sha256,
            "schedule_sha256": loader.schedule_sha256,
            "attention_profile": str(attention_path),
            "component_profile": str(component_path),
            "attention_sampled_layers": attention["sampled_layers"],
            "attention_record_count": attention["record_count"],
            "network": {
                "model": "effective alpha-beta for all TP collectives",
                "latency_us": network.latency_us,
                "bandwidth_gbps": network.bandwidth_gbps,
            },
            "scheduler": "measured fixed-batch metadata update; no admission or preemption",
            "sampling": "TP-sharded greedy local maxima plus modeled compact gather",
            "valid_claims": [
                "trace-composed fixed-batch single-token decode latency",
                "measured single-A100 GPU operator terms",
                "explicit communication-model sensitivity",
                "barrier and decoder-wave overlap bounds",
            ],
            "invalid_claims": [
                "measured TP8 end-to-end latency",
                "measured TP8 tokens per second",
                "continuous-batching serving throughput",
                "optimized custom variable-width attention performance",
            ],
        },
        "attention_curves": attention["curves"],
        "simulations": simulations,
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
    print(json.dumps([row["aggregate"] for row in simulations], indent=2))


if __name__ == "__main__":
    main()
