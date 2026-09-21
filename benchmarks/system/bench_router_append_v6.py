"""Matched SM89 microbenchmark for the production router and decode append."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

import torch
from torch.utils.cpp_extension import load

from basisserve.kernels.mapped_host_paged_attention import _load_extension


def _build_shared_router_baseline() -> object:
    source_root = Path("basisserve/kernels/csrc")
    sources = [
        source_root / "mapped_host_paged_attention.cpp",
        source_root / "mapped_host_paged_attention.cu",
        source_root / "conditional_router_page32.cu",
    ]
    digest = hashlib.sha256()
    for source in sources:
        digest.update(source.read_bytes())
    digest.update(b"shared-router-b16r16-v128")
    name = f"basisserve_shared_router_baseline_{digest.hexdigest()[:12]}"
    return load(
        name=name,
        sources=[str(source) for source in sources],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=[
            "-O3",
            "-std=c++17",
            "--use_fast_math",
            "--threads",
            "4",
            "-DBASIS_VALUE_DIM=128",
            "-DBASIS_GQA=4",
            "-DBASIS_PAGE_SIZE=32",
            "-DBASIS_BASE_RANK=16",
            "-DBASIS_RESIDUAL_RANK=16",
            "-DBASIS_DISABLE_REGISTER_ROUTER=1",
        ],
        with_cuda=True,
        verbose=os.environ.get("BASISSERVE_VERBOSE_BUILD", "0") == "1",
    )


def _timing(call, *, warmup: int, iterations: int) -> dict[str, float]:
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    wall_start = time.perf_counter()
    for _ in range(iterations):
        call()
    end.record()
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - wall_start) * 1000.0 / iterations
    return {
        "cuda_ms": begin.elapsed_time(end) / iterations,
        "wall_ms": wall_ms,
    }


def _median_timing(call, *, warmup: int, iterations: int) -> dict[str, float]:
    samples = [
        _timing(call, warmup=warmup, iterations=iterations)
        for _ in range(3)
    ]
    return {
        key: statistics.median(sample[key] for sample in samples)
        for key in ("cuda_ms", "wall_ms")
    }


def _router_call(extension, tensors, output) -> None:
    extension.conditional_router_page_lse(
        tensors["query"],
        tensors["base"],
        tensors["residual"],
        tensors["base_right"],
        tensors["base_bias"],
        tensors["residual_query"],
        tensors["rope_cos"],
        tensors["rope_sin"],
        tensors["query_code"],
        output,
        128**-0.5,
        True,
    )


def _append_call(extension, tensors, *, pointer: int, capacity: int) -> None:
    extension.conditional_router_append_decode(
        tensors["key"],
        tensors["value"],
        tensors["base_left"],
        tensors["base_right"],
        tensors["base_bias"],
        tensors["residual_encoder"],
        tensors["append_cos"],
        tensors["append_sin"],
        tensors["value_cache"],
        tensors["base_cache"],
        tensors["residual_cache"],
        tensors["append_cos"],
        tensors["append_sin"],
        tensors["append_start"],
        False,
        pointer,
        capacity,
    )


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lengths", type=int, nargs="+", default=[16384, 32768, 65536, 131072]
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument(
        "--output", type=Path, default=Path("results/system_benchmarks/router_append_v6")
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    log_path = args.output / "run.log"
    log_path.write_text("status=running\n")

    def emit(value) -> None:
        line = json.dumps(value, sort_keys=True)
        print(line, flush=True)
        with log_path.open("a") as log:
            log.write(line + "\n")

    assert torch.cuda.is_available()
    assert torch.cuda.get_device_capability() == (8, 9)
    torch.manual_seed(127)
    torch.backends.cuda.matmul.allow_tf32 = False
    optimized = _load_extension(
        value_dim=128,
        queries_per_kv=4,
        page_size=32,
        base_rank=16,
        residual_rank=16,
    )
    baseline = _build_shared_router_baseline()
    emit({"status": "compiled", "gpu": torch.cuda.get_device_name()})

    batch, kv_heads, maximum = 1, 8, max(args.lengths)
    query = torch.randn(batch, kv_heads * 4, 1, 128, device="cuda", dtype=torch.bfloat16)
    base = torch.randn(batch, kv_heads, maximum, 16, device="cuda", dtype=torch.bfloat16) * 0.2
    residual = torch.randn_like(base) * 0.2
    base_right = (
        torch.randn(kv_heads, 16, 128, device="cuda", dtype=torch.bfloat16) * 0.2
    ).contiguous()
    base_bias = (
        torch.randn(kv_heads, 128, device="cuda", dtype=torch.bfloat16) * 0.1
    ).contiguous()
    residual_query = (
        torch.randn(kv_heads * 4, 128, 16, device="cuda", dtype=torch.bfloat16)
        * 0.2
    ).contiguous()
    angles = torch.randn(maximum, 64, device="cuda")
    rope_cos = angles.cos().bfloat16()
    rope_sin = angles.sin().bfloat16()
    query_code = torch.empty(batch, kv_heads, 4, 16, device="cuda", dtype=torch.bfloat16)
    optimized.conditional_router_query_code(query, residual_query, query_code)
    exact_key = torch.randn(
        batch, kv_heads, maximum, 128, device="cuda", dtype=torch.bfloat16
    )
    value = torch.randn_like(exact_key)
    host_key = optimized.mapped_host_bf16_empty(
        batch, kv_heads, maximum, 128
    )
    optimized.append(host_key, exact_key, 0)
    host_key_pointer = int(optimized.device_pointer(host_key))
    attention_workspace = torch.empty(
        batch * kv_heads * 4, 32, 130, device="cuda", dtype=torch.float32
    )
    torch.cuda.synchronize()

    router_rows = []
    for length in args.lengths:
        pages = (length + 31) // 32
        tensors = {
            "query": query,
            "base": base[:, :, :length],
            "residual": residual[:, :, :length],
            "base_right": base_right,
            "base_bias": base_bias,
            "residual_query": residual_query,
            "rope_cos": rope_cos[:length],
            "rope_sin": rope_sin[:length],
            "query_code": query_code,
        }
        baseline_output = torch.empty(batch, kv_heads, 4, pages, device="cuda")
        optimized_output = torch.empty_like(baseline_output)
        baseline_call = lambda: _router_call(baseline, tensors, baseline_output)
        optimized_call = lambda: _router_call(optimized, tensors, optimized_output)
        baseline_call()
        optimized_call()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            optimized_output, baseline_output, rtol=0.003, atol=0.003
        )
        selected = min(62, pages)
        baseline_pages = torch.empty(batch, kv_heads, selected, device="cuda", dtype=torch.long)
        optimized_pages = torch.empty_like(baseline_pages)
        optimized.select_fixed_group_max_pages(
            baseline_output, baseline_pages, selected, 1, False
        )
        optimized.select_fixed_group_max_pages(
            optimized_output, optimized_pages, selected, 1, False
        )
        baseline_times = _median_timing(
            baseline_call, warmup=args.warmup, iterations=args.iterations
        )
        optimized_times = _median_timing(
            optimized_call, warmup=args.warmup, iterations=args.iterations
        )
        row = {
            "length": length,
            "pages": pages,
            "baseline": baseline_times,
            "optimized": optimized_times,
            "cuda_speedup": baseline_times["cuda_ms"] / optimized_times["cuda_ms"],
            "cuda_reduction_percent": 100.0
            * (1.0 - optimized_times["cuda_ms"] / baseline_times["cuda_ms"]),
            "projected_32_layer_saving_ms": 32.0
            * (baseline_times["cuda_ms"] - optimized_times["cuda_ms"]),
            "max_abs": float((optimized_output - baseline_output).abs().max()),
            "selected_pages_equal": bool(torch.equal(baseline_pages, optimized_pages)),
        }

        historical = length - 64
        historical_pages = historical // 32
        assert historical > 0 and historical % 32 == 0
        pipeline_tensors = {
            **tensors,
            "base": base[:, :, :historical],
            "residual": residual[:, :, :historical],
            "rope_cos": rope_cos[:historical],
            "rope_sin": rope_sin[:historical],
        }
        baseline_logs = torch.empty(
            batch, kv_heads, 4, historical_pages, device="cuda"
        )
        optimized_logs = torch.empty_like(baseline_logs)
        baseline_support = torch.empty(
            batch, kv_heads, 64, device="cuda", dtype=torch.long
        )
        optimized_support = torch.empty_like(baseline_support)
        baseline_selected = torch.empty(
            batch, kv_heads, 62, device="cuda", dtype=torch.long
        )
        optimized_selected = torch.empty_like(baseline_selected)
        recent_pages = torch.arange(
            historical_pages, historical_pages + 2, device="cuda"
        )
        baseline_support[..., 62:].copy_(recent_pages)
        optimized_support[..., 62:].copy_(recent_pages)
        baseline_attention_output = torch.empty(
            batch, kv_heads * 4, 1, 128, device="cuda", dtype=torch.bfloat16
        )
        optimized_attention_output = torch.empty_like(
            baseline_attention_output
        )

        def route_select(extension, logs, selected_pages, support) -> None:
            _router_call(extension, pipeline_tensors, logs)
            extension.select_fixed_group_max_pages(
                logs, selected_pages, 62, 1, False
            )
            support[..., :62].copy_(selected_pages)

        def selected_attention(extension, pointer, support, output) -> None:
            extension.attention(
                pointer,
                maximum,
                query,
                value,
                support,
                attention_workspace,
                output,
                length,
                128**-0.5,
                32,
                value,
                0,
            )

        def baseline_local_pipeline() -> None:
            route_select(
                baseline, baseline_logs, baseline_selected, baseline_support
            )
            selected_attention(
                baseline,
                int(exact_key.data_ptr()),
                baseline_support,
                baseline_attention_output,
            )

        def optimized_local_pipeline() -> None:
            route_select(
                optimized,
                optimized_logs,
                optimized_selected,
                optimized_support,
            )
            selected_attention(
                optimized,
                int(exact_key.data_ptr()),
                optimized_support,
                optimized_attention_output,
            )

        def baseline_offload_pipeline() -> None:
            route_select(
                baseline, baseline_logs, baseline_selected, baseline_support
            )
            selected_attention(
                baseline,
                host_key_pointer,
                baseline_support,
                baseline_attention_output,
            )

        def optimized_offload_pipeline() -> None:
            route_select(
                optimized,
                optimized_logs,
                optimized_selected,
                optimized_support,
            )
            selected_attention(
                optimized,
                host_key_pointer,
                optimized_support,
                optimized_attention_output,
            )

        baseline_local_pipeline()
        optimized_local_pipeline()
        torch.cuda.synchronize()
        assert torch.equal(baseline_support, optimized_support)
        torch.testing.assert_close(
            baseline_attention_output,
            optimized_attention_output,
            rtol=0,
            atol=0,
        )
        optimized_offload_pipeline()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            baseline_attention_output,
            optimized_attention_output,
            rtol=0,
            atol=0,
        )

        all_pages = torch.arange(
            length // 32, device="cuda", dtype=torch.long
        ).view(1, 1, -1).expand(batch, kv_heads, -1).contiguous()

        def dense_local() -> None:
            selected_attention(
                optimized,
                int(exact_key.data_ptr()),
                all_pages,
                optimized_attention_output,
            )

        def dense_offload() -> None:
            selected_attention(
                optimized,
                host_key_pointer,
                all_pages,
                optimized_attention_output,
            )

        dense_local_time = _median_timing(
            dense_local, warmup=args.warmup, iterations=args.iterations
        )
        dense_offload_time = _median_timing(
            dense_offload, warmup=args.warmup, iterations=args.iterations
        )
        baseline_local_time = _median_timing(
            baseline_local_pipeline,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        optimized_local_time = _median_timing(
            optimized_local_pipeline,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        baseline_offload_time = _median_timing(
            baseline_offload_pipeline,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        optimized_offload_time = _median_timing(
            optimized_offload_pipeline,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        row["pipeline"] = {
            "dense_local": dense_local_time,
            "dense_offload": dense_offload_time,
            "baseline_sparse_local": baseline_local_time,
            "optimized_sparse_local": optimized_local_time,
            "baseline_sparse_offload": baseline_offload_time,
            "optimized_sparse_offload": optimized_offload_time,
            "local_reduction_percent": 100.0
            * (
                1.0
                - optimized_local_time["cuda_ms"]
                / baseline_local_time["cuda_ms"]
            ),
            "offload_reduction_percent": 100.0
            * (
                1.0
                - optimized_offload_time["cuda_ms"]
                / baseline_offload_time["cuda_ms"]
            ),
            "optimized_sparse_over_dense_local": (
                optimized_local_time["cuda_ms"] / dense_local_time["cuda_ms"]
            ),
            "dense_over_sparse_offload_speedup": (
                dense_offload_time["cuda_ms"]
                / optimized_offload_time["cuda_ms"]
            ),
            "selected_pages": 64,
            "selected_tokens": 2048,
            "outputs_exact": True,
        }
        router_rows.append(row)
        emit({"router": row})

    append_capacity, append_start = 128, 64
    append_tensors = {
        "key": torch.randn(
            batch, 1, kv_heads, 128, device="cuda", dtype=torch.bfloat16
        ).transpose(1, 2),
        "value": torch.randn(
            batch, 1, kv_heads, 128, device="cuda", dtype=torch.bfloat16
        ).transpose(1, 2),
        "base_left": (
            torch.randn(kv_heads, 128, 16, device="cuda", dtype=torch.bfloat16) / 8
        ).contiguous(),
        "base_right": base_right,
        "base_bias": base_bias,
        "residual_encoder": (
            torch.randn(kv_heads, 128, 16, device="cuda", dtype=torch.bfloat16) / 8
        ).contiguous(),
        "append_cos": rope_cos[:1],
        "append_sin": rope_sin[:1],
        "value_cache": torch.empty(
            batch, kv_heads, append_capacity, 128, device="cuda", dtype=torch.bfloat16
        ),
        "base_cache": torch.empty(
            batch, kv_heads, append_capacity, 16, device="cuda", dtype=torch.bfloat16
        ),
        "residual_cache": torch.empty(
            batch, kv_heads, append_capacity, 16, device="cuda", dtype=torch.bfloat16
        ),
        "append_start": append_start,
    }
    baseline_host = baseline.mapped_host_bf16_empty(
        batch, kv_heads, append_capacity, 128
    )
    optimized_host = optimized.mapped_host_bf16_empty(
        batch, kv_heads, append_capacity, 128
    )
    optimized_pointer = int(optimized.device_pointer(optimized_host))

    def baseline_append() -> None:
        _append_call(baseline, append_tensors, pointer=0, capacity=0)
        baseline.append(baseline_host, append_tensors["key"], append_start)

    def optimized_append() -> None:
        _append_call(
            optimized,
            append_tensors,
            pointer=optimized_pointer,
            capacity=append_capacity,
        )

    baseline_append()
    optimized_append()
    torch.cuda.synchronize()
    torch.testing.assert_close(
        optimized_host[:, :, append_start : append_start + 1],
        baseline_host[:, :, append_start : append_start + 1],
        rtol=0,
        atol=0,
    )
    baseline_append_times = _median_timing(
        baseline_append, warmup=args.warmup, iterations=args.iterations
    )
    optimized_append_times = _median_timing(
        optimized_append, warmup=args.warmup, iterations=args.iterations
    )
    append_row = {
        "baseline": baseline_append_times,
        "optimized": optimized_append_times,
        "cuda_speedup": baseline_append_times["cuda_ms"]
        / optimized_append_times["cuda_ms"],
        "cuda_reduction_percent": 100.0
        * (1.0 - optimized_append_times["cuda_ms"] / baseline_append_times["cuda_ms"]),
        "projected_32_layer_saving_ms": 32.0
        * (baseline_append_times["cuda_ms"] - optimized_append_times["cuda_ms"]),
        "mapped_key_exact": True,
    }
    emit({"append": append_row})

    source_paths = [
        Path("basisserve/kernels/csrc/conditional_router_page32.cu"),
        Path("basisserve/kernels/csrc/mapped_host_paged_attention.cpp"),
        Path("basisserve/kernels/mapped_host_paged_attention.py"),
        Path(__file__),
    ]
    report = {
        "status": "complete",
        "environment": "basis",
        "gpu": torch.cuda.get_device_name(),
        "compute_capability": list(torch.cuda.get_device_capability()),
        "configuration": {
            "batch": batch,
            "kv_heads": kv_heads,
            "queries_per_kv": 4,
            "page_size": 32,
            "base_rank": 16,
            "residual_rank": 16,
            "value_dim": 128,
            "warmup": args.warmup,
            "iterations": args.iterations,
        },
        "router": router_rows,
        "append": append_row,
        "sources": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in source_paths
        },
        "scope": "Matched synthetic single-layer CUDA benchmark. Projected 32-layer savings are arithmetic projections, not end-to-end measurements.",
    }
    (args.output / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
    with log_path.open("a") as log:
        log.write("status=complete\n")
    emit({"status": "complete", "output": str(args.output / "benchmark.json")})


if __name__ == "__main__":
    main()
