#!/usr/bin/env python3
"""Benchmark a 32K Qwen3-8B K-only offload decode kernel.

This is a hardware-path benchmark, not a quality evaluation.  Synthetic
tensors have Qwen3-8B GQA geometry; exact Keys are genuinely stored in pinned
CPU memory and selected pages genuinely cross PCIe for every offloaded run.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import statistics
import sys
import time
from typing import Callable

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basisserve.core.c1_k_offload import (  # noqa: E402
    PinnedCPUExactKeyPageStore,
    dense_exact_k_c1_attention,
    fetch_and_pack_exact_key_pages,
    kq_svd_gqa_page_selection,
    offloaded_kq_svd_c1_attention,
    sparse_exact_k_c1_attention,
)


FORMAT = "basisserve.qwen3_8b.c1_k_offload_benchmark.v1"


@dataclass(frozen=True)
class Timing:
    mean_ms: float
    median_ms: float
    p95_ms: float
    minimum_ms: float
    iterations: int


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def _benchmark(
    function: Callable[[], object],
    *,
    warmup: int,
    iterations: int,
) -> Timing:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        started = time.perf_counter()
        function()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - started) * 1.0e3)
    return Timing(
        mean_ms=statistics.fmean(samples),
        median_ms=statistics.median(samples),
        p95_ms=_percentile(samples, 0.95),
        minimum_ms=min(samples),
        iterations=iterations,
    )


def _dtype(name: str) -> torch.dtype:
    try:
        return {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }[name]
    except KeyError as error:
        raise ValueError(f"unsupported benchmark dtype {name!r}") from error


def _markdown(payload: dict[str, object]) -> str:
    metadata = payload["metadata"]
    assert isinstance(metadata, dict)
    storage = payload["storage"]
    assert isinstance(storage, dict)
    timings = payload["timings"]
    assert isinstance(timings, dict)
    derived = payload["derived"]
    assert isinstance(derived, dict)
    lines = [
        "# Qwen3-8B 32K K-only offload benchmark",
        "",
        "## Geometry",
        "",
        (
            f"- GPU: `{metadata['gpu']}`; layers `{metadata['layers']}`; "
            f"sequence `{metadata['sequence_length']}`; dtype `{metadata['dtype']}`."
        ),
        (
            f"- Q/K: `{metadata['query_heads']}` Query heads / "
            f"`{metadata['kv_heads']}` KV heads / D`{metadata['head_dim']}`; "
            f"routing R`{metadata['routing_rank']}`."
        ),
        (
            f"- Pages: size `{metadata['page_size']}`, "
            f"B`{metadata['exact_token_budget']}` per Query head."
        ),
        "",
        "## Decode latency",
        "",
        "| Path (all layers) | Median ms | Mean ms | P95 ms | Per-layer median ms |",
        "|---|---:|---:|---:|---:|",
    ]
    labels = {
        "resident_dense_full_scan": "GPU-resident BF16 exact-K full scan",
        "full_k_pcie_copy": "Pinned CPU -> GPU full exact-K copy",
        "full_k_onload_and_scan": "Full exact-K onload + full scan",
        "r32_routing_scan": "GPU R32 routing scan + page Top-k",
        "selected_k_fetch_and_pack": "Selected exact-K CPU gather + PCIe + pack",
        "prefetched_sparse_attention": "Prefetched sparse exact-K/C1-V attention",
        "offloaded_end_to_end": "R32 route + exact-K offload + sparse attention",
    }
    layers = int(metadata["layers"])
    for key, label in labels.items():
        row = timings[key]
        assert isinstance(row, dict)
        lines.append(
            f"| {label} | {row['median_ms']:.3f} | {row['mean_ms']:.3f} | "
            f"{row['p95_ms']:.3f} | {row['median_ms'] / layers:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Traffic and ratios",
            "",
            (
                f"- Exact K cache: `{storage['exact_k_gib']:.3f} GiB` in pinned CPU "
                f"memory; dense baseline keeps the same amount on GPU."
            ),
            (
                f"- Resident R32 sidecar: `{storage['routing_sidecar_gib']:.3f} GiB`; "
                f"resident C1-V: `{storage['c1_value_gib']:.3f} GiB`."
            ),
            (
                f"- Selected exact-K traffic: `{storage['selected_k_mib_per_token']:.3f} "
                f"MiB/token` = `{storage['selected_k_fraction']:.3%}` of a full K onload."
            ),
            (
                f"- Full-scan / R32 routing latency ratio: "
                f"`{derived['full_scan_over_routing']:.2f}x`."
            ),
            (
                f"- Offloaded end-to-end / resident full-scan latency ratio: "
                f"`{derived['offload_over_resident_full_scan']:.2f}x`."
            ),
            (
                f"- Measured full-copy effective bandwidth: "
                f"`{derived['full_copy_effective_gib_s']:.2f} GiB/s`."
            ),
            "",
            "The benchmark uses synthetic activations. It measures the real pinned-host "
            "gather and H2D path, but it is not an end-to-end model tokens/s result.",
        ]
    )
    return "\n".join(lines) + "\n"


def benchmark(args: argparse.Namespace) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("K offload benchmark requires CUDA")
    for name in (
        "layers",
        "sequence_length",
        "query_heads",
        "kv_heads",
        "head_dim",
        "routing_rank",
        "value_rank",
        "page_size",
        "exact_token_budget",
        "warmup",
        "iterations",
        "cpu_threads",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"{name} must be positive")
    if args.query_heads % args.kv_heads:
        raise ValueError("Query heads must be divisible by KV heads")
    if args.sequence_length % args.page_size:
        raise ValueError("benchmark sequence length must be page aligned")
    if args.exact_token_budget % args.page_size:
        raise ValueError("exact token budget must be page aligned")
    if args.routing_rank > args.head_dim:
        raise ValueError("routing rank must not exceed head dim")

    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    device = torch.device("cuda", args.device)
    torch.cuda.set_device(device)
    dtype = _dtype(args.dtype)
    batch = 1
    layers = args.layers
    host_shape = (
        layers,
        batch,
        args.kv_heads,
        args.sequence_length,
        args.head_dim,
    )
    print(f"[setup] allocating pinned exact K {host_shape}", flush=True)
    exact_key_cpu = torch.empty(
        host_shape, dtype=dtype, device="cpu", pin_memory=True
    )
    exact_key_cpu.normal_()
    print("[setup] copying dense exact-K baseline to GPU", flush=True)
    exact_key_gpu = exact_key_cpu.to(device=device, non_blocking=True)
    query = torch.randn(
        layers,
        batch,
        args.query_heads,
        1,
        args.head_dim,
        dtype=dtype,
        device=device,
    )
    routing_sidecar = torch.randn(
        layers,
        batch,
        args.kv_heads,
        args.sequence_length,
        args.routing_rank,
        dtype=dtype,
        device=device,
    )
    c1_value = torch.randn(
        layers,
        batch,
        args.kv_heads,
        args.sequence_length,
        args.value_rank,
        dtype=dtype,
        device=device,
    )
    query_projector = torch.randn(
        layers,
        args.kv_heads,
        args.head_dim,
        args.routing_rank,
        dtype=dtype,
        device=device,
    ) / math.sqrt(args.head_dim)
    stores = [
        PinnedCPUExactKeyPageStore(exact_key_cpu[layer], layer_idx=layer)
        for layer in range(layers)
    ]
    torch.cuda.synchronize()
    print("[setup] precomputing fixed-B routing decisions", flush=True)
    selections = [
        kq_svd_gqa_page_selection(
            query[layer],
            routing_sidecar[layer],
            query_projector[layer],
            page_size=args.page_size,
            pages_per_query_head=(
                args.exact_token_budget // args.page_size
            ),
        )
        for layer in range(layers)
    ]
    prefetched = [
        fetch_and_pack_exact_key_pages(
            stores[layer],
            selections[layer].page_mask,
            layer_idx=layer,
            page_size=args.page_size,
            head_dim=args.head_dim,
            device=device,
        )
        for layer in range(layers)
    ]
    torch.cuda.synchronize()
    selected_bytes = sum(item.requested_bytes for item in prefetched)
    selected_pages = sum(item.requested_pages for item in prefetched)
    dense_buffer = torch.empty_like(exact_key_gpu[0])

    def resident_dense_full_scan() -> Tensor:
        output = query[0, :, :, :, : args.value_rank]
        for layer in range(layers):
            output = dense_exact_k_c1_attention(
                query[layer], exact_key_gpu[layer], c1_value[layer]
            )
        return output

    def full_k_pcie_copy() -> Tensor:
        for layer in range(layers):
            dense_buffer.copy_(exact_key_cpu[layer], non_blocking=True)
        return dense_buffer

    def full_k_onload_and_scan() -> Tensor:
        output = query[0, :, :, :, : args.value_rank]
        for layer in range(layers):
            dense_buffer.copy_(exact_key_cpu[layer], non_blocking=True)
            output = dense_exact_k_c1_attention(
                query[layer], dense_buffer, c1_value[layer]
            )
        return output

    def r32_routing_scan() -> Tensor:
        selected = selections[0].page_mask
        for layer in range(layers):
            selected = kq_svd_gqa_page_selection(
                query[layer],
                routing_sidecar[layer],
                query_projector[layer],
                page_size=args.page_size,
                pages_per_query_head=(
                    args.exact_token_budget // args.page_size
                ),
            ).page_mask
        return selected

    def selected_k_fetch_and_pack() -> Tensor:
        fetched = prefetched[0].exact_key
        for layer in range(layers):
            fetched = fetch_and_pack_exact_key_pages(
                stores[layer],
                selections[layer].page_mask,
                layer_idx=layer,
                page_size=args.page_size,
                head_dim=args.head_dim,
                device=device,
            ).exact_key
        return fetched

    def prefetched_sparse_attention() -> Tensor:
        output = query[0, :, :, :, : args.value_rank]
        for layer in range(layers):
            output = sparse_exact_k_c1_attention(
                query[layer],
                c1_value[layer],
                prefetched[layer],
                page_size=args.page_size,
            )
        return output

    def offloaded_end_to_end() -> Tensor:
        output = query[0, :, :, :, : args.value_rank]
        for layer in range(layers):
            output = offloaded_kq_svd_c1_attention(
                query[layer],
                routing_sidecar[layer],
                query_projector[layer],
                c1_value[layer],
                stores[layer],
                layer_idx=layer,
                page_size=args.page_size,
                pages_per_query_head=(
                    args.exact_token_budget // args.page_size
                ),
            ).output
        return output

    functions = {
        "resident_dense_full_scan": resident_dense_full_scan,
        "full_k_pcie_copy": full_k_pcie_copy,
        "full_k_onload_and_scan": full_k_onload_and_scan,
        "r32_routing_scan": r32_routing_scan,
        "selected_k_fetch_and_pack": selected_k_fetch_and_pack,
        "prefetched_sparse_attention": prefetched_sparse_attention,
        "offloaded_end_to_end": offloaded_end_to_end,
    }
    timings: dict[str, dict[str, float | int]] = {}
    for name, function in functions.items():
        print(f"[benchmark] {name}", flush=True)
        timings[name] = asdict(
            _benchmark(
                function,
                warmup=args.warmup,
                iterations=args.iterations,
            )
        )

    element_size = torch.empty((), dtype=dtype).element_size()
    exact_key_bytes = math.prod(host_shape) * element_size
    routing_bytes = routing_sidecar.numel() * routing_sidecar.element_size()
    c1_bytes = c1_value.numel() * c1_value.element_size()
    dense_ms = float(timings["resident_dense_full_scan"]["median_ms"])
    routing_ms = float(timings["r32_routing_scan"]["median_ms"])
    offload_ms = float(timings["offloaded_end_to_end"]["median_ms"])
    copy_ms = float(timings["full_k_pcie_copy"]["median_ms"])
    payload: dict[str, object] = {
        "format": FORMAT,
        "metadata": {
            "gpu": torch.cuda.get_device_name(device),
            "cuda_capability": list(torch.cuda.get_device_capability(device)),
            "torch": torch.__version__,
            "layers": layers,
            "sequence_length": args.sequence_length,
            "query_heads": args.query_heads,
            "kv_heads": args.kv_heads,
            "head_dim": args.head_dim,
            "routing_rank": args.routing_rank,
            "value_rank": args.value_rank,
            "page_size": args.page_size,
            "exact_token_budget": args.exact_token_budget,
            "dtype": args.dtype,
            "cpu_threads": args.cpu_threads,
            "seed": args.seed,
            "synthetic": True,
        },
        "timings": timings,
        "storage": {
            "exact_k_bytes": exact_key_bytes,
            "exact_k_gib": exact_key_bytes / (1 << 30),
            "routing_sidecar_bytes": routing_bytes,
            "routing_sidecar_gib": routing_bytes / (1 << 30),
            "c1_value_bytes": c1_bytes,
            "c1_value_gib": c1_bytes / (1 << 30),
            "selected_pages_per_token": selected_pages,
            "selected_k_bytes_per_token": selected_bytes,
            "selected_k_mib_per_token": selected_bytes / (1 << 20),
            "selected_k_fraction": selected_bytes / exact_key_bytes,
        },
        "derived": {
            "full_scan_over_routing": dense_ms / routing_ms,
            "offload_over_resident_full_scan": offload_ms / dense_ms,
            "offload_speedup_vs_full_onload_scan": (
                float(timings["full_k_onload_and_scan"]["median_ms"])
                / offload_ms
            ),
            "full_copy_effective_gib_s": (
                exact_key_bytes / (1 << 30) / (copy_ms / 1.0e3)
            ),
            "selected_fetch_effective_gib_s": (
                selected_bytes
                / (1 << 30)
                / (
                    float(timings["selected_k_fetch_and_pack"]["median_ms"])
                    / 1.0e3
                )
            ),
        },
        "warnings": [
            "Synthetic tensors measure runtime geometry, not model quality.",
            "This is a standalone attention/offload kernel, not full-model tokens/s.",
            "Selected-page traffic depends on synthetic Query-head overlap.",
        ],
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--layers", type=int, default=36)
    parser.add_argument("--sequence-length", type=int, default=32768)
    parser.add_argument("--query-heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--routing-rank", type=int, default=32)
    parser.add_argument("--value-rank", type=int, default=96)
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument("--exact-token-budget", type=int, default=1024)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260830)
    args = parser.parse_args()

    payload = benchmark(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "result.json"
    summary_path = args.output_dir / "summary.md"
    result_path.write_text(json.dumps(payload, indent=2) + "\n")
    summary_path.write_text(_markdown(payload))
    print(_markdown(payload), end="", flush=True)
    print(f"[saved] {result_path}", flush=True)
    print(f"[saved] {summary_path}", flush=True)


if __name__ == "__main__":
    main()
