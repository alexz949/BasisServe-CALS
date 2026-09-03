#!/usr/bin/env python3
"""Validate cached-Base16 routing and the fused single-token append kernel."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shlex
import sys
import time
from typing import Callable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
from torch import Tensor  # noqa: E402

from basisserve.core.qwen3_8b_tp4_k_offload import (  # noqa: E402
    conditional_router_page_log_mass,
    select_fixed_group_max_pages,
)
from basisserve.kernels.mapped_host_paged_attention import (  # noqa: E402
    conditional_router_append_decode,
    conditional_router_page32_lse,
    prepare_mapped_host_paged_attention_extension,
    select_fixed_group_max_pages_cuda,
)


def _inputs(tokens: int, *, device: torch.device) -> dict[str, Tensor]:
    generator = torch.Generator(device=device).manual_seed(20260902 + tokens)
    query = torch.randn(
        1,
        1,
        8,
        128,
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    ).transpose(1, 2)
    positions = torch.arange(tokens, dtype=torch.float32, device=device)
    frequencies = torch.exp(
        -math.log(10000.0)
        * torch.arange(64, dtype=torch.float32, device=device)
        / 64
    )
    angles = positions[:, None] * frequencies[None]
    value = torch.randn(
        1,
        2,
        tokens,
        80,
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    )
    base_left = torch.randn(
        2,
        80,
        16,
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    ) / 8
    result = {
        "query": query,
        "value": value,
        "residual_code": torch.randn(
            1,
            2,
            tokens,
            8,
            generator=generator,
            dtype=torch.bfloat16,
            device=device,
        ),
        "base_left": base_left,
        "base_right": torch.randn(
            2,
            16,
            128,
            generator=generator,
            dtype=torch.bfloat16,
            device=device,
        )
        / 8,
        "base_bias": torch.randn(
            2,
            128,
            generator=generator,
            dtype=torch.bfloat16,
            device=device,
        )
        / 8,
        "residual_query": torch.randn(
            8,
            128,
            8,
            generator=generator,
            dtype=torch.bfloat16,
            device=device,
        )
        / 8,
        "residual_encoder": torch.randn(
            2,
            128,
            8,
            generator=generator,
            dtype=torch.bfloat16,
            device=device,
        )
        / 8,
        "rope_cos": angles.cos().to(torch.bfloat16),
        "rope_sin": angles.sin().to(torch.bfloat16),
    }
    result["base_code"] = torch.einsum(
        "bgtv,gvr->bgtr",
        value,
        base_left,
    )
    return result


def _fused(inputs: dict[str, Tensor]) -> Tensor:
    return conditional_router_page32_lse(
        inputs["query"],
        inputs["base_code"],
        inputs["residual_code"],
        base_right=inputs["base_right"],
        base_bias=inputs["base_bias"],
        residual_query=inputs["residual_query"],
        rope_cos=inputs["rope_cos"],
        rope_sin=inputs["rope_sin"],
        scale=128**-0.5,
    )


def _reference(inputs: dict[str, Tensor]) -> Tensor:
    return conditional_router_page_log_mass(
        inputs["query"],
        inputs["value"],
        inputs["residual_code"],
        base_left=inputs["base_left"],
        base_right=inputs["base_right"],
        base_bias=inputs["base_bias"],
        residual_query=inputs["residual_query"],
        rope_cos=inputs["rope_cos"],
        rope_sin=inputs["rope_sin"],
        page_size=32,
        page_chunk=64,
        scale=128**-0.5,
    )


def _apply_rope(values: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    first, second = values.chunk(2, dim=-1)
    return torch.cat(
        (first * cos - second * sin, second * cos + first * sin),
        dim=-1,
    )


def _reference_append(
    inputs: dict[str, Tensor],
    key: Tensor,
    caches: dict[str, Tensor],
) -> None:
    caches["value"][:, :, 1:2].copy_(inputs["value"])
    caches["cos"][1:2].copy_(inputs["rope_cos"])
    caches["sin"][1:2].copy_(inputs["rope_sin"])
    base_code = torch.einsum(
        "bgtv,gvr->bgtr",
        inputs["value"],
        inputs["base_left"],
    )
    caches["base"][:, :, 1:2].copy_(base_code)
    base_pre = torch.einsum(
        "bgtr,grd->bgtd",
        base_code,
        inputs["base_right"],
    )
    base_pre.add_(inputs["base_bias"][None, :, None])
    base_post = _apply_rope(
        base_pre,
        inputs["rope_cos"][None, None],
        inputs["rope_sin"][None, None],
    )
    residual = torch.einsum(
        "bgtd,gdr->bgtr",
        key - base_post,
        inputs["residual_encoder"],
    )
    caches["residual"][:, :, 1:2].copy_(residual)


def _fused_append(
    inputs: dict[str, Tensor],
    key: Tensor,
    caches: dict[str, Tensor],
) -> None:
    conditional_router_append_decode(
        key,
        inputs["value"],
        base_left=inputs["base_left"],
        base_right=inputs["base_right"],
        base_bias=inputs["base_bias"],
        residual_encoder=inputs["residual_encoder"],
        rope_cos=inputs["rope_cos"],
        rope_sin=inputs["rope_sin"],
        value_cache=caches["value"],
        base_cache=caches["base"],
        residual_cache=caches["residual"],
        rope_cos_cache=caches["cos"],
        rope_sin_cache=caches["sin"],
        start=1,
        write_rope=True,
    )


def _append_caches(*, device: torch.device) -> dict[str, Tensor]:
    return {
        "value": torch.zeros(
            1, 2, 2, 80, dtype=torch.bfloat16, device=device
        ),
        "base": torch.zeros(
            1, 2, 2, 16, dtype=torch.bfloat16, device=device
        ),
        "residual": torch.zeros(
            1, 2, 2, 8, dtype=torch.bfloat16, device=device
        ),
        "cos": torch.zeros(2, 64, dtype=torch.bfloat16, device=device),
        "sin": torch.zeros(2, 64, dtype=torch.bfloat16, device=device),
    }


def _time_cuda(call: Callable[[], object], *, warmup: int, repeat: int) -> float:
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        call()
    stop.record()
    stop.synchronize()
    return float(start.elapsed_time(stop)) / repeat


def main() -> None:
    wall_started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-length", type=int, default=65536)
    parser.add_argument("--correctness-length", type=int, default=4097)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--reference-repeat", type=int, default=3)
    parser.add_argument("--append-repeat", type=int, default=1000)
    parser.add_argument("--selection-repeat", type=int, default=1000)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    assert torch.cuda.is_available()
    assert torch.cuda.get_device_capability()[0] >= 8
    assert args.sequence_length > 0 and args.correctness_length > 0
    assert args.warmup >= 0 and args.repeat > 0 and args.reference_repeat > 0
    assert args.append_repeat > 0 and args.selection_repeat > 0
    device = torch.device("cuda")
    prepare_mapped_host_paged_attention_extension()

    correctness_inputs = _inputs(args.correctness_length, device=device)
    observed = _fused(correctness_inputs)
    expected = _reference(correctness_inputs)
    difference = observed - expected
    page_budget = min(128, int(observed.shape[-1]))
    expected_observed_pages = select_fixed_group_max_pages(
        observed,
        pages_per_kv_head=page_budget,
        pinned_prefix_pages=1,
    )
    observed_pages = select_fixed_group_max_pages_cuda(
        observed,
        pages_per_kv_head=page_budget,
        pinned_prefix_pages=1,
    )
    expected_pages = select_fixed_group_max_pages(
        expected,
        pages_per_kv_head=page_budget,
        pinned_prefix_pages=1,
    )
    page_agreement = float((observed_pages == expected_pages).float().mean())
    selector_agreement = float(
        (observed_pages == expected_observed_pages).float().mean()
    )
    torch.testing.assert_close(observed, expected, rtol=1.0e-2, atol=2.0e-2)
    torch.testing.assert_close(
        observed_pages,
        expected_observed_pages,
        rtol=0,
        atol=0,
    )

    append_inputs = _inputs(1, device=device)
    append_key = torch.randn(
        1,
        1,
        2,
        128,
        dtype=torch.bfloat16,
        device=device,
    ).transpose(1, 2)
    expected_caches = _append_caches(device=device)
    observed_caches = _append_caches(device=device)
    _reference_append(append_inputs, append_key, expected_caches)
    _fused_append(append_inputs, append_key, observed_caches)
    torch.cuda.synchronize()
    base_difference = observed_caches["base"] - expected_caches["base"]
    residual_difference = (
        observed_caches["residual"] - expected_caches["residual"]
    )
    torch.testing.assert_close(
        observed_caches["value"], expected_caches["value"]
    )
    torch.testing.assert_close(
        observed_caches["base"],
        expected_caches["base"],
        rtol=2.0e-2,
        atol=2.0e-2,
    )
    torch.testing.assert_close(
        observed_caches["residual"],
        expected_caches["residual"],
        rtol=2.0e-2,
        atol=2.0e-2,
    )
    torch.testing.assert_close(observed_caches["cos"], expected_caches["cos"])
    torch.testing.assert_close(observed_caches["sin"], expected_caches["sin"])
    append_fused_ms = _time_cuda(
        lambda: _fused_append(append_inputs, append_key, observed_caches),
        warmup=10,
        repeat=args.append_repeat,
    )
    append_reference_ms = _time_cuda(
        lambda: _reference_append(append_inputs, append_key, expected_caches),
        warmup=1,
        repeat=min(args.append_repeat, 100),
    )

    benchmark_inputs = _inputs(args.sequence_length, device=device)
    benchmark_page_scores = _fused(benchmark_inputs)
    selected_count = min(128, int(benchmark_page_scores.shape[-1]))
    selected_output = torch.empty(
        int(benchmark_page_scores.shape[0]),
        int(benchmark_page_scores.shape[1]),
        selected_count,
        dtype=torch.long,
        device=device,
    )
    benchmark_expected_pages = select_fixed_group_max_pages(
        benchmark_page_scores,
        pages_per_kv_head=128,
        pinned_prefix_pages=1,
    )
    benchmark_observed_pages = select_fixed_group_max_pages_cuda(
        benchmark_page_scores,
        pages_per_kv_head=128,
        pinned_prefix_pages=1,
        output=selected_output,
    )
    torch.cuda.synchronize()
    benchmark_selector_agreement = float(
        (benchmark_observed_pages == benchmark_expected_pages).float().mean()
    )
    torch.testing.assert_close(
        benchmark_observed_pages,
        benchmark_expected_pages,
        rtol=0,
        atol=0,
    )
    fused_selection_ms = _time_cuda(
        lambda: select_fixed_group_max_pages_cuda(
            benchmark_page_scores,
            pages_per_kv_head=128,
            pinned_prefix_pages=1,
            output=selected_output,
        ),
        warmup=10,
        repeat=args.selection_repeat,
    )
    pytorch_selection_ms = _time_cuda(
        lambda: select_fixed_group_max_pages(
            benchmark_page_scores,
            pages_per_kv_head=128,
            pinned_prefix_pages=1,
        ),
        warmup=1,
        repeat=min(args.selection_repeat, 100),
    )
    fused_ms = _time_cuda(
        lambda: _fused(benchmark_inputs),
        warmup=args.warmup,
        repeat=args.repeat,
    )
    reference_ms = _time_cuda(
        lambda: _reference(benchmark_inputs),
        warmup=1,
        repeat=args.reference_repeat,
    )
    payload = {
        "format": "basisserve.conditional_router_page32_lse_validation.v2",
        "status": "complete",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(sys.argv),
        "environment": {
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
        },
        "geometry": {
            "sequence_length": args.sequence_length,
            "correctness_length": args.correctness_length,
            "value_rank": 80,
            "base_rank": 16,
            "residual_rank": 8,
            "query_key_dim": 128,
            "page_size": 32,
        },
        "correctness": {
            "maximum_absolute_page_lse_error": float(difference.abs().max()),
            "mean_absolute_page_lse_error": float(difference.abs().mean()),
            "selected_page_id_agreement": page_agreement,
            "fused_selector_page_id_agreement": selector_agreement,
            "fused_selector_64k_page_id_agreement": (
                benchmark_selector_agreement
            ),
            "maximum_absolute_append_base_error": float(
                base_difference.abs().max()
            ),
            "maximum_absolute_append_residual_error": float(
                residual_difference.abs().max()
            ),
        },
        "timing": {
            "fused_cuda_ms": fused_ms,
            "pytorch_ms": reference_ms,
            "speedup": reference_ms / fused_ms,
            "warmup": args.warmup,
            "fused_repeat": args.repeat,
            "reference_repeat": args.reference_repeat,
            "fused_append_ms": append_fused_ms,
            "pytorch_append_ms": append_reference_ms,
            "append_speedup": append_reference_ms / append_fused_ms,
            "append_repeat": args.append_repeat,
            "fused_selection_ms": fused_selection_ms,
            "pytorch_selection_ms": pytorch_selection_ms,
            "selection_speedup": pytorch_selection_ms / fused_selection_ms,
            "selection_repeat": args.selection_repeat,
        },
        "wall_seconds": time.perf_counter() - wall_started,
    }
    output = args.output_json.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
