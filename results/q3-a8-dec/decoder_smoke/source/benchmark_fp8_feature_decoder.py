#!/usr/bin/env python3
"""Microbenchmark BF16 versus cast-based and direct-FP8 feature decoders."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import statistics
import sys
from typing import Callable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from basisserve.kernels.feature_ragged_allgather import (  # noqa: E402
    decode_feature_major,
    decode_feature_major_e4m3,
)
from basisserve.kernels.fp8_wire import (  # noqa: E402
    FP8_E4M3_MAX,
    decode_e4m3_bytes,
    quantize_e4m3_static,
    quantize_e4m3_tensorwise_col_major,
)


FORMAT = "basisserve.fp8_feature_decoder_benchmark.v1"


def _time_cuda(
    operation: Callable[[], torch.Tensor],
    *,
    warmup: int,
    repeats: int,
) -> tuple[float, torch.Tensor]:
    output: torch.Tensor | None = None
    for _ in range(warmup):
        output = operation()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        output = operation()
        stop.record()
        stop.synchronize()
        samples.append(float(start.elapsed_time(stop)))
    assert output is not None
    return statistics.median(samples), output


def _error(reference: torch.Tensor, observed: torch.Tensor) -> dict[str, float]:
    reference_fp32 = reference.float()
    observed_fp32 = observed.float()
    delta = observed_fp32 - reference_fp32
    denominator = reference_fp32.square().mean().sqrt().clamp_min(1.0e-12)
    return {
        "relative_rmse": float((delta.square().mean().sqrt() / denominator).item()),
        "mean_absolute_error": float(delta.abs().mean().item()),
        "maximum_absolute_error": float(delta.abs().amax().item()),
        "cosine_similarity": float(
            torch.nn.functional.cosine_similarity(
                reference_fp32.flatten(),
                observed_fp32.flatten(),
                dim=0,
            ).item()
        ),
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=64)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--hidden", type=int, default=5120)
    parser.add_argument("--local-k", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    if min(args.rows, args.k, args.hidden, args.local_k, args.warmup, args.repeats) <= 0:
        raise ValueError("benchmark dimensions and iteration counts must be positive")
    if args.k % 16 or args.hidden % 16 or args.local_k > args.k:
        raise ValueError("K/hidden must be divisible by 16 and local K must not exceed K")
    if not torch.cuda.is_available():
        raise RuntimeError("FP8 feature decoder benchmark requires CUDA")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)

    coordinates = torch.randn(
        args.rows,
        args.k,
        device=device,
        dtype=torch.bfloat16,
    )
    decoder = (
        torch.randn(
            args.k,
            args.hidden,
            device=device,
            dtype=torch.bfloat16,
        )
        / args.k**0.5
    ).contiguous()
    wire_scale = (coordinates.float().abs().amax() / FP8_E4M3_MAX).clamp_min(
        torch.finfo(torch.float32).tiny
    )
    codes = quantize_e4m3_static(coordinates, wire_scale)
    arena_uint8 = codes.transpose(0, 1).contiguous().view(torch.uint8)
    scaled_decoder = (decoder.float() * wire_scale).to(torch.bfloat16).contiguous()
    decoder_fp8, decoder_scale = quantize_e4m3_tensorwise_col_major(
        scaled_decoder
    )
    unit_scale = torch.ones((), device=device, dtype=torch.float32)
    arena_bf16 = coordinates.transpose(0, 1).contiguous()

    bf16_ms, bf16_output = _time_cuda(
        lambda: decode_feature_major(arena_bf16, decoder),
        warmup=args.warmup,
        repeats=args.repeats,
    )
    cast_ms, cast_output = _time_cuda(
        lambda: decode_feature_major(
            decode_e4m3_bytes(arena_uint8, dtype=torch.bfloat16),
            scaled_decoder,
        ),
        warmup=args.warmup,
        repeats=args.repeats,
    )
    direct_ms, direct_output = _time_cuda(
        lambda: decode_feature_major_e4m3(
            arena_uint8,
            decoder_fp8,
            arena_scale=unit_scale,
            decoder_scale=decoder_scale,
        ),
        warmup=args.warmup,
        repeats=args.repeats,
    )
    local_coordinates = coordinates[:, : args.local_k].contiguous()
    quantize_ms, _ = _time_cuda(
        lambda: quantize_e4m3_static(local_coordinates, wire_scale),
        warmup=args.warmup,
        repeats=args.repeats,
    )

    payload: dict[str, object] = {
        "format": FORMAT,
        "status": "complete",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(sys.argv),
        "shape": {
            "rows": args.rows,
            "global_k": args.k,
            "local_k": args.local_k,
            "hidden": args.hidden,
        },
        "protocol": {
            "warmup": args.warmup,
            "repeats": args.repeats,
            "wire_dtype": "float8_e4m3fn",
            "decoder_fp8_scale": "tensorwise",
            "fp8_accumulation": "torch._scaled_mm use_fast_accum=False",
            "direct_path_pack": "feature-major uint8 -> row-major E4M3 byte transpose",
        },
        "latency_ms": {
            "bf16_decoder": bf16_ms,
            "fp8_wire_cast_to_bf16_decoder": cast_ms,
            "direct_fp8_decoder": direct_ms,
            "local_fp8_quantize": quantize_ms,
        },
        "speedup": {
            "direct_fp8_vs_bf16_decoder": bf16_ms / direct_ms,
            "direct_fp8_vs_cast_path": cast_ms / direct_ms,
        },
        "error_vs_bf16": {
            "wire_fp8_decoder_bf16": _error(bf16_output, cast_output),
            "wire_and_decoder_fp8": _error(bf16_output, direct_output),
        },
        "storage_bytes": {
            "bf16_decoder": decoder.numel() * decoder.element_size(),
            "fp8_decoder": decoder_fp8.numel() * decoder_fp8.element_size(),
            "bf16_arena": arena_bf16.numel() * arena_bf16.element_size(),
            "fp8_arena": arena_uint8.numel() * arena_uint8.element_size(),
        },
        "environment": {
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "gpu": torch.cuda.get_device_name(device),
            "capability": list(torch.cuda.get_device_capability(device)),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
    }
    output = Path(args.output_json).expanduser().resolve()
    _write_json(output, payload)
    print(json.dumps(payload, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
