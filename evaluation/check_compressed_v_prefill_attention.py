#!/usr/bin/env python3
"""Check compact-Value Triton prefill against the causal PyTorch reference."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basisserve.kernels.compressed_v_decode_attention import (  # noqa: E402
    compressed_v_prefill_attention,
    reference_compressed_v_prefill_attention,
)


def _positive_ints(value: str) -> tuple[int, ...]:
    selected = tuple(int(item) for item in value.split(","))
    if not selected or any(item <= 0 for item in selected):
        raise argparse.ArgumentTypeError("values must be comma-separated positive integers")
    return selected


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-lengths", type=_positive_ints, default=(5, 8, 129))
    parser.add_argument("--value-dims", type=_positive_ints, default=(48, 64, 80, 112))
    parser.add_argument("--rtol", type=float, default=2.0e-2)
    parser.add_argument("--atol", type=float, default=2.0e-2)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if args.rtol < 0.0 or args.atol < 0.0:
        raise ValueError("tolerances must be nonnegative")

    device = torch.device("cuda:0")
    cases: list[dict[str, float | int | str]] = []
    for value_dim in args.value_dims:
        for sequence_length in args.sequence_lengths:
            generator = torch.Generator(device=device).manual_seed(
                20260825 + value_dim + sequence_length
            )
            query = torch.randn(
                2,
                8,
                sequence_length,
                128,
                dtype=torch.bfloat16,
                device=device,
                generator=generator,
            )
            key = torch.randn(
                2,
                2,
                sequence_length,
                128,
                dtype=torch.bfloat16,
                device=device,
                generator=generator,
            )
            value = torch.randn(
                2,
                2,
                sequence_length,
                value_dim,
                dtype=torch.bfloat16,
                device=device,
                generator=generator,
            )
            observed = compressed_v_prefill_attention(query, key, value)
            expected = reference_compressed_v_prefill_attention(query, key, value)
            difference = observed.float() - expected.float()
            relative_l2 = float(
                difference.norm() / expected.float().norm().clamp_min(1.0e-12)
            )
            maximum_absolute = float(difference.abs().max())
            torch.testing.assert_close(
                observed,
                expected,
                rtol=args.rtol,
                atol=args.atol,
            )
            record: dict[str, float | int | str] = {
                "status": "pass",
                "sequence_length": sequence_length,
                "value_dim": value_dim,
                "relative_l2_error": relative_l2,
                "maximum_absolute_error": maximum_absolute,
            }
            cases.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)

    payload = {
        "format": "basisserve.compact_v_prefill_correctness.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass",
        "environment": {
            "hostname": platform.node(),
            "python": sys.version,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
        },
        "configuration": {
            "sequence_lengths": list(args.sequence_lengths),
            "value_dims": list(args.value_dims),
            "rtol": args.rtol,
            "atol": args.atol,
        },
        "summary": {
            "case_count": len(cases),
            "maximum_relative_l2_error": max(
                float(case["relative_l2_error"]) for case in cases
            ),
            "maximum_absolute_error": max(
                float(case["maximum_absolute_error"]) for case in cases
            ),
        },
        "cases": cases,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["summary"], sort_keys=True), flush=True)
    print(f"wrote {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
