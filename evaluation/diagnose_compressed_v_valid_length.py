#!/usr/bin/env python3
"""Check fixed-capacity compressed-V Triton decode against a sliced cache."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from basisserve.kernels.compressed_v_decode_attention import (  # noqa: E402
    compressed_v_decode_attention_triton,
)


FORMAT = "basisserve.compressed_v_valid_length_diagnostic.v1"


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-length", type=int, default=257)
    parser.add_argument("--valid-length", type=int, default=193)
    parser.add_argument("--query-heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--qk-dim", type=int, default=128)
    parser.add_argument("--value-dim", type=int, default=96)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    shape = (1, args.kv_heads, args.cache_length)
    query = torch.randn(
        1,
        args.query_heads,
        1,
        args.qk_dim,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    key = torch.randn(
        *shape,
        args.qk_dim,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    value = torch.randn(
        *shape,
        args.value_dim,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    valid = torch.tensor(args.valid_length, dtype=torch.int64, device=device)

    observed = compressed_v_decode_attention_triton(
        query,
        key,
        value,
        valid_sequence_length=valid,
    )
    expected = compressed_v_decode_attention_triton(
        query,
        key[:, :, : args.valid_length],
        value[:, :, : args.valid_length],
    )
    torch.cuda.synchronize(device)
    difference = observed.float() - expected.float()
    maximum_absolute_error = float(difference.abs().max())
    relative_l2_error = float(
        torch.linalg.vector_norm(difference)
        / torch.linalg.vector_norm(expected.float()).clamp_min(1e-30)
    )
    finite = bool(torch.isfinite(observed).all())
    passed = finite and relative_l2_error <= 2e-2
    payload = {
        "format": FORMAT,
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(sys.argv),
        "configuration": vars(args) | {"output_json": str(args.output_json)},
        "environment": {
            "gpu": torch.cuda.get_device_name(device),
            "capability": list(torch.cuda.get_device_capability(device)),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "metrics": {
            "finite": finite,
            "maximum_absolute_error": maximum_absolute_error,
            "relative_l2_error": relative_l2_error,
            "passed": passed,
        },
    }
    output_path = args.output_json.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output_path)
    print(json.dumps(payload["metrics"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
