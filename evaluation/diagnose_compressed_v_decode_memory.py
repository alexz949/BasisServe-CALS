#!/usr/bin/env python3
"""Attribute persistent and peak memory around compressed-V CUDA loading."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
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
    compressed_v_decode_attention_cuda,
    compressed_v_decode_attention_triton,
)


FORMAT = "basisserve.compressed_v_decode_memory_diagnostic.v1"


def _snapshot(label: str, device: torch.device) -> dict[str, int | str]:
    torch.cuda.synchronize(device)
    record: dict[str, int | str] = {
        "label": label,
        "allocated_bytes": torch.cuda.memory_allocated(device),
        "reserved_bytes": torch.cuda.memory_reserved(device),
        "maximum_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "maximum_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }
    print(json.dumps({"event": "memory_snapshot", **record}), flush=True)
    return record


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--context", type=int, default=1024)
    parser.add_argument("--value-dim", type=int, default=64)
    parser.add_argument("--splits", type=int, default=1)
    parser.add_argument("--query-heads", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--qk-dim", type=int, default=128)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    if args.batch <= 0 or args.context <= 0:
        raise ValueError("batch and context must be positive")
    if args.query_heads != 4 * args.kv_heads or args.qk_dim != 128:
        raise ValueError("diagnostic requires Qwen3 local GQA geometry")

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    generator = torch.Generator(device=device).manual_seed(20260825)
    dtype = torch.bfloat16
    scale = args.qk_dim**-0.5
    query = torch.randn(
        args.batch,
        args.query_heads,
        1,
        args.qk_dim,
        dtype=dtype,
        device=device,
        generator=generator,
    )
    key = torch.randn(
        args.batch,
        args.kv_heads,
        args.context,
        args.qk_dim,
        dtype=dtype,
        device=device,
        generator=generator,
    )
    value = torch.randn(
        args.batch,
        args.kv_heads,
        args.context,
        args.value_dim,
        dtype=dtype,
        device=device,
        generator=generator,
    )
    workspace = torch.empty(
        args.batch * args.query_heads,
        args.splits,
        args.value_dim + 2,
        dtype=torch.float32,
        device=device,
    )
    output = torch.empty(
        args.batch,
        args.query_heads,
        1,
        args.value_dim,
        dtype=dtype,
        device=device,
    )
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    snapshots = [_snapshot("inputs_and_preallocated_outputs", device)]

    triton_output = compressed_v_decode_attention_triton(
        query,
        key,
        value,
        scale=scale,
    )
    snapshots.append(_snapshot("after_first_triton_launch", device))
    del triton_output
    gc.collect()
    torch.cuda.empty_cache()
    snapshots.append(_snapshot("after_triton_release", device))

    cuda_output = compressed_v_decode_attention_cuda(
        query,
        key,
        value,
        scale=scale,
        splits=args.splits,
        workspace=workspace,
        output=output,
    )
    snapshots.append(_snapshot("after_first_cuda_launch", device))
    second_output = compressed_v_decode_attention_cuda(
        query,
        key,
        value,
        scale=scale,
        splits=args.splits,
        workspace=workspace,
        output=output,
    )
    snapshots.append(_snapshot("after_second_cuda_launch", device))
    if cuda_output.data_ptr() != output.data_ptr() or second_output.data_ptr() != output.data_ptr():
        raise AssertionError("CUDA launch did not reuse the preallocated output")
    del cuda_output, second_output
    gc.collect()
    torch.cuda.empty_cache()
    snapshots.append(_snapshot("after_cuda_release", device))

    payload = {
        "format": FORMAT,
        "status": "complete",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(sys.argv),
        "configuration": vars(args),
        "environment": {
            "gpu": torch.cuda.get_device_name(device),
            "capability": list(torch.cuda.get_device_capability(device)),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "snapshots": snapshots,
    }
    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output_path)
    print(json.dumps({"event": "result_written", "path": str(output_path)}), flush=True)


if __name__ == "__main__":
    main()
