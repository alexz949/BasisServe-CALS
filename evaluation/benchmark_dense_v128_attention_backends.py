#!/usr/bin/env python3
"""Compare dense V128 SDPA backends with the deployed Triton attention paths."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import shlex
import statistics
import sys
from typing import Callable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
from torch import Tensor  # noqa: E402
from torch.nn import functional as F  # noqa: E402
from torch.nn.attention import SDPBackend, sdpa_kernel  # noqa: E402

from basisserve.kernels.compressed_v_decode_attention import (  # noqa: E402
    compressed_v_prefill_attention,
)
from vllm.v1.attention.ops.triton_decode_attention import (  # noqa: E402
    decode_attention_fwd,
)


FORMAT = "basisserve.dense_v128_attention_backends.v1"


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _measure(
    function: Callable[[], Tensor],
    *,
    warmup: int,
    iterations: int,
) -> dict[str, float]:
    result: Tensor | None = None
    for _ in range(warmup):
        result = function()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    for start, end in zip(starts, ends, strict=True):
        start.record()
        result = function()
        end.record()
    torch.cuda.synchronize()
    assert result is not None
    timings = [start.elapsed_time(end) for start, end in zip(starts, ends, strict=True)]
    ordered = sorted(timings)
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "minimum_ms": min(timings),
        "p50_ms": statistics.median(timings),
        "p95_ms": ordered[p95_index],
        "maximum_ms": max(timings),
    }


def _relative_l2(reference: Tensor, observed: Tensor) -> float:
    squared_error = 0.0
    squared_reference = 0.0
    for batch_index in range(int(reference.shape[0])):
        reference_slice = reference[batch_index].float()
        difference = observed[batch_index].float() - reference_slice
        squared_error += float(torch.sum(difference * difference))
        squared_reference += float(torch.sum(reference_slice * reference_slice))
    return math.sqrt(squared_error / max(squared_reference, 1e-30))


def _sdpa(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    backend: SDPBackend | None,
    causal: bool,
    scale: float,
) -> Tensor:
    if backend is None:
        return F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=0.0,
            is_causal=causal,
            scale=scale,
            enable_gqa=True,
        )
    with sdpa_kernel(backend):
        return F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=0.0,
            is_causal=causal,
            scale=scale,
            enable_gqa=True,
        )


def _paged_cache(tensor: Tensor, page_size: int) -> Tensor:
    batch, heads, sequence_length, width = map(int, tensor.shape)
    assert sequence_length % page_size == 0
    pages_per_sequence = sequence_length // page_size
    return (
        tensor.transpose(1, 2)
        .reshape(batch, pages_per_sequence, page_size, heads, width)
        .reshape(batch * pages_per_sequence, page_size, heads, width)
        .contiguous()
    )


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--query-heads", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--kv-splits", type=int, default=8)
    parser.add_argument("--decode-warmup", type=int, default=10)
    parser.add_argument("--decode-iterations", type=int, default=50)
    parser.add_argument("--prefill-warmup", type=int, default=1)
    parser.add_argument("--prefill-iterations", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    assert min(
        args.batch_size,
        args.context,
        args.query_heads,
        args.kv_heads,
        args.head_dim,
        args.page_size,
        args.kv_splits,
        args.decode_iterations,
        args.prefill_iterations,
    ) > 0
    assert args.query_heads % args.kv_heads == 0
    assert args.context % args.page_size == 0
    assert args.decode_warmup >= 0 and args.prefill_warmup >= 0

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    dtype = torch.bfloat16
    scale = args.head_dim**-0.5
    generator = torch.Generator(device=device).manual_seed(args.seed)

    decode_query = torch.randn(
        args.batch_size,
        args.query_heads,
        1,
        args.head_dim,
        dtype=dtype,
        device=device,
        generator=generator,
    )
    decode_key = torch.randn(
        args.batch_size,
        args.kv_heads,
        args.context,
        args.head_dim,
        dtype=dtype,
        device=device,
        generator=generator,
    )
    decode_value = torch.randn(
        decode_key.shape,
        dtype=dtype,
        device=device,
        generator=generator,
    )
    key_cache = _paged_cache(decode_key, args.page_size)
    value_cache = _paged_cache(decode_value, args.page_size)
    pages_per_sequence = args.context // args.page_size
    block_table = torch.arange(
        args.batch_size * pages_per_sequence,
        dtype=torch.int32,
        device=device,
    ).view(args.batch_size, pages_per_sequence)
    sequence_lengths = torch.full(
        (args.batch_size,),
        args.context,
        dtype=torch.int32,
        device=device,
    )
    paged_output = torch.empty(
        args.batch_size,
        args.query_heads,
        args.head_dim,
        dtype=dtype,
        device=device,
    )
    paged_lse = torch.empty(
        args.batch_size,
        args.query_heads,
        dtype=torch.float32,
        device=device,
    )
    paged_workspace = torch.empty(
        args.batch_size,
        args.query_heads,
        args.kv_splits,
        args.head_dim + 1,
        dtype=torch.float32,
        device=device,
    )
    unit_scale = torch.ones((), dtype=torch.float32, device=device)

    def paged_triton_decode() -> Tensor:
        decode_attention_fwd(
            decode_query[:, :, 0],
            key_cache,
            value_cache,
            paged_output,
            paged_lse,
            block_table,
            sequence_lengths,
            paged_workspace,
            args.kv_splits,
            scale,
            page_size=args.page_size,
            logit_cap=0.0,
            k_scale=unit_scale,
            v_scale=unit_scale,
        )
        return paged_output

    decode_functions = {
        "sdpa_default": lambda: _sdpa(
            decode_query,
            decode_key,
            decode_value,
            backend=None,
            causal=False,
            scale=scale,
        ),
        "sdpa_flash": lambda: _sdpa(
            decode_query,
            decode_key,
            decode_value,
            backend=SDPBackend.FLASH_ATTENTION,
            causal=False,
            scale=scale,
        ),
        "sdpa_math": lambda: _sdpa(
            decode_query,
            decode_key,
            decode_value,
            backend=SDPBackend.MATH,
            causal=False,
            scale=scale,
        ),
        "vllm_paged_triton": paged_triton_decode,
    }
    decode_reference = decode_functions["sdpa_math"]()
    decode_results = {}
    for name, function in decode_functions.items():
        observed = function()
        if observed.ndim == 3:
            observed = observed.unsqueeze(2)
        decode_results[name] = {
            "timing": _measure(
                function,
                warmup=args.decode_warmup,
                iterations=args.decode_iterations,
            ),
            "relative_l2_error_vs_math": _relative_l2(decode_reference, observed),
        }

    output = args.output_json.expanduser().resolve()
    payload: dict[str, object] = {
        "format": FORMAT,
        "status": "decode_complete_prefill_pending",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(sys.argv),
        "environment": {
            "hostname": platform.node(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "vllm": __import__("vllm").__version__,
            "gpu": torch.cuda.get_device_name(device),
            "capability": list(torch.cuda.get_device_capability(device)),
            "dtype": str(dtype),
        },
        "geometry": {
            "batch_size": args.batch_size,
            "context": args.context,
            "query_heads_per_tp_rank": args.query_heads,
            "kv_heads_per_tp_rank": args.kv_heads,
            "qk_dim": args.head_dim,
            "value_dim": args.head_dim,
            "page_size": args.page_size,
            "kv_splits": args.kv_splits,
        },
        "protocol": {
            "execution": "eager CUDA launches without CUDA Graph",
            "decode_warmup": args.decode_warmup,
            "decode_iterations": args.decode_iterations,
            "prefill_warmup": args.prefill_warmup,
            "prefill_iterations": args.prefill_iterations,
        },
        "decode": decode_results,
        "prefill": {},
    }
    _write_json(output, payload)
    print(json.dumps({"event": "decode_complete", "path": str(output)}), flush=True)

    prefill_query = torch.randn(
        args.batch_size,
        args.query_heads,
        args.context,
        args.head_dim,
        dtype=dtype,
        device=device,
        generator=generator,
    )
    prefill_key = torch.randn(
        args.batch_size,
        args.kv_heads,
        args.context,
        args.head_dim,
        dtype=dtype,
        device=device,
        generator=generator,
    )
    prefill_value = torch.randn(
        prefill_key.shape,
        dtype=dtype,
        device=device,
        generator=generator,
    )
    prefill_functions = {
        "sdpa_default": lambda: _sdpa(
            prefill_query,
            prefill_key,
            prefill_value,
            backend=None,
            causal=True,
            scale=scale,
        ),
        "sdpa_flash": lambda: _sdpa(
            prefill_query,
            prefill_key,
            prefill_value,
            backend=SDPBackend.FLASH_ATTENTION,
            causal=True,
            scale=scale,
        ),
        "sdpa_math": lambda: _sdpa(
            prefill_query,
            prefill_key,
            prefill_value,
            backend=SDPBackend.MATH,
            causal=True,
            scale=scale,
        ),
        "basisserve_triton": lambda: compressed_v_prefill_attention(
            prefill_query,
            prefill_key,
            prefill_value,
            scale=scale,
        ),
    }
    prefill_reference = prefill_functions["sdpa_math"]()
    prefill_results = {}
    for name, function in prefill_functions.items():
        observed = function()
        prefill_results[name] = {
            "timing": _measure(
                function,
                warmup=args.prefill_warmup,
                iterations=args.prefill_iterations,
            ),
            "relative_l2_error_vs_math": _relative_l2(prefill_reference, observed),
        }

    payload["status"] = "complete"
    payload["prefill"] = prefill_results
    _write_json(output, payload)
    print(json.dumps(payload, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
