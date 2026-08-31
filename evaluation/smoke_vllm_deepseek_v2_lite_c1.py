#!/usr/bin/env python3
"""Smoke-test dense or uniform-R128 C1 DeepSeek-V2-Lite in vLLM TP4."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basisserve.core.deepseek_v2_lite_tp4_c1 import file_sha256  # noqa: E402
from basisserve.vllm import (  # noqa: E402
    DEEPSEEK_V2_LITE_MODEL_ARCHITECTURE,
)


ARMS = ("dense", "c1_r128")
FORMAT = "basisserve.vllm.deepseek_v2_lite.tp4_smoke.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--factor-dir", type=Path)
    parser.add_argument("--result-sha256")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--prefill-tokens", type=int, default=128)
    parser.add_argument("--decode-tokens", type=int, default=8)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> tuple[Path, Path | None]:
    model = args.model.expanduser().resolve()
    if not (model / "config.json").is_file():
        raise FileNotFoundError(model / "config.json")
    if min(args.batch_size, args.prefill_tokens, args.decode_tokens) <= 0:
        raise ValueError("batch, prefill, and decode sizes must be positive")
    if not 0.0 < args.gpu_memory_utilization < 1.0:
        raise ValueError("gpu memory utilization must be between zero and one")
    if args.output_json.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_json}")
    if args.arm == "dense":
        if args.factor_dir is not None or args.result_sha256 is not None:
            raise ValueError("dense smoke does not accept C1 factors")
        return model, None
    if args.factor_dir is None or not args.result_sha256:
        raise ValueError("C1 smoke requires --factor-dir and --result-sha256")
    factor_dir = args.factor_dir.expanduser().resolve()
    observed = file_sha256(factor_dir / "results.json")
    if observed != args.result_sha256:
        raise ValueError(
            f"factor manifest hash mismatch: expected {args.result_sha256}, "
            f"observed {observed}"
        )
    return model, factor_dir


def repeat_to_length(token_ids: list[int], length: int) -> list[int]:
    if not token_ids:
        raise ValueError("tokenizer produced no seed tokens")
    repeats = (length + len(token_ids) - 1) // len(token_ids)
    return (token_ids * repeats)[:length]


def build_prompts(tokenizer: Any, batch_size: int, length: int):
    from vllm.inputs import TokensPrompt

    body = tokenizer.encode(
        "BasisServe validates DeepSeek MLA tensor-parallel serving. ",
        add_special_tokens=False,
    )
    return [
        TokensPrompt(
            prompt_token_ids=repeat_to_length(
                tokenizer.encode(
                    f"request {request_index}: ",
                    add_special_tokens=False,
                )
                + body,
                length,
            )
        )
        for request_index in range(batch_size)
    ]


def main() -> None:
    args = parse_args()
    model, factor_dir = validate_args(args)

    import torch
    import vllm
    from vllm import LLM, SamplingParams

    hf_overrides = None
    if args.arm == "c1_r128":
        assert factor_dir is not None
        hf_overrides = {
            "architectures": [DEEPSEEK_V2_LITE_MODEL_ARCHITECTURE],
            "basisserve_c1_factor_dir": str(factor_dir),
            "basisserve_c1_result_sha256": args.result_sha256,
        }
    llm = LLM(
        model=str(model),
        tensor_parallel_size=4,
        dtype="bfloat16",
        max_model_len=args.prefill_tokens + args.decode_tokens,
        max_num_seqs=args.batch_size,
        max_num_batched_tokens=args.batch_size * args.prefill_tokens,
        enable_chunked_prefill=False,
        enable_prefix_caching=False,
        async_scheduling=False,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        disable_custom_all_reduce=True,
        disable_log_stats=False,
        hf_overrides=hf_overrides,
    )
    prompts = build_prompts(
        llm.get_tokenizer(),
        args.batch_size,
        args.prefill_tokens,
    )
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=args.decode_tokens,
        ignore_eos=True,
        detokenize=False,
    )
    start = time.perf_counter()
    outputs = llm.generate(prompts, sampling, use_tqdm=False)
    wall_seconds = time.perf_counter() - start
    generated = [list(map(int, output.outputs[0].token_ids)) for output in outputs]
    if len(generated) != args.batch_size:
        raise RuntimeError("vLLM returned the wrong request count")
    if any(len(tokens) != args.decode_tokens for tokens in generated):
        raise RuntimeError("vLLM returned the wrong decode length")
    serialized = json.dumps(generated, separators=(",", ":")).encode()

    payload = {
        "format": FORMAT,
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "arm": args.arm,
        "environment": {
            "hostname": platform.node(),
            "python": sys.version,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "vllm": vllm.__version__,
            "gpu_names": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ],
        },
        "configuration": {
            "model": str(model),
            "factor_dir": None if factor_dir is None else str(factor_dir),
            "result_sha256": args.result_sha256,
            "tensor_parallel_size": 4,
            "batch_size": args.batch_size,
            "prefill_tokens": args.prefill_tokens,
            "decode_tokens": args.decode_tokens,
            "dtype": "bfloat16",
            "enforce_eager": True,
            "gpu_memory_utilization": args.gpu_memory_utilization,
        },
        "metrics": {
            "wall_seconds": wall_seconds,
            "output_tokens_per_second": (
                args.batch_size * args.decode_tokens / wall_seconds
            ),
            "generated_token_sha256": hashlib.sha256(serialized).hexdigest(),
            "first_request_token_ids": generated[0],
        },
    }
    output_path = args.output_json.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["metrics"], sort_keys=True), flush=True)
    print(f"wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
