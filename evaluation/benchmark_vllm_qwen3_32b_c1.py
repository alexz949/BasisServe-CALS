#!/usr/bin/env python3
"""Benchmark fixed-batch Qwen3-32B TP4 serving in vLLM."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import statistics
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basisserve.core.qwen3_32b_tp4_decode import file_sha256  # noqa: E402
from basisserve.vllm import (  # noqa: E402
    DENSE_DIFFKV_MODEL_ARCHITECTURE,
    MODEL_ARCHITECTURE,
)


ARMS = ("dense", "dense_diffkv_v128", "compact_v64")
EXECUTION_MODES = ("eager", "cuda_graph")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--factor-dir", type=Path)
    parser.add_argument("--result-sha256")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--prefill-tokens", type=int, default=2048)
    parser.add_argument("--decode-tokens", type=int, default=128)
    parser.add_argument("--warmup-decode-tokens", type=int, default=2)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.82)
    parser.add_argument(
        "--execution-mode",
        choices=EXECUTION_MODES,
        default="eager",
        help="Use eager execution or a full decode-only CUDA Graph.",
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> tuple[Path, Path | None]:
    model = args.model.expanduser().resolve()
    if not (model / "config.json").is_file():
        raise FileNotFoundError(model / "config.json")
    for name in ("batch_size", "prefill_tokens", "decode_tokens"):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"{name} must be positive")
    if args.warmup_decode_tokens < 2:
        raise ValueError("warmup_decode_tokens must exercise at least one decode step")
    if not 0.0 < args.gpu_memory_utilization < 1.0:
        raise ValueError("gpu_memory_utilization must be between zero and one")

    if args.arm != "compact_v64":
        if args.factor_dir is not None or args.result_sha256 is not None:
            raise ValueError("dense arms do not accept C1 factors")
        return model, None

    if args.factor_dir is None or not args.result_sha256:
        raise ValueError("the compact_v64 arm requires factors and result SHA256")
    factor_dir = args.factor_dir.expanduser().resolve()
    result_path = factor_dir / "result.json"
    observed = file_sha256(result_path)
    if observed != args.result_sha256:
        raise ValueError(
            f"factor result hash mismatch: expected {args.result_sha256}, "
            f"observed {observed}"
        )
    return model, factor_dir


def _repeat_to_length(token_ids: list[int], length: int) -> list[int]:
    if not token_ids:
        raise ValueError("benchmark seed text produced no tokens")
    repeats = (length + len(token_ids) - 1) // len(token_ids)
    return (token_ids * repeats)[:length]


def _build_prompts(tokenizer: Any, batch_size: int, length: int, tag: str):
    from vllm.inputs import TokensPrompt

    body = tokenizer.encode(
        "BasisServe evaluates tensor parallel attention and communication. ",
        add_special_tokens=False,
    )
    prompts = []
    for request_index in range(batch_size):
        prefix = tokenizer.encode(
            f"{tag} request {request_index}: ",
            add_special_tokens=False,
        )
        prompt_token_ids = _repeat_to_length(prefix + body, length)
        prompts.append(TokensPrompt(prompt_token_ids=prompt_token_ids))
    return prompts


def _request_metrics(outputs: list[Any]) -> list[Any]:
    metrics = [output.metrics for output in outputs]
    if any(metric is None for metric in metrics):
        raise RuntimeError("vLLM request metrics are disabled")
    return metrics


def _run_admission_barrier_batch(
    llm: Any,
    prompts: list[Any],
    sampling_params: Any,
) -> tuple[list[Any], float]:
    """Queue the complete batch behind a paused scheduler, then release it."""

    llm.sleep(level=0, mode="keep")
    request_ids = llm.enqueue(
        prompts,
        sampling_params,
        use_tqdm=False,
    )
    if len(request_ids) != len(prompts):
        raise RuntimeError("vLLM did not enqueue the complete fixed batch")
    start = time.perf_counter()
    llm.wake_up(tags=["scheduling"])
    outputs = llm.wait_for_completion(use_tqdm=False)
    return outputs, time.perf_counter() - start


def _distribution(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, int(0.95 * len(ordered)))
    return {
        "min": ordered[0],
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "p95": ordered[p95_index],
        "max": ordered[-1],
    }


def _summarize_run(
    outputs: list[Any],
    *,
    wall_seconds: float,
    batch_size: int,
    prefill_tokens: int,
    decode_tokens: int,
) -> dict[str, Any]:
    metrics = _request_metrics(outputs)
    generated = [
        list(map(int, output.outputs[0].token_ids))
        for output in outputs
    ]
    if len(outputs) != batch_size:
        raise RuntimeError(f"vLLM returned {len(outputs)} requests, expected {batch_size}")
    if any(len(tokens) != decode_tokens for tokens in generated):
        raise RuntimeError("a request did not produce the required decode length")

    scheduled_start = min(metric.scheduled_ts for metric in metrics)
    all_first_tokens = max(metric.first_token_ts for metric in metrics)
    all_last_tokens = max(metric.last_token_ts for metric in metrics)
    prefill_seconds = all_first_tokens - scheduled_start
    decode_seconds = all_last_tokens - all_first_tokens
    if prefill_seconds <= 0.0 or decode_seconds <= 0.0:
        raise RuntimeError("vLLM returned invalid prefill/decode timestamps")

    prompt_token_count = batch_size * prefill_tokens
    decode_step_token_count = batch_size * (decode_tokens - 1)
    serialized_tokens = json.dumps(generated, separators=(",", ":")).encode()
    return {
        "wall_seconds": wall_seconds,
        "batch_prefill_seconds": prefill_seconds,
        "batch_decode_seconds": decode_seconds,
        "batch_inference_seconds": all_last_tokens - scheduled_start,
        "prefill_tokens_per_second": prompt_token_count / prefill_seconds,
        "decode_tokens_per_second": decode_step_token_count / decode_seconds,
        "inter_token_latency_seconds": decode_seconds / (decode_tokens - 1),
        "output_tokens_per_second": (batch_size * decode_tokens) / wall_seconds,
        "requests_per_second": batch_size / wall_seconds,
        "request_prefill_seconds": _distribution(
            [
                float(metric.first_token_ts - metric.scheduled_ts)
                for metric in metrics
            ]
        ),
        "request_decode_seconds": _distribution(
            [float(metric.last_token_ts - metric.first_token_ts) for metric in metrics]
        ),
        "generated_token_sha256": hashlib.sha256(serialized_tokens).hexdigest(),
        "first_request_token_ids": generated[0],
    }


def main() -> None:
    args = _parse_args()
    model, factor_dir = _validate_args(args)

    import torch
    import vllm
    from vllm import LLM, SamplingParams

    max_model_len = args.prefill_tokens + args.decode_tokens
    max_num_batched_tokens = args.batch_size * args.prefill_tokens
    hf_overrides = None
    if args.arm == "compact_v64":
        assert factor_dir is not None
        hf_overrides = {
            "architectures": [MODEL_ARCHITECTURE],
            "basisserve_c1_factor_dir": str(factor_dir),
            "basisserve_c1_result_sha256": args.result_sha256,
        }
    elif args.arm == "dense_diffkv_v128":
        hf_overrides = {
            "architectures": [DENSE_DIFFKV_MODEL_ARCHITECTURE],
        }

    use_cuda_graph = args.execution_mode == "cuda_graph"
    compilation_config = None
    if use_cuda_graph:
        # Isolate CUDA Graph replay from torch.compile/Inductor so the eager
        # baseline and this arm differ only in decode launch orchestration.
        compilation_config = {
            "mode": "NONE",
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [args.batch_size],
        }

    llm = LLM(
        model=str(model),
        tensor_parallel_size=4,
        dtype="bfloat16",
        max_model_len=max_model_len,
        max_num_seqs=args.batch_size,
        max_num_batched_tokens=max_num_batched_tokens,
        enable_chunked_prefill=False,
        enable_prefix_caching=False,
        async_scheduling=False,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=not use_cuda_graph,
        compilation_config=compilation_config,
        disable_custom_all_reduce=True,
        disable_log_stats=False,
        hf_overrides=hf_overrides,
    )
    tokenizer = llm.get_tokenizer()
    warmup_prompts = _build_prompts(
        tokenizer,
        args.batch_size,
        args.prefill_tokens,
        "warmup",
    )
    measured_prompts = _build_prompts(
        tokenizer,
        args.batch_size,
        args.prefill_tokens,
        "measured",
    )

    warmup_sampling = SamplingParams(
        temperature=0.0,
        max_tokens=args.warmup_decode_tokens,
        ignore_eos=True,
        detokenize=False,
    )
    measured_sampling = SamplingParams(
        temperature=0.0,
        max_tokens=args.decode_tokens,
        ignore_eos=True,
        detokenize=False,
    )
    warmup_outputs, _ = _run_admission_barrier_batch(
        llm,
        warmup_prompts,
        warmup_sampling,
    )
    if any(
        len(output.outputs[0].token_ids) != args.warmup_decode_tokens
        for output in warmup_outputs
    ):
        raise RuntimeError("warmup did not produce the requested token count")

    outputs, wall_seconds = _run_admission_barrier_batch(
        llm,
        measured_prompts,
        measured_sampling,
    )
    measured = _summarize_run(
        outputs,
        wall_seconds=wall_seconds,
        batch_size=args.batch_size,
        prefill_tokens=args.prefill_tokens,
        decode_tokens=args.decode_tokens,
    )

    payload = {
        "format": "basisserve.vllm.qwen3_32b.tp4_fixed_batch.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
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
            "model_architecture": (
                DENSE_DIFFKV_MODEL_ARCHITECTURE
                if args.arm == "dense_diffkv_v128"
                else MODEL_ARCHITECTURE
                if args.arm == "compact_v64"
                else "Qwen3ForCausalLM"
            ),
            "attention_backend": (
                "basisserve_grouped_triton_diffkv_v128"
                if args.arm == "dense_diffkv_v128"
                else "basisserve_grouped_triton_diffkv_v64"
                if args.arm == "compact_v64"
                else "vllm_flash_attention"
            ),
            "key_head_size": 128,
            "value_head_size": 64 if args.arm == "compact_v64" else 128,
            "tensor_parallel_size": 4,
            "batch_size": args.batch_size,
            "prefill_tokens_per_request": args.prefill_tokens,
            "decode_tokens_per_request": args.decode_tokens,
            "warmup_decode_tokens_per_request": args.warmup_decode_tokens,
            "max_model_len": max_model_len,
            "max_num_batched_tokens": max_num_batched_tokens,
            "enable_chunked_prefill": False,
            "enable_prefix_caching": False,
            "async_scheduling": False,
            "admission_barrier": "scheduler_level0_pause_enqueue_wake",
            "execution_mode": args.execution_mode,
            "enforce_eager": not use_cuda_graph,
            "compilation_mode": "NONE",
            "cudagraph_mode": "FULL_DECODE_ONLY" if use_cuda_graph else "NONE",
            "cudagraph_capture_sizes": [args.batch_size] if use_cuda_graph else [],
            "dtype": "bfloat16",
            "gpu_memory_utilization": args.gpu_memory_utilization,
        },
        "metrics": measured,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(measured, sort_keys=True), flush=True)
    print(f"wrote {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
