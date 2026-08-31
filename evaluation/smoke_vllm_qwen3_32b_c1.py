#!/usr/bin/env python3
"""Run a minimal Qwen3-32B uniform-C1 request through vLLM serving."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basisserve.core.qwen3_32b_tp4_decode import file_sha256  # noqa: E402
from basisserve.vllm import MODEL_ARCHITECTURE  # noqa: E402


DEFAULT_PROMPTS = (
    "The capital of France is",
    "In one sentence, tensor parallelism is",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--factor-dir", type=Path, required=True)
    parser.add_argument("--result-sha256", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.82)
    parser.add_argument("--prompt", action="append", dest="prompts")
    parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> tuple[Path, Path]:
    model = args.model.expanduser().resolve()
    factor_dir = args.factor_dir.expanduser().resolve()
    if not (model / "config.json").is_file():
        raise FileNotFoundError(model / "config.json")
    result_path = factor_dir / "result.json"
    observed = file_sha256(result_path)
    if observed != args.result_sha256:
        raise ValueError(
            f"factor result hash mismatch: expected {args.result_sha256}, observed {observed}"
        )
    if args.max_model_len <= 0 or args.max_tokens <= 0:
        raise ValueError("token limits must be positive")
    if not 0.0 < args.gpu_memory_utilization < 1.0:
        raise ValueError("GPU memory utilization must be between zero and one")
    return model, factor_dir


def _request_record(output: Any) -> dict[str, Any]:
    candidates = []
    for candidate in output.outputs:
        candidates.append(
            {
                "index": int(candidate.index),
                "token_ids": list(map(int, candidate.token_ids)),
                "text": candidate.text,
                "finish_reason": candidate.finish_reason,
            }
        )
    return {
        "request_id": str(output.request_id),
        "prompt": output.prompt,
        "prompt_token_ids": list(map(int, output.prompt_token_ids)),
        "outputs": candidates,
    }


def main() -> None:
    args = _parse_args()
    model, factor_dir = _validate_args(args)

    import torch
    import vllm
    from vllm import LLM, SamplingParams

    prompts = tuple(args.prompts or DEFAULT_PROMPTS)
    hf_overrides = {
        "architectures": [MODEL_ARCHITECTURE],
        "basisserve_c1_factor_dir": str(factor_dir),
        "basisserve_c1_result_sha256": args.result_sha256,
    }
    llm = LLM(
        model=str(model),
        tensor_parallel_size=4,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        max_num_seqs=len(prompts),
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        hf_overrides=hf_overrides,
    )
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
    )
    start = time.perf_counter()
    outputs = llm.generate(list(prompts), sampling, use_tqdm=False)
    elapsed = time.perf_counter() - start

    payload = {
        "format": "basisserve.vllm.qwen3_32b.compact_v64_tp4_smoke.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
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
            "factor_dir": str(factor_dir),
            "result_sha256": args.result_sha256,
            "architecture": MODEL_ARCHITECTURE,
            "tensor_parallel_size": 4,
            "c1_source_rank": 64,
            "c1_local_wire_width": 1024,
            "c1_global_wire_width": 4096,
            "wire_dtype": "bfloat16",
            "paged_key_head_size": 128,
            "paged_value_head_size": 64,
            "prefill_attention_backend": "basisserve_compact_v_triton",
            "decode_attention_backend": "grouped_triton_diffkv",
            "max_model_len": args.max_model_len,
            "max_tokens": args.max_tokens,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "enforce_eager": args.enforce_eager,
        },
        "metrics": {
            "request_count": len(outputs),
            "elapsed_seconds": elapsed,
            "requests_per_second": len(outputs) / elapsed,
            "generated_tokens": sum(
                len(candidate.token_ids)
                for output in outputs
                for candidate in output.outputs
            ),
        },
        "requests": [_request_record(output) for output in outputs],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["metrics"], sort_keys=True), flush=True)
    print(f"wrote {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
