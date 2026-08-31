#!/usr/bin/env python3
"""Evaluate Qwen3-8B MLP ``down_proj`` C1 factors on WikiText-2."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import sys
import time
from typing import Any, Mapping

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basisserve.core.mlp_down_c1 import (  # noqa: E402
    install_mlp_down_c1,
    load_mlp_down_c1_manifest,
)
from evaluation.capture_attention_o_proj_ppl_snapshots import (  # noqa: E402
    _git_commit,
    _installed_version,
)
from evaluation.eval_attention_o_proj_collective_ppl import (  # noqa: E402
    _atomic_json,
    _eval_ppl_fp32_loss,
)


FORMAT = "basisserve.qwen3_8b.mlp_down_c1_wikitext2_ppl.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--factor-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--dataset", default="wikitext2")
    parser.add_argument("--split", default="test")
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument(
        "--model-dtype", choices=("bfloat16", "float16"), default="bfloat16"
    )
    parser.add_argument(
        "--attn-implementation", choices=("eager", "sdpa"), default="sdpa"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--torch-num-threads", type=int, default=4)
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if min(args.seqlen, args.batch_size, args.torch_num_threads) <= 0:
        raise ValueError("sequence, batch, and thread counts must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("MLP C1 PPL evaluation requires CUDA")
    torch.set_num_threads(args.torch_num_threads)
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("--device must be CUDA")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    model_path = Path(args.model_path).expanduser().resolve()
    factor_dir = Path(args.factor_dir).expanduser().resolve()
    output_path = Path(args.output_json).expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite result: {output_path}")
    factor_manifest = load_mlp_down_c1_manifest(
        factor_dir, model_config_path=model_path / "config.json"
    )
    if factor_manifest["model"]["model_type"] != "qwen3":
        raise ValueError("factor artifact is not for Qwen3")

    model_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[args.model_dtype]
    started = time.perf_counter()
    print(f"[MLP C1 PPL] loading model={model_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=True, use_fast=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        dtype=model_dtype,
        attn_implementation=args.attn_implementation,
        device_map={"": str(device)},
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).eval()
    model.config.use_cache = False
    installation = install_mlp_down_c1(
        model,
        factor_dir,
        distributed=False,
        factor_dtype=model_dtype,
    )
    if len(installation) != int(model.config.num_hidden_layers):
        raise RuntimeError("MLP C1 installation did not cover every layer")
    torch.cuda.empty_cache()
    result = _eval_ppl_fp32_loss(
        model,
        tokenizer,
        dataset=args.dataset,
        split=args.split,
        seqlen=args.seqlen,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        max_tokens=args.max_tokens,
    )
    elapsed = time.perf_counter() - started
    payload: Mapping[str, Any] = {
        "format": FORMAT,
        "status": "complete",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(sys.argv),
        "git_commit": _git_commit(),
        "model": str(model_path),
        "factor_dir": str(factor_dir),
        "factor_manifest": factor_manifest,
        "installation": installation,
        "evaluation": result,
        "ppl": result["ppl"],
        "elapsed_seconds": elapsed,
        "environment": {
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": _installed_version("transformers"),
            "cuda_device": torch.cuda.get_device_name(device),
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "torch_num_threads": torch.get_num_threads(),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_path, payload)
    print(
        f"[MLP C1 PPL] complete ppl={result['ppl']:.9f} "
        f"output={output_path} seconds={elapsed:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
