#!/usr/bin/env python3
"""Evaluate Qwen3-8B MLP C1 factors on exact tokenized C4 windows."""

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
from typing import Any, Mapping

from safetensors.torch import load_file
import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM


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
    _sha256,
)
from evaluation.eval_attention_o_proj_collective_ppl import (  # noqa: E402
    _atomic_json,
)


FORMAT = "basisserve.qwen3_8b.mlp_down_c1_exact_windows_ppl.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--factor-dir", required=True)
    parser.add_argument("--windows", required=True)
    parser.add_argument("--window-start", type=int, required=True)
    parser.add_argument("--window-count", type=int, required=True)
    parser.add_argument("--split-name", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
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
    if min(args.window_count, args.batch_size, args.torch_num_threads) <= 0:
        raise ValueError("window count, batch size, and threads must be positive")
    if args.window_start < 0:
        raise ValueError("window start must be nonnegative")
    if not torch.cuda.is_available():
        raise RuntimeError("exact-window MLP C1 evaluation requires CUDA")
    torch.set_num_threads(args.torch_num_threads)
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("--device must be CUDA")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)

    model_path = Path(args.model_path).expanduser().resolve()
    factor_dir = Path(args.factor_dir).expanduser().resolve()
    windows_path = Path(args.windows).expanduser().resolve()
    output_path = Path(args.output_json).expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite result: {output_path}")
    payload = load_file(str(windows_path), device="cpu")
    if set(payload) != {"input_ids"}:
        raise ValueError("exact-window artifact must contain only input_ids")
    all_windows = payload["input_ids"].long().contiguous()
    stop = args.window_start + args.window_count
    if all_windows.ndim != 2 or stop > len(all_windows):
        raise ValueError("requested exact-window slice is out of range")
    windows = all_windows[args.window_start:stop].contiguous()
    windows_manifest_path = windows_path.parent / "manifest.json"
    if not windows_manifest_path.is_file():
        raise FileNotFoundError(windows_manifest_path)
    windows_manifest = json.loads(
        windows_manifest_path.read_text(encoding="utf-8")
    )
    if windows_manifest.get("artifact", {}).get("sha256") != _sha256(
        windows_path
    ):
        raise ValueError("exact-window artifact hash differs from its manifest")
    factor_manifest = load_mlp_down_c1_manifest(
        factor_dir, model_config_path=model_path / "config.json"
    )

    model_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[args.model_dtype]
    started = time.perf_counter()
    print(f"[MLP C1 windows] loading model={model_path}", flush=True)
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

    nll_sum = 0.0
    token_count = 0
    sequence_length = int(windows.shape[1])
    for start in range(0, len(windows), args.batch_size):
        stop = min(start + args.batch_size, len(windows))
        input_ids = windows[start:stop].to(device=device, non_blocking=True)
        outputs = model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            use_cache=False,
            return_dict=True,
        )
        logits = outputs.logits[:, :-1].float()
        labels = input_ids[:, 1:]
        nll_sum += float(
            F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                reduction="sum",
            ).item()
        )
        token_count += int(labels.numel())
        print(
            f"[MLP C1 windows] split={args.split_name} "
            f"windows={start}:{stop}/{len(windows)}",
            flush=True,
        )
        del input_ids, outputs, logits, labels
    mean_nll = nll_sum / token_count
    ppl = math.exp(mean_nll)
    elapsed = time.perf_counter() - started
    result: Mapping[str, Any] = {
        "format": FORMAT,
        "status": "complete",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(sys.argv),
        "git_commit": _git_commit(),
        "model": str(model_path),
        "factor_dir": str(factor_dir),
        "factor_manifest_sha256": _sha256(factor_dir / "manifest.json"),
        "relative_damping": factor_manifest["fit_config"].get(
            "relative_covariance_damping", 0.0
        ),
        "windows": {
            "path": str(windows_path),
            "sha256": _sha256(windows_path),
            "manifest_sha256": _sha256(windows_manifest_path),
            "start": args.window_start,
            "count": args.window_count,
            "sequence_length": sequence_length,
            "split_name": args.split_name,
        },
        "evaluation": {
            "nll_sum": nll_sum,
            "mean_nll": mean_nll,
            "ppl": ppl,
            "tokens": token_count,
            "batch_size": args.batch_size,
            "loss_dtype": "float32",
        },
        "ppl": ppl,
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
    _atomic_json(output_path, result)
    print(
        f"[MLP C1 windows] complete split={args.split_name} ppl={ppl:.9f} "
        f"output={output_path} seconds={elapsed:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
