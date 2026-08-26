#!/usr/bin/env python3
"""Run lm-eval zero-shot MCQ tasks on dense or PaLU-M GQA models."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import shlex
import sys
import time
from typing import Any, Mapping, Sequence

import lm_eval
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager
from lm_eval.utils import make_table
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.eval_gqa_palu_m_wikitext import (  # noqa: E402
    _decoder_layers,
    _load_checkpoint,
    _sha256,
    install_palu_m_factors,
)
from evaluation.lm_eval_task_specs import (  # noqa: E402
    task_evaluation_specifications,
)
from evaluation.hf_legacy_dataset_compat import (  # noqa: E402
    install_mathqa_alias_compatibility,
)


FORMAT = "basisserve.gqa.palu_m_lm_eval_mcq.v1"
DEFAULT_TASKS = (
    "openbookqa",
    "hellaswag",
    "piqa",
    "arc_easy",
    "arc_challenge",
    "winogrande",
)


def _dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[name]


def _task_names(raw: str) -> list[str]:
    names = [name.strip() for name in raw.split(",") if name.strip()]
    if not names:
        raise ValueError("--tasks must contain at least one task")
    if len(names) != len(set(names)):
        raise ValueError("--tasks contains duplicates")
    return names


def _layer_ranks(
    manifest: Mapping[str, Any],
    *,
    num_layers: int,
) -> list[list[int]]:
    compression = manifest["compression"]
    if "layer_ranks" in compression:
        ranks = [list(map(int, row)) for row in compression["layer_ranks"]]
    else:
        uniform = list(map(int, compression["ranks"]))
        ranks = [uniform] * num_layers
    if len(ranks) != num_layers:
        raise ValueError("PaLU rank schedule does not cover every model layer")
    return ranks


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("dense", "palu_m"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    parser.add_argument(
        "--task-fewshot",
        help=(
            "comma-separated task=count overrides; selected tasks omitted from "
            "the mapping use zero-shot"
        ),
    )
    parser.add_argument("--batch-size", default="8")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--limit", type=float)
    parser.add_argument("--output-json", required=True)
    parser.add_argument(
        "--model-dtype", choices=("bfloat16", "float16"), default="bfloat16"
    )
    parser.add_argument(
        "--attn-implementation", choices=("eager", "sdpa"), default="sdpa"
    )
    parser.add_argument(
        "--device-map",
        choices=("none", "auto", "balanced", "balanced_low_0"),
        default="none",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-memory-per-gpu-gib", type=int, default=44)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--log-samples", action="store_true")
    parser.add_argument("--torch-num-threads", type=int, default=8)
    return parser.parse_args()


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> None:
    if args.arm == "palu_m" and args.checkpoint_dir is None:
        raise ValueError("--checkpoint-dir is required for --arm palu_m")
    if args.max_length <= 0 or args.max_memory_per_gpu_gib <= 0:
        raise ValueError("length and GPU-memory limits must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("lm-eval PaLU evaluation requires CUDA")
    tasks = _task_names(args.tasks)
    dataset_loader_compatibility = (
        install_mathqa_alias_compatibility() if "mathqa" in tasks else None
    )
    evaluation_tasks, global_num_fewshot, task_num_fewshot = (
        task_evaluation_specifications(tasks, args.task_fewshot)
    )
    torch.set_num_threads(args.torch_num_threads)
    device = torch.device(args.device)
    torch.cuda.set_device(device)

    model_path = Path(args.model).expanduser().resolve()
    output_path = Path(args.output_json).expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint_dir: Path | None = None
    manifest: dict[str, Any] | None = None
    factor_payload: dict[str, torch.Tensor] | None = None
    if args.arm == "palu_m":
        checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
        manifest, factor_payload = _load_checkpoint(checkpoint_dir, model_path)

    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        local_files_only=args.local_files_only,
        use_fast=True,
    )
    model_kwargs: dict[str, Any] = {
        "dtype": _dtype(args.model_dtype),
        "low_cpu_mem_usage": True,
        "local_files_only": args.local_files_only,
        "attn_implementation": args.attn_implementation,
    }
    if args.device_map != "none":
        model_kwargs["device_map"] = args.device_map
        model_kwargs["max_memory"] = {
            index: f"{args.max_memory_per_gpu_gib}GiB"
            for index in range(torch.cuda.device_count())
        }
    model = AutoModelForCausalLM.from_pretrained(str(model_path), **model_kwargs)
    if args.device_map == "none":
        model.to(device)
    model.eval()
    model.config.use_cache = True

    geometry = {
        "num_query_heads": int(model.config.num_attention_heads),
        "num_physical_kv_heads": int(model.config.num_key_value_heads),
        "head_dim": int(
            getattr(model.config, "head_dim", 0)
            or model.config.hidden_size // model.config.num_attention_heads
        ),
    }
    installation: list[dict[str, Any]] = []
    checkpoint_record: dict[str, Any] | None = None
    compression: Mapping[str, Any] | None = None
    if args.arm == "palu_m":
        assert manifest is not None and factor_payload is not None
        compression = manifest["compression"]
        for key, value in geometry.items():
            if value != int(compression[key]):
                raise ValueError(f"model/checkpoint geometry mismatch for {key}")
        installation = install_palu_m_factors(
            model,
            factor_payload,
            layer_ranks=_layer_ranks(
                manifest,
                num_layers=len(_decoder_layers(model)),
            ),
            head_dim=int(compression["head_dim"]),
            require_cuda_resident=args.device_map != "none",
        )
        checkpoint_record = {
            "directory": str(checkpoint_dir),
            "manifest_sha256": _sha256(checkpoint_dir / "manifest.json"),
            "artifact_sha256": manifest["artifact"]["sha256"],
            "format": manifest["format"],
        }
    del factor_payload

    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        device=str(device),
        batch_size=args.batch_size,
        max_length=args.max_length,
        add_bos_token=False,
    )
    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=evaluation_tasks,
        num_fewshot=global_num_fewshot,
        task_manager=TaskManager(),
        limit=args.limit,
        log_samples=args.log_samples,
    )
    if results is None:
        raise RuntimeError("lm-evaluation-harness returned no results")

    payload = {
        "format": FORMAT,
        "status": "complete",
        "command": shlex.join(sys.argv),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "arm": args.arm,
        "model": {
            "path": str(model_path),
            "config_sha256": _sha256(model_path / "config.json"),
        },
        "checkpoint": checkpoint_record,
        "compression": compression,
        "quality_reference_runtime": {
            "description": (
                "stored latent V factors reconstructed immediately before attention; "
                "mathematically equivalent quality path, not a cache-performance benchmark"
                if args.arm == "palu_m"
                else "unaltered dense Hugging Face reference model"
            ),
            "layers": installation,
        },
        "protocol": {
            "tasks": tasks,
            "num_fewshot": global_num_fewshot,
            **(
                {"task_num_fewshot": task_num_fewshot}
                if task_num_fewshot is not None
                else {}
            ),
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "limit": args.limit,
            "log_samples": args.log_samples,
            "model_dtype": args.model_dtype,
            "attention_implementation": args.attn_implementation,
            "device_map": args.device_map,
            "dataset_loader_compatibility": dataset_loader_compatibility,
        },
        "evaluation": results,
        "environment": {
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "datasets": importlib.metadata.version("datasets"),
            "huggingface_hub": importlib.metadata.version("huggingface-hub"),
            "lm_eval": importlib.metadata.version("lm-eval"),
            "cuda_devices": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ],
            "peak_cuda_allocated_bytes": {
                str(index): int(torch.cuda.max_memory_allocated(index))
                for index in range(torch.cuda.device_count())
            },
            "torch_num_threads": torch.get_num_threads(),
        },
    }
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output_path)
    print(make_table(results), flush=True)
    print(f"wrote {output_path}", flush=True)


if __name__ == "__main__":
    evaluate(parse_args())
