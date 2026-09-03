#!/usr/bin/env python3
"""Run lm-eval zero-shot tasks on a finalized Qwen3-32B C1 allocation."""

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
from safetensors.torch import load_file
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.allocate_qwen3_32b_c1_tp_source_global_kl import (  # noqa: E402
    HEAD_DIM,
    HIDDEN_SIZE,
    NUM_KV_HEADS,
    NUM_LAYERS,
    NUM_QUERY_HEADS,
    _fold_ragged_to_padded_weights,
    _sha256,
)
from evaluation import eval_qwen3_32b_c1_wikitext as c1_quality  # noqa: E402
from evaluation.eval_qwen3_32b_c1_wikitext import (  # noqa: E402
    _load_ragged_results,
    _load_results as _load_uniform_results,
    install_ragged_c1_factors,
    install_c1_factors as install_uniform_c1_factors,
)
from evaluation.lm_eval_task_specs import (  # noqa: E402
    task_evaluation_specifications,
)
from evaluation.hf_legacy_dataset_compat import (  # noqa: E402
    install_mathqa_alias_compatibility,
)


FORMAT = "basisserve.qwen3_32b.gqa_c1_lm_eval_mcq.v1"
ALLOCATION_FORMAT = "basisserve.qwen3_32b.gqa_c1.tp_source_global_kl_allocation.v2"
MODEL_LABEL = "Qwen3-32B"


def activate_model_profile(name: str) -> None:
    global FORMAT, ALLOCATION_FORMAT, MODEL_LABEL
    global HEAD_DIM, HIDDEN_SIZE, NUM_KV_HEADS, NUM_LAYERS, NUM_QUERY_HEADS
    c1_quality.activate_model_profile(name)
    if name == "qwen3_32b":
        slug = "qwen3_32b"
        MODEL_LABEL = "Qwen3-32B"
    else:
        assert name == "qwen3_8b"
        slug = "qwen3_8b"
        MODEL_LABEL = "Qwen3-8B-Base"
    HEAD_DIM = c1_quality.HEAD_DIM
    HIDDEN_SIZE = c1_quality.HIDDEN_SIZE
    NUM_KV_HEADS = c1_quality.NUM_KV_HEADS
    NUM_LAYERS = c1_quality.NUM_LAYERS
    NUM_QUERY_HEADS = c1_quality.NUM_QUERY_HEADS
    FORMAT = f"basisserve.{slug}.gqa_c1_lm_eval_mcq.v1"
    ALLOCATION_FORMAT = c1_quality.LAYER_ALLOCATION_FORMAT


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


def _decoder_layers(model: nn.Module) -> Sequence[nn.Module]:
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise TypeError("expected a Qwen-style decoder model")
    return layers


def _validate_model_geometry(model: nn.Module) -> None:
    config = model.config
    observed = (
        str(config.model_type),
        int(config.num_hidden_layers),
        int(config.hidden_size),
        int(config.num_attention_heads),
        int(config.num_key_value_heads),
        int(getattr(config, "head_dim", 0)),
    )
    expected = (
        "qwen3",
        NUM_LAYERS,
        HIDDEN_SIZE,
        NUM_QUERY_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
    )
    assert observed == expected, f"unexpected {MODEL_LABEL} geometry: {observed}"


def _load_allocation(allocation_dir: Path, model_path: Path) -> dict[str, Any]:
    result_path = allocation_dir / "result.json"
    if not result_path.is_file():
        raise FileNotFoundError(result_path)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("format") != ALLOCATION_FORMAT or result.get("status") != "complete":
        raise ValueError("C1 Global-KL allocation is incomplete or incompatible")
    if result.get("model_config_sha256") != _sha256(model_path / "config.json"):
        raise ValueError("C1 allocation belongs to another model config")
    artifacts = result.get("selected_artifacts", {})
    assert set(map(int, artifacts)) == set(range(NUM_LAYERS)), (
        f"C1 allocation does not contain all {NUM_LAYERS} selected layers"
    )
    schedule = result.get("selection", {}).get("selected_schedule", ())
    if len(schedule) != NUM_LAYERS or any(len(ranks) != NUM_KV_HEADS for ranks in schedule):
        raise ValueError("C1 allocation has an incompatible selected rank schedule")
    return result


@torch.no_grad()
def install_selected_c1_factors(
    model: nn.Module,
    allocation_dir: Path,
    result: Mapping[str, Any],
) -> list[dict[str, Any]]:
    _validate_model_geometry(model)
    records = []
    schedule = result["selection"]["selected_schedule"]
    for layer_index, layer in enumerate(_decoder_layers(model)):
        started = time.perf_counter()
        artifact = result["selected_artifacts"][str(layer_index)]
        path = allocation_dir / artifact["file"]
        if _sha256(path) != artifact["sha256"]:
            raise ValueError(f"C1 artifact hash mismatch at layer {layer_index}")
        payload = load_file(str(path), device="cpu")
        if set(payload) != {
            "value_coordinate_encoders",
            "head_output_decoders",
            "source_ranks",
        }:
            raise ValueError(f"unexpected C1 tensors at layer {layer_index}")
        ranks = tuple(map(int, payload["source_ranks"].tolist()))
        if list(ranks) != list(map(int, schedule[layer_index])):
            raise ValueError(f"C1 rank schedule mismatch at layer {layer_index}")
        maximum_rank = max(ranks)
        A = payload["value_coordinate_encoders"]
        D = payload["head_output_decoders"]
        if tuple(A.shape) != (NUM_KV_HEADS, HEAD_DIM, maximum_rank):
            raise ValueError(f"unexpected C1 encoder shape at layer {layer_index}")
        if tuple(D.shape) != (NUM_QUERY_HEADS, maximum_rank, HIDDEN_SIZE):
            raise ValueError(f"unexpected C1 decoder shape at layer {layer_index}")

        v_proj = layer.self_attn.v_proj
        o_proj = layer.self_attn.o_proj
        dense_k = layer.self_attn.k_proj
        if (
            not isinstance(v_proj, nn.Linear)
            or not isinstance(o_proj, nn.Linear)
            or v_proj.bias is not None
            or o_proj.bias is not None
            or v_proj.weight.device.type != "cuda"
            or o_proj.weight.device.type != "cuda"
        ):
            raise TypeError(f"unsupported/offloaded Qwen projections at layer {layer_index}")
        padded_v, padded_o = _fold_ragged_to_padded_weights(
            dense_v_weight=v_proj.weight.detach(),
            A=A,
            D=D,
            source_ranks=ranks,
        )
        v_proj.weight.copy_(padded_v.to(dtype=v_proj.weight.dtype))
        o_proj.weight.copy_(
            padded_o.to(device=o_proj.weight.device, dtype=o_proj.weight.dtype)
        )
        if layer.self_attn.k_proj is not dense_k:
            raise RuntimeError(f"C1 installation changed K at layer {layer_index}")
        records.append(
            {
                "layer": layer_index,
                "factor_file": str(path.relative_to(allocation_dir)),
                "factor_sha256": artifact["sha256"],
                "source_ranks": list(ranks),
                "maximum_padded_rank": maximum_rank,
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        del payload, A, D, padded_v, padded_o
        torch.cuda.empty_cache()
    return records


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    checkpoint = parser.add_mutually_exclusive_group(required=True)
    checkpoint.add_argument("--allocation-dir")
    checkpoint.add_argument("--factor-dir")
    checkpoint.add_argument("--ragged-factor-dir")
    parser.add_argument("--tasks", default="boolq")
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
        choices=("auto", "balanced", "balanced_low_0"),
        default="balanced",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-memory-per-gpu-gib", type=int, default=42)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--log-samples", action="store_true")
    parser.add_argument("--torch-num-threads", type=int, default=8)
    return parser.parse_args()


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> None:
    if args.max_length <= 0 or args.max_memory_per_gpu_gib <= 0:
        raise ValueError("length and GPU-memory limits must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("lm-eval C1 evaluation requires CUDA")
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
    for index in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(index)

    model_path = Path(args.model).expanduser().resolve()
    output_path = Path(args.output_json).expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    allocation_dir: Path | None = None
    factor_dir: Path | None = None
    ragged_factor_dir: Path | None = None
    allocation_result: dict[str, Any] | None = None
    uniform_result: dict[str, Any] | None = None
    ragged_result: dict[str, Any] | None = None
    if args.allocation_dir is not None:
        allocation_dir = Path(args.allocation_dir).expanduser().resolve()
        allocation_result = _load_allocation(allocation_dir, model_path)
    elif args.ragged_factor_dir is not None:
        ragged_factor_dir = Path(args.ragged_factor_dir).expanduser().resolve()
        ragged_result = _load_ragged_results(ragged_factor_dir, model_path)
    else:
        factor_dir = Path(args.factor_dir).expanduser().resolve()
        uniform_result = _load_uniform_results(factor_dir, model_path)

    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=args.local_files_only, use_fast=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        dtype=_dtype(args.model_dtype),
        low_cpu_mem_usage=True,
        local_files_only=args.local_files_only,
        attn_implementation=args.attn_implementation,
        device_map=args.device_map,
        max_memory={
            index: f"{args.max_memory_per_gpu_gib}GiB"
            for index in range(torch.cuda.device_count())
        },
    ).eval()
    model.config.use_cache = True
    install_started = time.perf_counter()
    if allocation_result is not None:
        assert allocation_dir is not None
        installation = install_selected_c1_factors(
            model, allocation_dir, allocation_result
        )
    elif ragged_result is not None:
        assert ragged_factor_dir is not None
        installation = install_ragged_c1_factors(
            model, ragged_factor_dir, ragged_result
        )
    else:
        assert factor_dir is not None and uniform_result is not None
        installation = install_uniform_c1_factors(model, factor_dir, uniform_result)
    install_seconds = time.perf_counter() - install_started

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

    if allocation_result is not None:
        assert allocation_dir is not None
        result_path = allocation_dir / "result.json"
        selection = allocation_result["selection"]
        arm = f"c1_layer_global_kl_{selection['selected_candidate']}"
        checkpoint_record = {
            "directory": str(allocation_dir),
            "result_sha256": _sha256(result_path),
            "format": allocation_result["format"],
            "selected_candidate": selection["selected_candidate"],
        }
        compression = selection["selected_accounting"]
        runtime_description = (
            "ragged C1 coordinates zero-padded into native Qwen V/O slots; "
            "function-equivalent quality path, not a cache-performance benchmark"
        )
    elif ragged_result is not None:
        assert ragged_factor_dir is not None
        result_path = ragged_factor_dir / "result.json"
        fit_config = ragged_result["fit_config"]
        selection = ragged_result["selection"]
        source_rank_sum = int(selection["source_rank_sum"])
        dense_source_rank_sum = NUM_LAYERS * NUM_KV_HEADS * HEAD_DIM
        sweeps = int(fit_config["encoder_sweeps"])
        arm = f"c1_aasvd_global_kl_ragged_als_s{sweeps}"
        checkpoint_record = {
            "directory": str(ragged_factor_dir),
            "result_sha256": _sha256(result_path),
            "format": ragged_result["format"],
            "allocation_selected_candidate": fit_config[
                "allocation_selected_candidate"
            ],
            "aggregate": ragged_result["aggregate"],
        }
        compression = {
            "target": "value_cache_only",
            "key_cache": "dense",
            "num_query_heads": NUM_QUERY_HEADS,
            "num_physical_kv_heads": NUM_KV_HEADS,
            "head_dim": HEAD_DIM,
            "source_rank_sum": source_rank_sum,
            "dense_source_rank_sum": dense_source_rank_sum,
            "retained_v_ratio": source_rank_sum / dense_source_rank_sum,
            "v_cache_compression_ratio": 1.0
            - source_rank_sum / dense_source_rank_sum,
            "rank_histogram": fit_config["rank_histogram"],
        }
        runtime_description = (
            "ragged C1 coordinates zero-padded into native Qwen V/O slots; "
            "function-equivalent quality path, not a cache-performance benchmark"
        )
    else:
        assert factor_dir is not None and uniform_result is not None
        result_path = factor_dir / "results.json"
        rank = int(uniform_result["fit_config"]["cache_rank_per_head"])
        arm = f"c1_uniform_rank{rank}_als"
        checkpoint_record = {
            "directory": str(factor_dir),
            "result_sha256": _sha256(result_path),
            "format": uniform_result["format"],
            "aggregate": uniform_result["aggregate"],
        }
        compression = {
            "target": "value_cache_only",
            "key_cache": "dense",
            "num_query_heads": NUM_QUERY_HEADS,
            "num_physical_kv_heads": NUM_KV_HEADS,
            "head_dim": HEAD_DIM,
            "rank_per_physical_kv_head": rank,
            "source_rank_sum": NUM_LAYERS * NUM_KV_HEADS * rank,
            "retained_v_ratio": rank / HEAD_DIM,
            "v_cache_compression_ratio": 1.0 - rank / HEAD_DIM,
        }
        runtime_description = (
            f"uniform rank-{rank} C1 coordinates zero-padded into native Qwen V/O "
            "slots; function-equivalent quality path, not a cache-performance benchmark"
        )
    payload = {
        "format": FORMAT,
        "status": "complete",
        "command": shlex.join(sys.argv),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "arm": arm,
        "model": {
            "path": str(model_path),
            "config_sha256": _sha256(model_path / "config.json"),
        },
        "checkpoint": checkpoint_record,
        "compression": compression,
        "quality_reference_runtime": {
            "description": runtime_description,
            "installation_seconds": install_seconds,
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
