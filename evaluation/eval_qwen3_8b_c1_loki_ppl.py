#!/usr/bin/env python3
"""Full WikiText-2 PPL for repository-faithful Loki on Qwen3 C1-V80."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import sys
import time
from typing import Any

from datasets import load_dataset
from safetensors.torch import load_file
import torch
import torch.distributed as dist
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basisserve.checkpoint.gqa_vo_qwen3 import (  # noqa: E402
    GQATiedVOQwen3Attention,
    install_qwen3_gqa_vo_als_export,
)
from basisserve.core.c1_k_reverse_shadow import ReverseShadowConfig  # noqa: E402


FORMAT = "basisserve.qwen3_8b.c1_v80_loki_wikitext_ppl.v1"
LOKI_FORMAT = "basisserve.qwen3_8b.loki_key_pca.v1"
LOKI_COMMIT = "005913bc0b64c3b54d0d96871f4e51e799d7b17b"
ARMS = ("dense", "c1_exact", "c1_exact_topk", "loki")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    digest = hashlib.sha256()
    digest.update(value.contiguous().numpy().tobytes())
    return digest.hexdigest()


def _atomic_text(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _parse_arms(value: str) -> tuple[str, ...]:
    arms = tuple(item.strip() for item in value.split(",") if item.strip())
    assert arms and len(arms) == len(set(arms))
    assert all(arm in ARMS for arm in arms)
    assert tuple(arm for arm in ARMS if arm in arms) == arms
    return arms


def _attention_modules(model: torch.nn.Module) -> list[GQATiedVOQwen3Attention]:
    modules = [layer.self_attn for layer in model.model.layers]
    assert all(isinstance(module, GQATiedVOQwen3Attention) for module in modules)
    return modules


def _load_loki_factors(
    factor_dir: Path,
    *,
    model: Path,
    layers: int,
    kv_heads: int,
    head_dim: int,
    rank: int,
) -> tuple[torch.Tensor, dict[str, Any], Path, Path]:
    result_path = factor_dir / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["format"] == LOKI_FORMAT and result["status"] == "complete"
    assert result["model"]["config_sha256"] == _sha256(model / "config.json")
    assert result["method"]["repository_commit"] == LOKI_COMMIT
    factor_path = factor_dir / result["artifacts"]["factors"]["file"]
    assert result["artifacts"]["factors"]["sha256"] == _sha256(factor_path)
    tensors = load_file(str(factor_path), device="cpu")
    projector = tensors["key_projector"]
    assert tuple(projector.shape[:3]) == (layers, kv_heads, head_dim)
    assert int(projector.shape[-1]) >= rank
    return projector[..., :rank].contiguous(), result, result_path, factor_path


def _wiki_stream(
    model_path: Path,
    dataset_cache: Path,
    *,
    split: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        use_fast=True,
    )
    dataset = load_dataset(
        "Salesforce/wikitext",
        "wikitext-2-raw-v1",
        split=split,
        cache_dir=str(dataset_cache),
        download_mode="reuse_dataset_if_exists",
    )
    text = "\n\n".join(str(row["text"]) for row in dataset)
    stream = tokenizer(text, return_tensors="pt").input_ids[0].to(torch.int32)
    return stream, {
        "repo": "Salesforce/wikitext",
        "config": "wikitext-2-raw-v1",
        "split": split,
        "fingerprint": dataset._fingerprint,
        "cache_dir": str(dataset_cache),
        "token_stream_sha256": _tensor_sha256(stream),
        "tokenized_total": int(stream.numel()),
    }


@torch.inference_mode()
def _score_block(
    model: torch.nn.Module,
    tokens: torch.Tensor,
    *,
    logit_block_size: int,
) -> dict[str, Any]:
    assert tokens.ndim == 2 and int(tokens.shape[0]) == 1
    assert int(tokens.shape[1]) > 1
    torch.cuda.synchronize(tokens.device)
    started = time.perf_counter()
    hidden = model.model(
        input_ids=tokens,
        use_cache=False,
        return_dict=False,
    )[0]
    loss_parts = []
    predictions = int(tokens.shape[1]) - 1
    for start in range(0, predictions, logit_block_size):
        stop = min(start + logit_block_size, predictions)
        logits = model.lm_head(hidden[:, start:stop]).float()
        labels = tokens[:, start + 1 : stop + 1]
        loss_parts.append(
            F.cross_entropy(
                logits.reshape(-1, int(logits.shape[-1])),
                labels.reshape(-1),
                reduction="sum",
            )
        )
    nll_sum = float(torch.stack(loss_parts).sum().item())
    torch.cuda.synchronize(tokens.device)
    return {
        "tokens": predictions,
        "nll_sum": nll_sum,
        "mean_nll": nll_sum / predictions,
        "ppl": math.exp(nll_sum / predictions),
        "seconds": time.perf_counter() - started,
    }


def _aggregate_arm(
    local_rows: list[dict[str, Any]],
    *,
    device: torch.device,
    local_seconds: float,
    local_peak_bytes: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    values = torch.tensor(
        [
            sum(float(row["nll_sum"]) for row in local_rows),
            sum(int(row["tokens"]) for row in local_rows),
            sum(float(row["mean_nll"]) for row in local_rows),
            len(local_rows),
        ],
        dtype=torch.float64,
    )
    wall = torch.tensor([local_seconds], dtype=torch.float64)
    peak = torch.tensor([local_peak_bytes], dtype=torch.int64)
    if dist.is_initialized():
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        dist.all_reduce(wall, op=dist.ReduceOp.MAX)
        dist.all_reduce(peak, op=dist.ReduceOp.MAX)
        gathered: list[list[dict[str, Any]] | None] = [
            None for _ in range(dist.get_world_size())
        ]
        dist.all_gather_object(gathered, local_rows)
        rows = [row for shard in gathered if shard is not None for row in shard]
    else:
        rows = list(local_rows)
    rows.sort(key=lambda row: int(row["block_index"]))
    nll_sum, tokens, block_mean_sum, blocks = values.tolist()
    weighted_mean = nll_sum / tokens
    repository_mean = block_mean_sum / blocks
    return {
        "blocks": int(blocks),
        "tokens": int(tokens),
        "nll_sum": nll_sum,
        "token_weighted_mean_nll": weighted_mean,
        "token_weighted_ppl": math.exp(weighted_mean),
        "repository_block_mean_nll": repository_mean,
        "repository_ppl": math.exp(repository_mean),
        "wall_seconds": float(wall.item()),
        "aggregate_tokens_per_second": tokens / float(wall.item()),
        "maximum_allocated_bytes": int(peak.item()),
        "maximum_allocated_gib": int(peak.item()) / (1 << 30),
    }, rows


def _evaluate_arm(
    arm: str,
    model: torch.nn.Module,
    stream: torch.Tensor,
    block_ranges: list[tuple[int, int]],
    *,
    rank_index: int,
    world_size: int,
    device: torch.device,
    logit_block_size: int,
    before_block: Any | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    if dist.is_initialized():
        dist.barrier()
    started = time.perf_counter()
    local_rows = []
    local_indices = list(range(rank_index, len(block_ranges), world_size))
    for progress, block_index in enumerate(local_indices, start=1):
        start, stop = block_ranges[block_index]
        if before_block is not None:
            before_block(stop - start)
        tokens = stream[start:stop].to(device=device, dtype=torch.long).unsqueeze(0)
        row = _score_block(
            model,
            tokens,
            logit_block_size=logit_block_size,
        )
        row.update(
            {
                "block_index": block_index,
                "token_start": start,
                "token_stop": stop,
            }
        )
        local_rows.append(row)
        print(
            f"[Loki PPL] rank={rank_index} arm={arm} block={block_index} "
            f"progress={progress}/{len(local_indices)} ppl={row['ppl']:.8f} "
            f"seconds={row['seconds']:.2f}",
            flush=True,
        )
    torch.cuda.synchronize(device)
    return _aggregate_arm(
        local_rows,
        device=device,
        local_seconds=time.perf_counter() - started,
        local_peak_bytes=torch.cuda.max_memory_allocated(device),
    )


def _comparison(
    candidate: dict[str, Any], baseline: dict[str, Any]
) -> dict[str, float]:
    weighted_delta = (
        candidate["token_weighted_mean_nll"] - baseline["token_weighted_mean_nll"]
    )
    repository_delta = (
        candidate["repository_block_mean_nll"] - baseline["repository_block_mean_nll"]
    )
    return {
        "token_weighted_mean_nll_delta": weighted_delta,
        "token_weighted_ppl_ratio": math.exp(weighted_delta),
        "repository_mean_nll_delta": repository_delta,
        "repository_ppl_ratio": math.exp(repository_delta),
    }


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Qwen3-8B C1-V80 + Loki WikiText-2 PPL",
        "",
        "Loki uses a pre-RoPE Key-PCA basis, projects post-RoPE Q/K at runtime, "
        "selects 25% of tokens independently for every Query head (2,048 at "
        "length 8,192), and recomputes exact 128-dimensional QK on the selected "
        "support. The exact-TopK control ranks with full 128-dimensional QK. The "
        "Value payload is the fixed C1-V80 ALS5 checkpoint in all C1 arms.",
        "",
        "| Arm | Repository PPL | Token-weighted PPL | Tokens | Wall time (s) | Peak GiB |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in payload["protocol"]["arms"]:
        row = payload["results"][arm]
        lines.append(
            f"| {arm} | {row['repository_ppl']:.8f} | "
            f"{row['token_weighted_ppl']:.8f} | {row['tokens']} | "
            f"{row['wall_seconds']:.2f} | {row['maximum_allocated_gib']:.3f} |"
        )
    lines.extend(
        [
            "",
            "`Repository PPL` reproduces Loki's equal weighting of each non-overlapping "
            "sequence block, including the shorter final block. `Token-weighted PPL` "
            "weights every predicted token equally. Context resets at every block boundary.",
            "",
            "The query-axis tiling and sparse gather avoid the official Python path's full "
            "8192x8192 materialization; they do not change the selected indices or attention "
            "equations.",
        ]
    )
    return "\n".join(lines) + "\n"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--c1-checkpoint", type=Path, required=True)
    parser.add_argument("--loki-factors", type=Path, required=True)
    parser.add_argument(
        "--dataset-cache",
        type=Path,
        default=REPO_ROOT / "results/cache/huggingface/datasets",
    )
    parser.add_argument("--sequence-length", type=int, default=8192)
    parser.add_argument("--routing-rank", type=int, default=32)
    parser.add_argument("--top-k-ratio", type=float, default=0.25)
    parser.add_argument("--query-block-size", type=int, default=64)
    parser.add_argument("--logit-block-size", type=int, default=128)
    parser.add_argument("--max-blocks", type=int)
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--torch-num-threads", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


@torch.inference_mode()
def main() -> None:
    args = _parser().parse_args()
    arms = _parse_arms(args.arms)
    assert (
        min(
            args.sequence_length,
            args.routing_rank,
            args.query_block_size,
            args.logit_block_size,
            args.torch_num_threads,
        )
        > 0
    )
    assert 0.0 < args.top_k_ratio <= 1.0
    assert args.max_blocks is None or args.max_blocks > 0
    torch.set_num_threads(args.torch_num_threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank_index = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank_index)))
    assert torch.cuda.is_available()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world_size > 1:
        dist.init_process_group(backend="gloo")

    started = time.perf_counter()
    model_path = args.model.expanduser().resolve()
    c1_checkpoint = args.c1_checkpoint.expanduser().resolve()
    loki_factor_dir = args.loki_factors.expanduser().resolve()
    dataset_cache = args.dataset_cache.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    c1_result_path = c1_checkpoint / "results.json"
    c1_result = json.loads(c1_result_path.read_text(encoding="utf-8"))
    assert c1_result["status"] == "complete"
    assert c1_result["fit_config"]["model_config_sha256"] == _sha256(
        model_path / "config.json"
    )
    value_rank = int(c1_result["fit_config"]["cache_rank_per_head"])
    assert value_rank == 80

    stream, dataset = _wiki_stream(model_path, dataset_cache, split="test")
    block_ranges = [
        (start, min(start + args.sequence_length, int(stream.numel())))
        for start in range(0, int(stream.numel()), args.sequence_length)
        if int(stream.numel()) - start > 1
    ]
    if args.max_blocks is not None:
        block_ranges = block_ranges[: args.max_blocks]
    assert block_ranges

    model = (
        AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            local_files_only=True,
            attn_implementation="sdpa",
        )
        .to(device)
        .eval()
    )
    model.config.use_cache = False
    layers = int(model.config.num_hidden_layers)
    kv_heads = int(model.config.num_key_value_heads)
    query_heads = int(model.config.num_attention_heads)
    head_dim = int(
        getattr(model.config, "head_dim", model.config.hidden_size // query_heads)
    )
    loki_factors, loki_result, loki_result_path, loki_tensor_path = _load_loki_factors(
        loki_factor_dir,
        model=model_path,
        layers=layers,
        kv_heads=kv_heads,
        head_dim=head_dim,
        rank=args.routing_rank,
    )

    results: dict[str, Any] = {}
    samples: dict[str, list[dict[str, Any]]] = {}
    if "dense" in arms:
        results["dense"], samples["dense"] = _evaluate_arm(
            "dense",
            model,
            stream,
            block_ranges,
            rank_index=rank_index,
            world_size=world_size,
            device=device,
            logit_block_size=args.logit_block_size,
        )

    installation_seconds = 0.0
    if any(arm in arms for arm in ("c1_exact", "c1_exact_topk", "loki")):
        installation_started = time.perf_counter()
        install_qwen3_gqa_vo_als_export(
            model,
            c1_checkpoint,
            attention_backend="triton",
        )
        installation_seconds = time.perf_counter() - installation_started
        modules = _attention_modules(model)
        if "c1_exact" in arms:
            results["c1_exact"], samples["c1_exact"] = _evaluate_arm(
                "c1_exact",
                model,
                stream,
                block_ranges,
                rank_index=rank_index,
                world_size=world_size,
                device=device,
                logit_block_size=args.logit_block_size,
            )
        if "c1_exact_topk" in arms or "loki" in arms:

            def set_loki_budget(sequence_length: int) -> None:
                budget = max(1, int(args.top_k_ratio * sequence_length))
                for module in modules:
                    module.set_reverse_shadow_config(
                        ReverseShadowConfig(
                            page_size=1,
                            exact_token_budget=budget,
                            recent_exact_window=0,
                            selector="kq_svd",
                            landmark_dtype="bfloat16",
                            quest_support="per_query_head",
                        )
                    )

            for module in modules:
                module.attention_backend = "native"
                module.set_loki_query_block_size(
                    args.query_block_size,
                    collect_statistics=False,
                )
        if "c1_exact_topk" in arms:
            for module in modules:
                identity = torch.eye(
                    module.head_dim,
                    dtype=module.q_proj.weight.dtype,
                    device=module.q_proj.weight.device,
                ).expand(module.num_key_value_heads, -1, -1)
                module.set_routing_projectors(identity, identity)
            set_loki_budget(args.sequence_length)
            results["c1_exact_topk"], samples["c1_exact_topk"] = _evaluate_arm(
                "c1_exact_topk",
                model,
                stream,
                block_ranges,
                rank_index=rank_index,
                world_size=world_size,
                device=device,
                logit_block_size=args.logit_block_size,
                before_block=set_loki_budget,
            )
        if "loki" in arms:
            for layer, module in enumerate(modules):
                module.set_routing_projectors(
                    loki_factors[layer],
                    loki_factors[layer],
                )
            set_loki_budget(args.sequence_length)
            results["loki"], samples["loki"] = _evaluate_arm(
                "loki",
                model,
                stream,
                block_ranges,
                rank_index=rank_index,
                world_size=world_size,
                device=device,
                logit_block_size=args.logit_block_size,
                before_block=set_loki_budget,
            )

    comparisons = {}
    if "dense" in results:
        for arm in arms:
            if arm != "dense":
                comparisons[f"{arm}_vs_dense"] = _comparison(
                    results[arm], results["dense"]
                )
    if "c1_exact" in results:
        for arm in ("c1_exact_topk", "loki"):
            if arm in results:
                comparisons[f"{arm}_vs_c1_exact"] = _comparison(
                    results[arm], results["c1_exact"]
                )
    if "c1_exact_topk" in results and "loki" in results:
        comparisons["loki_vs_c1_exact_topk"] = _comparison(
            results["loki"], results["c1_exact_topk"]
        )

    if rank_index == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": FORMAT,
            "status": "complete",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "command": shlex.join(sys.argv),
            "model": {
                "path": str(model_path),
                "config_sha256": _sha256(model_path / "config.json"),
                "dtype": "bfloat16",
            },
            "dataset": dataset,
            "protocol": {
                "arms": list(arms),
                "sequence_length": args.sequence_length,
                "blocks": len(block_ranges),
                "covered_input_tokens": sum(
                    stop - start for start, stop in block_ranges
                ),
                "complete_corpus": block_ranges[-1][1] == int(stream.numel()),
                "routing_rank": args.routing_rank,
                "top_k_ratio": args.top_k_ratio,
                "top_k_per_query_head_at_full_length": int(
                    args.top_k_ratio * args.sequence_length
                ),
                "query_block_size": args.query_block_size,
                "logit_block_size": args.logit_block_size,
                "context_reset_per_block": True,
                "include_short_final_block": True,
                "repository_ppl_aggregation": "equal mean of per-block mean NLL",
                "additional_ppl_aggregation": "global token-weighted mean NLL",
                "loki_semantics": {
                    "pca_fit_coordinate": "pre_rope_after_key_rmsnorm",
                    "runtime_projection_coordinate": "post_rope",
                    "selection_support": "independent_per_query_head",
                    "selection_granularity": "token",
                    "candidate_score": "rank-32 projected QK",
                    "selected_score": "exact rank-128 QK",
                    "softmax_support": "selected tokens only",
                    "value_payload": "C1-V80 ALS5",
                    "query_axis_tiling": "mathematically exact memory optimization",
                },
                "c1_exact_topk_semantics": {
                    "candidate_score": "exact rank-128 QK",
                    "selection_support": "independent_per_query_head",
                    "selection_granularity": "token",
                    "selected_score": "exact rank-128 QK",
                    "softmax_support": "selected tokens only",
                    "value_payload": "C1-V80 ALS5",
                },
            },
            "loki": {
                "repository": "https://github.com/hpcgroup/loki",
                "repository_commit": LOKI_COMMIT,
                "factor_manifest": str(loki_result_path),
                "factor_manifest_sha256": _sha256(loki_result_path),
                "factor_tensor": str(loki_tensor_path),
                "factor_tensor_sha256": _sha256(loki_tensor_path),
                "calibration": loki_result["calibration"],
            },
            "c1": {
                "manifest": str(c1_result_path),
                "manifest_sha256": _sha256(c1_result_path),
                "format": c1_result["format"],
                "value_rank": value_rank,
                "installation_seconds_per_rank": installation_seconds,
            },
            "results": results,
            "comparisons": comparisons,
            "samples": samples,
            "environment": {
                "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
                "python": sys.version,
                "torch": torch.__version__,
                "transformers": __import__("transformers").__version__,
                "world_size": world_size,
                "gpu": torch.cuda.get_device_name(device),
                "torch_num_threads": torch.get_num_threads(),
            },
            "elapsed_seconds": time.perf_counter() - started,
        }
        _atomic_json(output_dir / "result.json", payload)
        _atomic_text(output_dir / "summary.md", _markdown(payload))
        print(
            f"[Loki PPL] result={output_dir / 'result.json'}",
            flush=True,
        )
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
