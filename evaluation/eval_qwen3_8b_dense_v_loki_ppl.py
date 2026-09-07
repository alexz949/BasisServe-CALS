#!/usr/bin/env python3
"""Full WikiText-2 PPL for exact Top-K and Loki with dense Values."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import shlex
import sys
import time
from typing import Any

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basisserve.checkpoint.gqa_vo_qwen3 import (  # noqa: E402
    GQATiedVOQwen3Attention,
)
from basisserve.core.c1_k_reverse_shadow import ReverseShadowConfig  # noqa: E402
from eval_qwen3_8b_c1_loki_ppl import (  # noqa: E402
    LOKI_COMMIT,
    _atomic_json,
    _atomic_text,
    _comparison,
    _evaluate_arm,
    _load_loki_factors,
    _sha256,
    _wiki_stream,
)


FORMAT = "basisserve.qwen3_8b.dense_v_loki_wikitext_ppl.v1"
ARMS = ("dense", "exact_topk", "loki")


def _parse_arms(value: str) -> tuple[str, ...]:
    arms = tuple(item.strip() for item in value.split(",") if item.strip())
    assert arms and len(arms) == len(set(arms))
    assert all(arm in ARMS for arm in arms)
    assert tuple(arm for arm in ARMS if arm in arms) == arms
    return arms


def _install_dense_value_router(
    model: torch.nn.Module,
) -> list[GQATiedVOQwen3Attention]:
    modules = []
    for layer in model.model.layers:
        base = layer.self_attn
        replacement = GQATiedVOQwen3Attention(
            base,
            v_proj_compressed_weight=base.v_proj.weight.detach(),
            o_decoder_weight=base.o_proj.weight.detach(),
            v_proj_compressed_bias=(
                None if base.v_proj.bias is None else base.v_proj.bias.detach()
            ),
            o_decoder_bias=(
                None if base.o_proj.bias is None else base.o_proj.bias.detach()
            ),
            attention_backend="sdpa",
        )
        replacement.attention_backend = "native"
        layer.self_attn = replacement
        modules.append(replacement)
    return modules


def _configure_selector(
    modules: list[GQATiedVOQwen3Attention],
    projectors: torch.Tensor | None,
    *,
    exact_token_budget: int,
    query_block_size: int,
) -> None:
    for layer, module in enumerate(modules):
        if projectors is None:
            identity = torch.eye(
                module.head_dim,
                dtype=module.q_proj.weight.dtype,
                device=module.q_proj.weight.device,
            ).expand(module.num_key_value_heads, -1, -1)
            module.set_routing_projectors(identity, identity)
        else:
            module.set_routing_projectors(projectors[layer], projectors[layer])
        module.set_loki_query_block_size(
            query_block_size,
            collect_statistics=False,
        )
        module.set_reverse_shadow_config(
            ReverseShadowConfig(
                page_size=1,
                exact_token_budget=exact_token_budget,
                recent_exact_window=0,
                selector="kq_svd",
                landmark_dtype="bfloat16",
                quest_support="per_query_head",
            )
        )


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Qwen3-8B Dense-V Exact Top-K and Loki WikiText-2 PPL",
        "",
        "All arms use the original dense rank-128 Value projection and dense output "
        "projection. Exact Top-K ranks tokens with full rank-128 QK; Loki ranks "
        "tokens with rank-32 projected QK. Both independently select 25% of tokens "
        "per Query head, recompute exact rank-128 QK on the selected support, and "
        "apply selected-token softmax to dense Values.",
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
            "`Repository PPL` reproduces Loki's equal weighting of non-overlapping "
            "sequence blocks, including the shorter final block. `Token-weighted "
            "PPL` weights every predicted token equally. Context resets at every "
            "block boundary.",
        ]
    )
    return "\n".join(lines) + "\n"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
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
    loki_factor_dir = args.loki_factors.expanduser().resolve()
    dataset_cache = args.dataset_cache.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
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
    loki_factors, loki_result, loki_result_path, loki_tensor_path = (
        _load_loki_factors(
            loki_factor_dir,
            model=model_path,
            layers=layers,
            kv_heads=kv_heads,
            head_dim=head_dim,
            rank=args.routing_rank,
        )
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

    installation_started = time.perf_counter()
    modules = _install_dense_value_router(model)
    installation_seconds = time.perf_counter() - installation_started

    def budget(sequence_length: int) -> int:
        return max(1, int(args.top_k_ratio * sequence_length))

    for arm, projectors in (("exact_topk", None), ("loki", loki_factors)):
        if arm not in arms:
            continue

        def configure(sequence_length: int, factors=projectors) -> None:
            _configure_selector(
                modules,
                factors,
                exact_token_budget=budget(sequence_length),
                query_block_size=args.query_block_size,
            )

        configure(args.sequence_length)
        results[arm], samples[arm] = _evaluate_arm(
            arm,
            model,
            stream,
            block_ranges,
            rank_index=rank_index,
            world_size=world_size,
            device=device,
            logit_block_size=args.logit_block_size,
            before_block=configure,
        )

    comparisons = {}
    if "dense" in results:
        for arm in arms:
            if arm != "dense":
                comparisons[f"{arm}_vs_dense"] = _comparison(
                    results[arm], results["dense"]
                )
    if "exact_topk" in results and "loki" in results:
        comparisons["loki_vs_exact_topk"] = _comparison(
            results["loki"], results["exact_topk"]
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
                "top_k_per_query_head_at_full_length": budget(
                    args.sequence_length
                ),
                "query_block_size": args.query_block_size,
                "logit_block_size": args.logit_block_size,
                "context_reset_per_block": True,
                "include_short_final_block": True,
                "selection_support": "independent_per_query_head",
                "selection_granularity": "token",
                "selected_score": "exact rank-128 QK",
                "softmax_support": "selected tokens only",
                "value_payload": "original dense rank-128 V",
                "arms_definition": {
                    "dense": "full dense QK softmax V",
                    "exact_topk": "rank-128 QK candidate ranking",
                    "loki": "rank-32 projected QK candidate ranking",
                },
                "repository_ppl_aggregation": (
                    "equal mean of per-block mean NLL"
                ),
                "additional_ppl_aggregation": "global token-weighted mean NLL",
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
                "dense_router_installation_seconds_per_rank": (
                    installation_seconds
                ),
            },
            "elapsed_seconds": time.perf_counter() - started,
        }
        _atomic_json(output_dir / "result.json", payload)
        _atomic_text(output_dir / "summary.md", _markdown(payload))
        print(f"[Dense-V Loki PPL] result={output_dir / 'result.json'}", flush=True)
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
