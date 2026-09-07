#!/usr/bin/env python3
"""Full WikiText-2 PPL for Qwen3 C1-V80 conditional Page routing."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import sys
import time
from typing import Any

import torch
import torch.distributed as dist
import transformers
from transformers import AutoModelForCausalLM


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basisserve.checkpoint.gqa_vo_qwen3 import (  # noqa: E402
    install_qwen3_gqa_vo_als_export,
)
from basisserve.core.c1_k_reverse_shadow import (  # noqa: E402
    ReverseShadowConfig,
)
from evaluation.eval_qwen3_8b_c1_kq_routing_ppl import (  # noqa: E402
    _merge_runtime,
    _runtime_statistics,
)
from evaluation.eval_qwen3_8b_c1_kq_routing_ruler import (  # noqa: E402
    _load_conditional_factors,
    _set_conditional_routing,
)
from evaluation.eval_qwen3_8b_c1_loki_ppl import (  # noqa: E402
    _atomic_json,
    _atomic_text,
    _attention_modules,
    _comparison,
    _evaluate_arm,
    _sha256,
    _wiki_stream,
)


FORMAT = "basisserve.qwen3_8b.c1_v80_conditional_page_wikitext_ppl.v1"
FACTOR_FORMAT = "basisserve.qwen3_8b.v80_base16_r8_nonsink_page32.v1"


def _parse_budgets(value: str) -> tuple[int, ...]:
    budgets = tuple(int(item) for item in value.split(",") if item)
    assert budgets and len(budgets) == len(set(budgets))
    assert all(budget > 0 for budget in budgets)
    return budgets


def _distributed_runtime_statistics(modules: list[torch.nn.Module]) -> dict[str, float]:
    local = _runtime_statistics(modules)
    if not dist.is_initialized():
        return local
    rows: list[dict[str, float] | None] = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(rows, local)
    return _merge_runtime([row for row in rows if row is not None])


def _markdown(payload: dict[str, Any]) -> str:
    protocol = payload["protocol"]
    lines = [
        "# Qwen3-8B C1-V80 Base16+R8 Page32 WikiText-2 PPL",
        "",
        "The conditional router reconstructs a rank-16 pre-RoPE Key base from the "
        "resident V80 code, adds an independently stored rank-8 residual-Key code, "
        "forms Page32 log-mass, and selects one fixed physical page set after a "
        "max across the four Query heads sharing each GQA Key head. The first Page32 "
        "is pinned within each fixed budget.",
        "",
        "| Arm | Repository PPL | Token-weighted PPL | vs Dense | vs C1 exact | "
        "Physical selected fraction | Wall time (s) | Peak GiB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in protocol["arms"]:
        row = payload["results"][arm]
        dense_ratio = payload["comparisons"].get(f"{arm}_vs_dense", {}).get(
            "repository_ppl_ratio"
        )
        c1_ratio = payload["comparisons"].get(f"{arm}_vs_c1_exact", {}).get(
            "repository_ppl_ratio"
        )
        runtime = payload["runtime_logical"].get(arm)
        dense_delta = (
            "--" if dense_ratio is None else f"{100 * (dense_ratio - 1):+.2f}%"
        )
        c1_delta = (
            "--" if c1_ratio is None else f"{100 * (c1_ratio - 1):+.2f}%"
        )
        selected_fraction = (
            "--"
            if runtime is None
            else f"{runtime['selected_token_fraction']:.6f}"
        )
        lines.append(
            f"| {arm} | {row['repository_ppl']:.8f} | "
            f"{row['token_weighted_ppl']:.8f} | "
            f"{dense_delta} | {c1_delta} | {selected_fraction} | "
            f"{row['wall_seconds']:.2f} | {row['maximum_allocated_gib']:.3f} |"
        )
    lines.extend(
        [
            "",
            f"The two routed arms use strict physical budgets B"
            f"{'/B'.join(str(value) for value in protocol['physical_token_budgets'])} "
            "per KV head. The deployable same-precision persistent KV scalar ratio is "
            f"`{protocol['persistent_gpu_scalar_ratio']:.6f}` versus dense KV.",
            "",
            "`Repository PPL` uses Loki's equal mean over non-overlapping blocks, "
            "including the shorter final block. `Token-weighted PPL` weights every "
            "predicted token equally. Context resets at each block boundary.",
        ]
    )
    return "\n".join(lines) + "\n"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--c1-checkpoint", type=Path, required=True)
    parser.add_argument("--routing-factors", type=Path, required=True)
    parser.add_argument(
        "--dataset-cache",
        type=Path,
        default=REPO_ROOT / "results/cache/huggingface/datasets",
    )
    parser.add_argument("--sequence-length", type=int, default=8192)
    parser.add_argument("--page-size", type=int, default=32)
    parser.add_argument("--physical-token-budgets", default="2048,4096")
    parser.add_argument("--base-rank", type=int, default=16)
    parser.add_argument("--residual-rank", type=int, default=8)
    parser.add_argument("--pinned-prefix-pages", type=int, default=1)
    parser.add_argument("--query-block-size", type=int, default=32)
    parser.add_argument("--logit-block-size", type=int, default=128)
    parser.add_argument("--max-blocks", type=int)
    parser.add_argument("--torch-num-threads", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


@torch.inference_mode()
def main() -> None:
    args = _parser().parse_args()
    budgets = _parse_budgets(args.physical_token_budgets)
    assert min(
        args.sequence_length,
        args.page_size,
        args.base_rank,
        args.residual_rank,
        args.query_block_size,
        args.logit_block_size,
        args.torch_num_threads,
    ) > 0
    assert all(budget % args.page_size == 0 for budget in budgets)
    assert 0 <= args.pinned_prefix_pages <= min(budgets) // args.page_size
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
    factor_root = args.routing_factors.expanduser().resolve()
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
    conditional_factors, factor_paths, factor_result_paths = (
        _load_conditional_factors(
            factor_root,
            layers=layers,
            base_rank=args.base_rank,
            residual_rank=args.residual_rank,
        )
    )
    factor_manifests = [
        json.loads(path.read_text(encoding="utf-8")) for path in factor_result_paths
    ]
    assert factor_manifests
    assert all(
        row["format"] == FACTOR_FORMAT and row["status"] == "complete"
        for row in factor_manifests
    )

    results: dict[str, Any] = {}
    samples: dict[str, list[dict[str, Any]]] = {}
    runtime_logical: dict[str, dict[str, float]] = {}
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
    install_qwen3_gqa_vo_als_export(
        model,
        c1_checkpoint,
        attention_backend="triton",
    )
    installation_seconds = time.perf_counter() - installation_started
    modules = _attention_modules(model)
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

    _set_conditional_routing(modules, conditional_factors)
    for module in modules:
        module.attention_backend = "native"
        module.set_conditional_page_query_block_size(
            args.query_block_size,
            collect_statistics=True,
        )
    for budget in budgets:
        arm = f"ours_b{budget}"
        config = ReverseShadowConfig(
            page_size=args.page_size,
            exact_token_budget=budget,
            recent_exact_window=0,
            selector="kq_svd",
            landmark_dtype="bfloat16",
            quest_support="physical_shared",
            query_head_aggregation="max_head",
            pinned_prefix_pages=args.pinned_prefix_pages,
        )
        for module in modules:
            module.set_reverse_shadow_config(config)
        results[arm], samples[arm] = _evaluate_arm(
            arm,
            model,
            stream,
            block_ranges,
            rank_index=rank_index,
            world_size=world_size,
            device=device,
            logit_block_size=args.logit_block_size,
        )
        runtime_logical[arm] = _distributed_runtime_statistics(modules)

    arms = ("dense", "c1_exact", *(f"ours_b{budget}" for budget in budgets))
    comparisons = {
        f"{arm}_vs_dense": _comparison(results[arm], results["dense"])
        for arm in arms
        if arm != "dense"
    }
    comparisons.update(
        {
            f"{arm}_vs_c1_exact": _comparison(results[arm], results["c1_exact"])
            for arm in arms
            if arm not in ("dense", "c1_exact")
        }
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
                "page_size": args.page_size,
                "physical_token_budgets": list(budgets),
                "selection_support": "fixed_physical_per_kv_head",
                "query_head_aggregation": "normalized_page_mass_group_max",
                "base_rank": args.base_rank,
                "residual_rank": args.residual_rank,
                "pinned_prefix_pages": args.pinned_prefix_pages,
                "query_block_size": args.query_block_size,
                "logit_block_size": args.logit_block_size,
                "context_reset_per_block": True,
                "include_short_final_block": True,
                "repository_ppl_aggregation": "equal mean of per-block mean NLL",
                "additional_ppl_aggregation": "global token-weighted mean NLL",
                "value_payload": "C1-V80 ALS5",
                "persistent_gpu_scalar_ratio": (
                    value_rank + args.residual_rank
                )
                / 256,
            },
            "routing": {
                "factor_root": str(factor_root),
                "factor_tensor_sha256": [
                    _sha256(path) for path in factor_paths
                ],
                "factor_manifest_sha256": [
                    _sha256(path) for path in factor_result_paths
                ],
                "factor_format": FACTOR_FORMAT,
                "calibration": "32K Page32 Fisher with non-sink routing",
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
            "runtime_logical": runtime_logical,
            "samples": samples,
            "environment": {
                "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
                "python": sys.version,
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "world_size": world_size,
                "gpu": torch.cuda.get_device_name(device),
                "torch_num_threads": torch.get_num_threads(),
            },
            "elapsed_seconds": time.perf_counter() - started,
        }
        _atomic_json(output_dir / "result.json", payload)
        _atomic_text(output_dir / "summary.md", _markdown(payload))
        print(
            f"[Conditional Page PPL] result={output_dir / 'result.json'}",
            flush=True,
        )
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
