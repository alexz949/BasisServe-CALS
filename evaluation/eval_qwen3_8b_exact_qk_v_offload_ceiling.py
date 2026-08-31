#!/usr/bin/env python3
"""Evaluate exact-QK token/page selection ceilings for Qwen3 Value offload."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shlex
import sys
import time
from typing import Any

from safetensors.torch import load_file
import torch
from torch import Tensor
from torch.nn import functional as F
from transformers import AutoConfig, AutoModelForCausalLM


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basisserve.core.exact_qk_v_offload import (  # noqa: E402
    full_gqa_value_attention,
    gqa_union_page_mass_mask,
    gqa_union_token_topk_mask,
    sparse_gqa_value_attention,
)
from evaluation import eval_qwen3_32b_c1_wikitext as c1_evaluator  # noqa: E402
from evaluation.eval_qwen3_8b_dense_v_k_proxy import (  # noqa: E402
    _extract_layer_features,
)
from evaluation.fit_qwen3_8b_c1_k_output_closure import (  # noqa: E402
    HEAD_DIM,
    HEADS_PER_GROUP,
    HIDDEN_SIZE,
    NUM_KV_HEADS,
    NUM_QUERY_HEADS,
    _batches,
    _load_layer_c1_factors,
    _propagate_dense_layer,
    _sha256,
    _validate_config,
)


FORMAT = "basisserve.qwen3_8b.exact_qk_v_offload_ceiling.v1"


def _parse_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _new_sums() -> dict[str, float | int]:
    return {
        "query_rows": 0,
        "query_heads": 0,
        "selected_teacher_mass_sum": 0.0,
        "dense_head_error": 0.0,
        "dense_head_energy": 0.0,
        "dense_decoded_error": 0.0,
        "dense_decoded_energy": 0.0,
        "dense_decoded_cosine_sum": 0.0,
        "c1_head_error": 0.0,
        "c1_head_energy": 0.0,
        "c1_decoded_error": 0.0,
        "c1_decoded_energy": 0.0,
        "c1_decoded_cosine_sum": 0.0,
        "c1_sparse_to_dense_error": 0.0,
        "c1_sparse_to_dense_energy": 0.0,
        "union_tokens": 0.0,
        "visible_tokens": 0.0,
        "union_pages": 0.0,
        "visible_pages": 0.0,
        "nominal_group_tokens": 0.0,
        "nominal_group_pages": 0.0,
        "logical_dense_v_token_bytes": 0.0,
        "logical_dense_v_page_bytes": 0.0,
        "logical_c1_v_token_bytes": 0.0,
        "logical_c1_v_page_bytes": 0.0,
    }


def _new_baseline_sums() -> dict[str, float | int]:
    return {
        "query_rows": 0,
        "c1_full_to_dense_error": 0.0,
        "dense_energy": 0.0,
        "cosine_sum": 0.0,
    }


def _square_sum(value: Tensor) -> float:
    return float(value.detach().double().square().sum())


def _cosine(reference: Tensor, candidate: Tensor) -> float:
    return float(
        F.cosine_similarity(
            reference.detach().float().reshape(1, -1),
            candidate.detach().float().reshape(1, -1),
        )[0]
    )


def _decode_dense_heads(
    weight: Tensor,
    bias: Tensor | None,
    head_output: Tensor,
) -> Tensor:
    return F.linear(
        head_output.reshape(1, NUM_QUERY_HEADS * HEAD_DIM),
        weight,
        bias,
    )[0]


def _decode_c1_heads(head_output: Tensor, decoder: Tensor) -> Tensor:
    grouped = head_output.reshape(NUM_KV_HEADS, HEADS_PER_GROUP, -1)
    return torch.einsum("ghr,ghro->o", grouped.float(), decoder.float())


def _page_mask_from_token_mask(token_mask: Tensor, *, page_size: int) -> Tensor:
    visible = int(token_mask.shape[1])
    pages = math.ceil(visible / page_size)
    padded = F.pad(token_mask, (0, pages * page_size - visible), value=False)
    return padded.reshape(int(token_mask.shape[0]), pages, page_size).any(dim=-1)


def _accumulate_baseline(
    sums: dict[str, float | int],
    *,
    dense_reference: Tensor,
    c1_reference: Tensor,
) -> None:
    difference = c1_reference - dense_reference
    sums["query_rows"] += 1
    sums["c1_full_to_dense_error"] += _square_sum(difference)
    sums["dense_energy"] += _square_sum(dense_reference)
    sums["cosine_sum"] += _cosine(dense_reference, c1_reference)


def _accumulate_policy(
    sums: dict[str, float | int],
    *,
    selected_mass: Tensor,
    dense_reference_heads: Tensor,
    dense_candidate_heads: Tensor,
    dense_reference_decoded: Tensor,
    dense_candidate_decoded: Tensor,
    c1_reference_heads: Tensor,
    c1_candidate_heads: Tensor,
    c1_reference_decoded: Tensor,
    c1_candidate_decoded: Tensor,
    union_token_mask: Tensor,
    union_page_mask: Tensor,
    nominal_tokens_per_kv_head: int,
    nominal_pages_per_kv_head: int,
    visible: int,
    page_size: int,
) -> None:
    dense_head_difference = dense_candidate_heads - dense_reference_heads
    dense_decoded_difference = dense_candidate_decoded - dense_reference_decoded
    c1_head_difference = c1_candidate_heads - c1_reference_heads
    c1_decoded_difference = c1_candidate_decoded - c1_reference_decoded
    c1_sparse_to_dense = c1_candidate_decoded - dense_reference_decoded
    pages = math.ceil(visible / page_size)
    sums["query_rows"] += 1
    sums["query_heads"] += int(selected_mass.numel())
    sums["selected_teacher_mass_sum"] += float(selected_mass.double().sum())
    sums["dense_head_error"] += _square_sum(dense_head_difference)
    sums["dense_head_energy"] += _square_sum(dense_reference_heads)
    sums["dense_decoded_error"] += _square_sum(dense_decoded_difference)
    sums["dense_decoded_energy"] += _square_sum(dense_reference_decoded)
    sums["dense_decoded_cosine_sum"] += _cosine(
        dense_reference_decoded, dense_candidate_decoded
    )
    sums["c1_head_error"] += _square_sum(c1_head_difference)
    sums["c1_head_energy"] += _square_sum(c1_reference_heads)
    sums["c1_decoded_error"] += _square_sum(c1_decoded_difference)
    sums["c1_decoded_energy"] += _square_sum(c1_reference_decoded)
    sums["c1_decoded_cosine_sum"] += _cosine(
        c1_reference_decoded, c1_candidate_decoded
    )
    sums["c1_sparse_to_dense_error"] += _square_sum(c1_sparse_to_dense)
    sums["c1_sparse_to_dense_energy"] += _square_sum(dense_reference_decoded)
    sums["union_tokens"] += int(union_token_mask.sum())
    sums["visible_tokens"] += NUM_KV_HEADS * visible
    sums["union_pages"] += int(union_page_mask.sum())
    sums["visible_pages"] += NUM_KV_HEADS * pages
    sums["nominal_group_tokens"] += NUM_KV_HEADS * nominal_tokens_per_kv_head
    sums["nominal_group_pages"] += NUM_KV_HEADS * nominal_pages_per_kv_head
    selected_tokens = int(union_token_mask.sum())
    selected_page_tokens = int(union_page_mask.sum()) * page_size
    dense_dim = int(dense_reference_heads.shape[-1])
    c1_dim = int(c1_reference_heads.shape[-1])
    sums["logical_dense_v_token_bytes"] += selected_tokens * dense_dim * 2
    sums["logical_dense_v_page_bytes"] += selected_page_tokens * dense_dim * 2
    sums["logical_c1_v_token_bytes"] += selected_tokens * c1_dim * 2
    sums["logical_c1_v_page_bytes"] += selected_page_tokens * c1_dim * 2


def _merge_sums(rows: list[dict[str, float | int]]) -> dict[str, float | int]:
    merged = _new_sums()
    for row in rows:
        for key in merged:
            merged[key] += row[key]
    return merged


def _merge_baseline_sums(
    rows: list[dict[str, float | int]],
) -> dict[str, float | int]:
    merged = _new_baseline_sums()
    for row in rows:
        for key in merged:
            merged[key] += row[key]
    return merged


def _safe_ratio(numerator: float | int, denominator: float | int) -> float:
    return float(numerator) / max(float(denominator), 1.0e-300)


def _finalize(sums: dict[str, float | int]) -> dict[str, Any]:
    query_rows = int(sums["query_rows"])
    query_heads = int(sums["query_heads"])
    return {
        **sums,
        "mean_selected_teacher_mass": float(sums["selected_teacher_mass_sum"])
        / query_heads,
        "dense_head_relative_l2": math.sqrt(
            _safe_ratio(sums["dense_head_error"], sums["dense_head_energy"])
        ),
        "dense_decoded_relative_l2": math.sqrt(
            _safe_ratio(sums["dense_decoded_error"], sums["dense_decoded_energy"])
        ),
        "dense_decoded_mean_cosine": float(sums["dense_decoded_cosine_sum"])
        / query_rows,
        "c1_head_relative_l2": math.sqrt(
            _safe_ratio(sums["c1_head_error"], sums["c1_head_energy"])
        ),
        "c1_decoded_relative_l2": math.sqrt(
            _safe_ratio(sums["c1_decoded_error"], sums["c1_decoded_energy"])
        ),
        "c1_decoded_mean_cosine": float(sums["c1_decoded_cosine_sum"])
        / query_rows,
        "c1_sparse_to_dense_relative_l2": math.sqrt(
            _safe_ratio(
                sums["c1_sparse_to_dense_error"],
                sums["c1_sparse_to_dense_energy"],
            )
        ),
        "union_visible_token_fraction": _safe_ratio(
            sums["union_tokens"], sums["visible_tokens"]
        ),
        "page_rounded_visible_fraction": _safe_ratio(
            sums["union_pages"], sums["visible_pages"]
        ),
        "gqa_token_union_amplification": _safe_ratio(
            sums["union_tokens"], sums["nominal_group_tokens"]
        ),
        "gqa_page_union_amplification": _safe_ratio(
            sums["union_pages"], sums["nominal_group_pages"]
        ),
    }


def _finalize_baseline(sums: dict[str, float | int]) -> dict[str, Any]:
    return {
        **sums,
        "c1_full_to_dense_relative_l2": math.sqrt(
            _safe_ratio(sums["c1_full_to_dense_error"], sums["dense_energy"])
        ),
        "mean_cosine": float(sums["cosine_sum"]) / int(sums["query_rows"]),
    }


def _aggregate(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(str(record["selector"]), int(record["nominal_token_budget"]))].append(
            record
        )
    result = []
    for (selector, budget), rows in sorted(grouped.items()):
        result.append(
            {
                "selector": selector,
                "nominal_token_budget": budget,
                "layers": sorted(int(row["layer"]) for row in rows),
                "metrics": _finalize(
                    _merge_sums([row["metric_sums"] for row in rows])
                ),
            }
        )
    return result


def _aggregate_baselines(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return _finalize_baseline(
        _merge_baseline_sums([row["metric_sums"] for row in rows])
    )


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Qwen3-8B exact-QK Value-offload ceiling",
        "",
        "Exact post-RoPE K is treated as GPU-resident. Selection is unioned "
        "across the four query heads in each GQA group, and all heads attend "
        "over the fetched union with exact QK scores. Reported traffic excludes K.",
        "",
        "## Aggregate",
        "",
        "| selector | nominal tokens/head | mass | dense-V decoded rel-L2 | "
        "dense cosine | C1-V decoded rel-L2 | C1 sparse→dense rel-L2 | "
        "unique-token fraction | page-rounded fraction | token union amp | page union amp |",
        "|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["aggregate"]:
        metric = row["metrics"]
        lines.append(
            f"| {row['selector']} | {row['nominal_token_budget']} | "
            f"{metric['mean_selected_teacher_mass']:.6f} | "
            f"{metric['dense_decoded_relative_l2']:.6e} | "
            f"{metric['dense_decoded_mean_cosine']:.6f} | "
            f"{metric['c1_decoded_relative_l2']:.6e} | "
            f"{metric['c1_sparse_to_dense_relative_l2']:.6e} | "
            f"{metric['union_visible_token_fraction']:.6f} | "
            f"{metric['page_rounded_visible_fraction']:.6f} | "
            f"{metric['gqa_token_union_amplification']:.6f} | "
            f"{metric['gqa_page_union_amplification']:.6f} |"
        )
    baseline = payload["aggregate_c1_full_baseline"]
    lines.extend(
        [
            "",
            "## Full-attention payload control",
            "",
            f"C1-V64 full attention versus dense-V full attention after decoding: "
            f"relative L2 `{baseline['c1_full_to_dense_relative_l2']:.6e}`, "
            f"mean cosine `{baseline['mean_cosine']:.6f}`.",
            "",
            "`token_topk` is the scatter-gather quality ceiling. `page_mass` "
            "ranks each page by exact QK log-sum-exp and is the page-granular "
            "offload ceiling. Logical bytes assume BF16 Value vectors and do not "
            "measure PCIe latency.",
            "",
        ]
    )
    return "\n".join(lines)


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("exact-QK V-offload evaluation requires CUDA")
    layers = _parse_ints(args.layers)
    budgets = _parse_ints(args.budgets)
    if not layers or not budgets or min(budgets) <= 0:
        raise ValueError("layers and positive budgets must be nonempty")
    if min(
        args.heldout_windows,
        args.sequence_length,
        args.query_stride,
        args.page_size,
        args.batch_size,
        args.torch_num_threads,
    ) <= 0:
        raise ValueError("evaluation sizes must be positive")
    if args.query_start < max(budgets) - 1:
        raise ValueError("query start must expose the maximum nominal budget")

    started = time.perf_counter()
    torch.set_num_threads(args.torch_num_threads)
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats(0)
    model_path = args.model.expanduser().resolve()
    windows_path = args.windows.expanduser().resolve()
    c1_dir = args.c1_export.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)

    config = AutoConfig.from_pretrained(str(model_path), local_files_only=True)
    _validate_config(config)
    if any(layer < 0 or layer >= int(config.num_hidden_layers) for layer in layers):
        raise ValueError("requested layer is outside the model")
    window_manifest_path = windows_path.parent / "manifest.json"
    window_manifest = json.loads(window_manifest_path.read_text(encoding="utf-8"))
    if window_manifest["artifact"]["sha256"] != _sha256(windows_path):
        raise ValueError("window bank hash mismatch")
    if window_manifest["model"]["config_sha256"] != _sha256(model_path / "config.json"):
        raise ValueError("window bank belongs to another model")
    stored = load_file(str(windows_path), device="cpu")["input_ids"]
    stop = args.heldout_start + args.heldout_windows
    if (
        stored.ndim != 2
        or int(stored.shape[1]) != args.sequence_length
        or stop > len(stored)
    ):
        raise ValueError("window bank does not cover the requested held-out slice")
    windows = stored[args.heldout_start:stop]
    del stored

    c1_evaluator.activate_model_profile("qwen3_8b")
    c1_result = c1_evaluator._load_results(c1_dir, model_path)
    value_rank = int(c1_result["fit_config"]["cache_rank_per_head"])
    dtype = torch.bfloat16 if args.model_dtype == "bfloat16" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        dtype=dtype,
        low_cpu_mem_usage=True,
        local_files_only=True,
        attn_implementation="sdpa",
        device_map={"": 0},
    ).eval()
    model.config.use_cache = False
    device = model.model.embed_tokens.weight.device
    position_ids = torch.arange(
        args.sequence_length, device=device, dtype=torch.long
    ).unsqueeze(0)
    hidden_bank = torch.empty(
        len(windows),
        args.sequence_length,
        HIDDEN_SIZE,
        dtype=dtype,
        device="cpu",
    )
    for completed, input_ids in _batches(windows, 0, len(windows), args.batch_size):
        embeddings = model.model.embed_tokens(input_ids.to(device=device, dtype=torch.long))
        start = completed - len(input_ids)
        hidden_bank[start:completed].copy_(embeddings.to(device="cpu"))
    del windows, embeddings
    position_embeddings = model.model.rotary_emb(
        hidden_bank[:1].to(device=device), position_ids
    )
    query_positions = list(
        range(args.query_start, args.sequence_length, args.query_stride)
    )
    if query_positions[-1] != args.sequence_length - 1:
        query_positions.append(args.sequence_length - 1)

    records = []
    baseline_records = []
    requested_layers = set(layers)
    for layer_index, layer in enumerate(model.model.layers):
        if layer_index in requested_layers:
            encoder, decoder, _ = _load_layer_c1_factors(
                c1_dir, c1_result, layer_index, value_rank
            )
            features = _extract_layer_features(
                layer,
                hidden_bank,
                value_encoder=encoder,
                position_embeddings=position_embeddings,
                batch_size=args.batch_size,
            )
            decoder = decoder.to(device=device, dtype=torch.float32)
            o_proj_weight = layer.self_attn.o_proj.weight.detach().float()
            o_proj_bias = (
                None
                if layer.self_attn.o_proj.bias is None
                else layer.self_attn.o_proj.bias.detach().float()
            )
            policy_sums = {
                (selector, budget): _new_sums()
                for selector in ("token_topk", "page_mass")
                for budget in budgets
            }
            baseline_sums = _new_baseline_sums()
            kv_head_index = torch.arange(NUM_QUERY_HEADS, device=device) // HEADS_PER_GROUP
            for example in range(args.heldout_windows):
                query = features["query"][example].to(device=device, dtype=torch.float32)
                key = features["post_key"][example].to(device=device, dtype=torch.float32)
                dense_value = features["dense_value"][example].to(
                    device=device, dtype=torch.float32
                )
                c1_value = features["c1_value"][example].to(
                    device=device, dtype=torch.float32
                )
                for position in query_positions:
                    visible = position + 1
                    repeated_key = key.index_select(0, kv_head_index)[:, :visible]
                    scores = torch.einsum(
                        "hd,hld->hl", query[:, position], repeated_key
                    ) / math.sqrt(HEAD_DIM)
                    dense_reference_heads = full_gqa_value_attention(
                        scores,
                        dense_value[:, :visible],
                        heads_per_group=HEADS_PER_GROUP,
                    )
                    c1_reference_heads = full_gqa_value_attention(
                        scores,
                        c1_value[:, :visible],
                        heads_per_group=HEADS_PER_GROUP,
                    )
                    dense_reference_decoded = _decode_dense_heads(
                        o_proj_weight, o_proj_bias, dense_reference_heads
                    )
                    c1_reference_decoded = _decode_c1_heads(
                        c1_reference_heads, decoder
                    )
                    _accumulate_baseline(
                        baseline_sums,
                        dense_reference=dense_reference_decoded,
                        c1_reference=c1_reference_decoded,
                    )
                    for budget in budgets:
                        nominal_tokens = min(budget, visible)
                        nominal_pages = min(
                            math.ceil(budget / args.page_size),
                            math.ceil(visible / args.page_size),
                        )
                        token_mask = gqa_union_token_topk_mask(
                            scores,
                            num_kv_heads=NUM_KV_HEADS,
                            top_k=budget,
                        )
                        token_page_mask = _page_mask_from_token_mask(
                            token_mask, page_size=args.page_size
                        )
                        page_token_mask, page_mask = gqa_union_page_mass_mask(
                            scores,
                            num_kv_heads=NUM_KV_HEADS,
                            page_size=args.page_size,
                            pages_per_query_head=math.ceil(budget / args.page_size),
                        )
                        for selector, union_token_mask, union_page_mask in (
                            ("token_topk", token_mask, token_page_mask),
                            ("page_mass", page_token_mask, page_mask),
                        ):
                            dense_candidate = sparse_gqa_value_attention(
                                scores,
                                dense_value[:, :visible],
                                union_token_mask,
                                heads_per_group=HEADS_PER_GROUP,
                            )
                            c1_candidate = sparse_gqa_value_attention(
                                scores,
                                c1_value[:, :visible],
                                union_token_mask,
                                heads_per_group=HEADS_PER_GROUP,
                            )
                            dense_candidate_decoded = _decode_dense_heads(
                                o_proj_weight, o_proj_bias, dense_candidate.output
                            )
                            c1_candidate_decoded = _decode_c1_heads(
                                c1_candidate.output, decoder
                            )
                            _accumulate_policy(
                                policy_sums[(selector, budget)],
                                selected_mass=dense_candidate.selected_teacher_mass,
                                dense_reference_heads=dense_reference_heads,
                                dense_candidate_heads=dense_candidate.output,
                                dense_reference_decoded=dense_reference_decoded,
                                dense_candidate_decoded=dense_candidate_decoded,
                                c1_reference_heads=c1_reference_heads,
                                c1_candidate_heads=c1_candidate.output,
                                c1_reference_decoded=c1_reference_decoded,
                                c1_candidate_decoded=c1_candidate_decoded,
                                union_token_mask=union_token_mask,
                                union_page_mask=union_page_mask,
                                nominal_tokens_per_kv_head=nominal_tokens,
                                nominal_pages_per_kv_head=nominal_pages,
                                visible=visible,
                                page_size=args.page_size,
                            )
                print(
                    f"[exact-QK V offload] layer={layer_index} heldout="
                    f"{example + 1}/{args.heldout_windows}",
                    flush=True,
                )
            for (selector, budget), sums in policy_sums.items():
                records.append(
                    {
                        "layer": layer_index,
                        "selector": selector,
                        "nominal_token_budget": budget,
                        "page_size": args.page_size,
                        "metric_sums": sums,
                        "metrics": _finalize(sums),
                    }
                )
            baseline_records.append(
                {
                    "layer": layer_index,
                    "metric_sums": baseline_sums,
                    "metrics": _finalize_baseline(baseline_sums),
                }
            )
            del (
                features,
                encoder,
                decoder,
                o_proj_weight,
                o_proj_bias,
                policy_sums,
                baseline_sums,
            )
            torch.cuda.empty_cache()

        _propagate_dense_layer(
            model,
            layer,
            hidden_bank,
            batch_size=args.batch_size,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            layer_index=layer_index,
        )

    c1_result_path = c1_dir / "results.json"
    payload = {
        "format": FORMAT,
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(sys.argv),
        "method": {
            "resident_routing": "full exact post-RoPE K and exact QK",
            "selectors": {
                "token_topk": "per-query-head exact token Top-k then GQA union",
                "page_mass": "per-query-head exact page log-sum-exp Top-m then GQA union",
            },
            "attention": "all query heads attend over the fetched GQA union with exact QK",
            "payloads": ["dense BF16 V128", f"C1 BF16 V{value_rank}"],
            "traffic_scope": "Value only; excludes resident K and selector compute",
        },
        "model": {
            "path": str(model_path),
            "config_sha256": _sha256(model_path / "config.json"),
        },
        "windows": {
            "path": str(windows_path),
            "sha256": _sha256(windows_path),
            "manifest_sha256": _sha256(window_manifest_path),
            "heldout_indices": list(range(args.heldout_start, stop)),
            "sequence_length": args.sequence_length,
            "query_positions": query_positions,
        },
        "c1_export": {
            "path": str(c1_dir),
            "results_sha256": _sha256(c1_result_path),
            "value_rank": value_rank,
        },
        "configuration": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "records": records,
        "aggregate": _aggregate(records),
        "c1_full_baselines": baseline_records,
        "aggregate_c1_full_baseline": _aggregate_baselines(baseline_records),
        "runtime": {
            "seconds": time.perf_counter() - started,
            "peak_cuda_bytes": torch.cuda.max_memory_allocated(0),
            "torch_version": torch.__version__,
        },
    }
    _atomic_text(output_dir / "result.json", json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _atomic_text(output_dir / "summary.md", _markdown(payload))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--c1-export", type=Path, required=True)
    parser.add_argument("--layers", default="0,17,35")
    parser.add_argument("--heldout-start", type=int, default=8)
    parser.add_argument("--heldout-windows", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--query-start", type=int, default=1023)
    parser.add_argument("--query-stride", type=int, default=256)
    parser.add_argument("--budgets", default="64,256,512,1024")
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--torch-num-threads", type=int, default=4)
    parser.add_argument("--model-dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


if __name__ == "__main__":
    evaluate(_parser().parse_args())
