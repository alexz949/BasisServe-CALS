#!/usr/bin/env python3
"""Evaluate C1-V64 + low-rank K-routing sidecars from 1K through 32K."""

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

from basisserve.core.c1_k_routing_sidecar import (  # noqa: E402
    build_routing_sidecar,
    routing_proxy_scores,
    routing_storage_ratio,
)
from basisserve.core.exact_qk_v_offload import (  # noqa: E402
    full_gqa_value_attention,
    gqa_union_page_mass_mask,
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


FORMAT = "basisserve.qwen3_8b.c1_k_routing_sidecar.length_sweep.v1"
WINDOW_FORMAT = "basisserve.qwen3_8b.c1_k_routing_sidecar.wikitext_windows.v1"
FACTOR_FORMAT = "basisserve.qwen3_8b.post_rope_kqsvd.v1"
PAIRWISE_FACTOR_FORMAT = "basisserve.qwen3_8b.pairwise_kq_svd.v1"


def _parse_ints(raw: str) -> list[int]:
    return [int(item) for item in raw.split(",") if item]


def _parse_strings(raw: str) -> list[str]:
    return [item for item in raw.split(",") if item]


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _square_sum(value: Tensor) -> float:
    return float(value.detach().double().square().sum())


def _safe_ratio(numerator: float | int, denominator: float | int) -> float:
    return float(numerator) / max(float(denominator), 1.0e-300)


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


def _new_baseline_sums() -> dict[str, float | int]:
    return {
        "layer_queries": 0,
        "c1_full_to_dense_error": 0.0,
        "dense_energy": 0.0,
    }


def _new_policy_sums() -> dict[str, float | int]:
    return {
        "layer_queries": 0,
        "query_heads": 0,
        "score_squared_error": 0.0,
        "score_centered_energy": 0.0,
        "attention_kl_sum": 0.0,
        "selected_teacher_mass_sum": 0.0,
        "oracle_teacher_mass_sum": 0.0,
        "page_intersection": 0,
        "oracle_pages": 0,
        "selected_pages": 0,
        "visible_pages": 0,
        "selected_tokens": 0,
        "visible_tokens": 0,
        "nominal_group_pages": 0,
        "logical_exact_k_bytes": 0,
        "c1_sparse_error": 0.0,
        "c1_full_energy": 0.0,
        "oracle_c1_sparse_error": 0.0,
        "c1_sparse_to_dense_error": 0.0,
        "dense_energy": 0.0,
    }


def _merge_sums(rows: list[dict[str, float | int]]) -> dict[str, float | int]:
    if not rows:
        raise ValueError("cannot merge an empty metric list")
    merged = {key: 0 for key in rows[0]}
    for row in rows:
        if row.keys() != merged.keys():
            raise ValueError("metric accumulators have different fields")
        for key, value in row.items():
            merged[key] += value
    return merged


def _finalize_baseline(sums: dict[str, float | int]) -> dict[str, Any]:
    return {
        **sums,
        "c1_full_to_dense_relative_l2": math.sqrt(
            _safe_ratio(sums["c1_full_to_dense_error"], sums["dense_energy"])
        ),
    }


def _finalize_policy(sums: dict[str, float | int]) -> dict[str, Any]:
    layer_queries = int(sums["layer_queries"])
    query_heads = int(sums["query_heads"])
    return {
        **sums,
        "centered_score_relative_rmse": math.sqrt(
            _safe_ratio(
                sums["score_squared_error"],
                sums["score_centered_energy"],
            )
        ),
        "mean_attention_kl_teacher_to_proxy": _safe_ratio(
            sums["attention_kl_sum"], query_heads
        ),
        "mean_selected_teacher_mass": _safe_ratio(
            sums["selected_teacher_mass_sum"], query_heads
        ),
        "mean_oracle_teacher_mass": _safe_ratio(
            sums["oracle_teacher_mass_sum"], query_heads
        ),
        "page_recall_vs_exact_qk_oracle": _safe_ratio(
            sums["page_intersection"], sums["oracle_pages"]
        ),
        "selected_page_fraction": _safe_ratio(
            sums["selected_pages"], sums["visible_pages"]
        ),
        "selected_token_fraction": _safe_ratio(
            sums["selected_tokens"], sums["visible_tokens"]
        ),
        "gqa_page_union_amplification": _safe_ratio(
            sums["selected_pages"], sums["nominal_group_pages"]
        ),
        "mean_logical_exact_k_bytes_per_layer_query": _safe_ratio(
            sums["logical_exact_k_bytes"], layer_queries
        ),
        "c1_sparse_to_full_relative_l2": math.sqrt(
            _safe_ratio(sums["c1_sparse_error"], sums["c1_full_energy"])
        ),
        "oracle_c1_sparse_to_full_relative_l2": math.sqrt(
            _safe_ratio(
                sums["oracle_c1_sparse_error"], sums["c1_full_energy"]
            )
        ),
        "c1_sparse_to_dense_relative_l2": math.sqrt(
            _safe_ratio(sums["c1_sparse_to_dense_error"], sums["dense_energy"])
        ),
    }


def _load_routing_factors(
    factor_dir: Path,
    *,
    model_path: Path,
    methods: list[str],
) -> tuple[dict[str, tuple[Tensor, Tensor]], dict[str, Any], Path]:
    result_path = factor_dir / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    factor_format = result.get("format")
    supported_formats = {FACTOR_FORMAT, PAIRWISE_FACTOR_FORMAT}
    if factor_format not in supported_formats or result.get("status") != "complete":
        raise ValueError("routing factor result is incomplete or incompatible")
    if result["model"]["config_sha256"] != _sha256(model_path / "config.json"):
        raise ValueError("routing factors belong to another model")
    factor_path = factor_dir / result["artifacts"]["factors"]["file"]
    if _sha256(factor_path) != result["artifacts"]["factors"]["sha256"]:
        raise ValueError("routing factor hash mismatch")
    tensors = load_file(str(factor_path), device="cpu")
    allowed = {"key_svd", "kq_svd"}
    if not set(methods) <= allowed:
        raise ValueError(f"routing methods must be drawn from {sorted(allowed)}")
    if factor_format == PAIRWISE_FACTOR_FORMAT and set(methods) != {"kq_svd"}:
        raise ValueError(
            "pairwise KQ-SVD exports provide only independent asymmetric "
            "KQ-SVD routing factors; use --methods kq_svd"
        )
    factors = {}
    for method in methods:
        if factor_format == PAIRWISE_FACTOR_FORMAT:
            key = tensors["independent_key_projector"]
            query = tensors["independent_query_projector"]
        elif method == "kq_svd":
            key = tensors["kq_svd_key_projector"]
            query = tensors["kq_svd_query_projector"]
        else:
            key = tensors["key_svd_projector"]
            query = key
        factors[method] = (key.contiguous(), query.contiguous())
    return factors, result, result_path


def _score_metrics(exact_scores: Tensor, proxy_scores: Tensor) -> dict[str, float]:
    exact_centered = exact_scores - exact_scores.mean(dim=-1, keepdim=True)
    proxy_centered = proxy_scores - proxy_scores.mean(dim=-1, keepdim=True)
    teacher_log_probability = F.log_softmax(exact_scores.float(), dim=-1)
    proxy_log_probability = F.log_softmax(proxy_scores.float(), dim=-1)
    teacher_probability = teacher_log_probability.exp()
    return {
        "score_squared_error": _square_sum(proxy_centered - exact_centered),
        "score_centered_energy": _square_sum(exact_centered),
        "attention_kl_sum": float(
            (
                teacher_probability
                * (teacher_log_probability - proxy_log_probability)
            ).double().sum()
        ),
    }


def _accumulate_policy(
    sums: dict[str, float | int],
    *,
    exact_scores: Tensor,
    proxy_scores: Tensor,
    proxy_token_mask: Tensor,
    proxy_page_mask: Tensor,
    oracle_token_mask: Tensor,
    oracle_page_mask: Tensor,
    c1_value: Tensor,
    decoder: Tensor,
    c1_full_decoded: Tensor,
    dense_full_decoded: Tensor,
    page_size: int,
    nominal_pages_per_head: int,
) -> None:
    score = _score_metrics(exact_scores, proxy_scores)
    for name, value in score.items():
        sums[name] += value
    proxy_attention = sparse_gqa_value_attention(
        exact_scores,
        c1_value,
        proxy_token_mask,
        heads_per_group=HEADS_PER_GROUP,
    )
    oracle_attention = sparse_gqa_value_attention(
        exact_scores,
        c1_value,
        oracle_token_mask,
        heads_per_group=HEADS_PER_GROUP,
    )
    proxy_decoded = _decode_c1_heads(proxy_attention.output, decoder)
    oracle_decoded = _decode_c1_heads(oracle_attention.output, decoder)
    visible = int(exact_scores.shape[-1])
    pages = math.ceil(visible / page_size)
    selected_tokens = int(proxy_token_mask.sum())
    sums["layer_queries"] += 1
    sums["query_heads"] += NUM_QUERY_HEADS
    sums["selected_teacher_mass_sum"] += float(
        proxy_attention.selected_teacher_mass.double().sum()
    )
    sums["oracle_teacher_mass_sum"] += float(
        oracle_attention.selected_teacher_mass.double().sum()
    )
    sums["page_intersection"] += int((proxy_page_mask & oracle_page_mask).sum())
    sums["oracle_pages"] += int(oracle_page_mask.sum())
    sums["selected_pages"] += int(proxy_page_mask.sum())
    sums["visible_pages"] += NUM_KV_HEADS * pages
    sums["selected_tokens"] += selected_tokens
    sums["visible_tokens"] += NUM_KV_HEADS * visible
    sums["nominal_group_pages"] += NUM_KV_HEADS * nominal_pages_per_head
    sums["logical_exact_k_bytes"] += selected_tokens * HEAD_DIM * 2
    sums["c1_sparse_error"] += _square_sum(proxy_decoded - c1_full_decoded)
    sums["c1_full_energy"] += _square_sum(c1_full_decoded)
    sums["oracle_c1_sparse_error"] += _square_sum(
        oracle_decoded - c1_full_decoded
    )
    sums["c1_sparse_to_dense_error"] += _square_sum(
        proxy_decoded - dense_full_decoded
    )
    sums["dense_energy"] += _square_sum(dense_full_decoded)


def _aggregate_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[int, str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = (
            int(record["sequence_length"]),
            str(record["method"]),
            int(record["routing_rank"]),
            int(record["nominal_token_budget"]),
        )
        groups[key].append(record)
    result = []
    for (length, method, rank, budget), rows in sorted(groups.items()):
        layers = sorted(int(row["layer"]) for row in rows)
        metrics = _finalize_policy(
            _merge_sums([row["metric_sums"] for row in rows])
        )
        layer_count = len(layers)
        metrics["estimated_exact_k_mib_per_full_decode"] = (
            metrics["mean_logical_exact_k_bytes_per_layer_query"]
            * layer_count
            / (1 << 20)
        )
        metrics["routing_sidecar_mib_for_evaluated_layers"] = (
            length * NUM_KV_HEADS * rank * 2 * layer_count / (1 << 20)
        )
        result.append(
            {
                "sequence_length": length,
                "method": method,
                "routing_rank": rank,
                "nominal_token_budget": budget,
                "layers": layers,
                "persistent_gpu_scalar_ratio": routing_storage_ratio(
                    value_rank=64,
                    routing_rank=rank,
                    key_width=HEAD_DIM,
                    value_width=HEAD_DIM,
                ),
                "metrics": metrics,
            }
        )
    return result


def _aggregate_baselines(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[int(record["sequence_length"])].append(record)
    return [
        {
            "sequence_length": length,
            "layers": sorted(int(row["layer"]) for row in rows),
            "metrics": _finalize_baseline(
                _merge_sums([row["metric_sums"] for row in rows])
            ),
        }
        for length, rows in sorted(groups.items())
    ]


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Qwen3-8B C1-V64 + K-routing sidecar length sweep",
        "",
        "Proxy page log-sum-exp selects a per-GQA-group page union. Exact "
        "post-RoPE K is used on every fetched page; all fetched tokens enter "
        "the sparse softmax and resident C1-V64 payload computation.",
        "",
        "| length | method | R | B | mass | exact-page oracle mass | page recall | "
        "C1 sparse rel-L2 | oracle rel-L2 | page fraction | union amp | "
        "K traffic MiB/decode | GPU cache ratio |",
        "|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["aggregate"]:
        metric = row["metrics"]
        lines.append(
            f"| {row['sequence_length']} | {row['method']} | "
            f"{row['routing_rank']} | {row['nominal_token_budget']} | "
            f"{metric['mean_selected_teacher_mass']:.6f} | "
            f"{metric['mean_oracle_teacher_mass']:.6f} | "
            f"{metric['page_recall_vs_exact_qk_oracle']:.6f} | "
            f"{metric['c1_sparse_to_full_relative_l2']:.6e} | "
            f"{metric['oracle_c1_sparse_to_full_relative_l2']:.6e} | "
            f"{metric['selected_page_fraction']:.6f} | "
            f"{metric['gqa_page_union_amplification']:.6f} | "
            f"{metric['estimated_exact_k_mib_per_full_decode']:.3f} | "
            f"{row['persistent_gpu_scalar_ratio']:.4f} |"
        )
    lines.extend(
        [
            "",
            "`exact-page oracle` ranks pages with exact QK log-sum-exp under the "
            "same budget and GQA union. K traffic assumes BF16 exact K and no hot-cache hits.",
            "",
        ]
    )
    return "\n".join(lines)


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("C1+R length sweep requires CUDA")
    lengths = sorted(set(_parse_ints(args.lengths)))
    ranks = sorted(set(_parse_ints(args.ranks)))
    budgets = sorted(set(_parse_ints(args.budgets)))
    layers = sorted(set(_parse_ints(args.layers)))
    methods = _parse_strings(args.methods)
    positive = (
        args.heldout_windows,
        args.batch_size,
        args.page_size,
        args.torch_num_threads,
        *lengths,
        *ranks,
        *budgets,
    )
    if not layers or not methods or min(positive) <= 0:
        raise ValueError("layers/methods and positive sweep values are required")
    if lengths != [1024, 2048, 4096, 8192, 16384, 32768]:
        raise ValueError("controlled sweep requires lengths 1K,2K,4K,8K,16K,32K")
    if args.heldout_start < 0:
        raise ValueError("held-out start must be nonnegative")
    if any(budget % args.page_size for budget in budgets):
        raise ValueError("all token budgets must align to complete pages")

    started = time.perf_counter()
    torch.set_num_threads(args.torch_num_threads)
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats(0)
    model_path = args.model.expanduser().resolve()
    windows_path = args.windows.expanduser().resolve()
    c1_dir = args.c1_export.expanduser().resolve()
    factor_dir = args.routing_factors.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)

    config = AutoConfig.from_pretrained(str(model_path), local_files_only=True)
    _validate_config(config)
    if min(layers) < 0 or max(layers) >= int(config.num_hidden_layers):
        raise ValueError("requested layer lies outside the model")
    factors, factor_result, factor_result_path = _load_routing_factors(
        factor_dir,
        model_path=model_path,
        methods=methods,
    )
    available_rank = min(int(pair[0].shape[-1]) for pair in factors.values())
    if max(ranks) > available_rank:
        raise ValueError("requested routing rank exceeds calibrated factors")

    manifest_path = windows_path.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != WINDOW_FORMAT or manifest.get("status") != "complete":
        raise ValueError("length-sweep windows are incomplete or incompatible")
    if manifest["artifact"]["sha256"] != _sha256(windows_path):
        raise ValueError("length-sweep window hash mismatch")
    if manifest["model"]["config_sha256"] != _sha256(model_path / "config.json"):
        raise ValueError("length-sweep windows belong to another model")
    stored = load_file(str(windows_path), device="cpu")["input_ids"]
    stop = args.heldout_start + args.heldout_windows
    if (
        stored.ndim != 2
        or int(stored.shape[1]) != lengths[-1]
        or stop > int(stored.shape[0])
    ):
        raise ValueError("window bank does not cover the requested 32K slice")
    windows = stored[args.heldout_start:stop]
    del stored

    c1_evaluator.activate_model_profile("qwen3_8b")
    c1_result = c1_evaluator._load_results(c1_dir, model_path)
    value_rank = int(c1_result["fit_config"]["cache_rank_per_head"])
    if value_rank != 64:
        raise ValueError("controlled C1+R sweep requires resident C1-V64")
    dtype = torch.bfloat16 if args.model_dtype == "bfloat16" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        dtype=dtype,
        low_cpu_mem_usage=True,
        local_files_only=True,
        attn_implementation=args.attn_implementation,
        device_map={"": 0},
    ).eval()
    model.config.use_cache = False
    device = model.model.embed_tokens.weight.device
    position_ids = torch.arange(
        lengths[-1], device=device, dtype=torch.long
    ).unsqueeze(0)
    hidden_bank = torch.empty(
        len(windows),
        lengths[-1],
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

    records = []
    baseline_records = []
    requested_layers = set(layers)
    kv_head_index = torch.arange(NUM_QUERY_HEADS, device=device) // HEADS_PER_GROUP
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
                (length, method, rank, budget): _new_policy_sums()
                for length in lengths
                for method in methods
                for rank in ranks
                for budget in budgets
            }
            baseline_sums = {length: _new_baseline_sums() for length in lengths}
            layer_factors = {
                method: (
                    key[layer_index].to(device=device, dtype=torch.float32),
                    query[layer_index].to(device=device, dtype=torch.float32),
                )
                for method, (key, query) in factors.items()
            }
            for example in range(args.heldout_windows):
                full_key = features["post_key"][example].to(
                    device=device, dtype=torch.float32
                )
                full_dense_value = features["dense_value"][example].to(
                    device=device, dtype=torch.float32
                )
                full_c1_value = features["c1_value"][example].to(
                    device=device, dtype=torch.float32
                )
                sidecars = {
                    method: build_routing_sidecar(full_key, key_projector)
                    for method, (key_projector, _) in layer_factors.items()
                }
                for length in lengths:
                    position = length - 1
                    query = features["query"][example, :, position].to(
                        device=device, dtype=torch.float32
                    )
                    key = full_key[:, :length]
                    dense_value = full_dense_value[:, :length]
                    c1_value = full_c1_value[:, :length]
                    exact_scores = torch.einsum(
                        "hd,hld->hl",
                        query,
                        key.index_select(0, kv_head_index),
                    ) / math.sqrt(HEAD_DIM)
                    dense_full_heads = full_gqa_value_attention(
                        exact_scores,
                        dense_value,
                        heads_per_group=HEADS_PER_GROUP,
                    )
                    c1_full_heads = full_gqa_value_attention(
                        exact_scores,
                        c1_value,
                        heads_per_group=HEADS_PER_GROUP,
                    )
                    dense_full_decoded = _decode_dense_heads(
                        o_proj_weight,
                        o_proj_bias,
                        dense_full_heads,
                    )
                    c1_full_decoded = _decode_c1_heads(c1_full_heads, decoder)
                    baseline = baseline_sums[length]
                    baseline["layer_queries"] += 1
                    baseline["c1_full_to_dense_error"] += _square_sum(
                        c1_full_decoded - dense_full_decoded
                    )
                    baseline["dense_energy"] += _square_sum(dense_full_decoded)

                    oracle_masks = {}
                    for budget in budgets:
                        pages_per_head = min(
                            math.ceil(budget / args.page_size),
                            math.ceil(length / args.page_size),
                        )
                        oracle_masks[budget] = gqa_union_page_mass_mask(
                            exact_scores,
                            num_kv_heads=NUM_KV_HEADS,
                            page_size=args.page_size,
                            pages_per_query_head=pages_per_head,
                        )
                    for method in methods:
                        _, query_projector = layer_factors[method]
                        for rank in ranks:
                            proxy_scores = routing_proxy_scores(
                                query,
                                sidecars[method][:, :length, :rank],
                                query_projector[..., :rank],
                                head_dim=HEAD_DIM,
                            )
                            for budget in budgets:
                                pages_per_head = min(
                                    math.ceil(budget / args.page_size),
                                    math.ceil(length / args.page_size),
                                )
                                proxy_token_mask, proxy_page_mask = (
                                    gqa_union_page_mass_mask(
                                        proxy_scores,
                                        num_kv_heads=NUM_KV_HEADS,
                                        page_size=args.page_size,
                                        pages_per_query_head=pages_per_head,
                                    )
                                )
                                oracle_token_mask, oracle_page_mask = oracle_masks[
                                    budget
                                ]
                                _accumulate_policy(
                                    policy_sums[(length, method, rank, budget)],
                                    exact_scores=exact_scores,
                                    proxy_scores=proxy_scores,
                                    proxy_token_mask=proxy_token_mask,
                                    proxy_page_mask=proxy_page_mask,
                                    oracle_token_mask=oracle_token_mask,
                                    oracle_page_mask=oracle_page_mask,
                                    c1_value=c1_value,
                                    decoder=decoder,
                                    c1_full_decoded=c1_full_decoded,
                                    dense_full_decoded=dense_full_decoded,
                                    page_size=args.page_size,
                                    nominal_pages_per_head=pages_per_head,
                                )
                print(
                    f"[C1+R sweep] layer={layer_index} heldout="
                    f"{example + 1}/{args.heldout_windows}",
                    flush=True,
                )
                del full_key, full_dense_value, full_c1_value, sidecars
            for (length, method, rank, budget), sums in policy_sums.items():
                records.append(
                    {
                        "layer": layer_index,
                        "sequence_length": length,
                        "method": method,
                        "routing_rank": rank,
                        "nominal_token_budget": budget,
                        "page_size": args.page_size,
                        "metric_sums": sums,
                        "metrics": _finalize_policy(sums),
                    }
                )
            for length, sums in baseline_sums.items():
                baseline_records.append(
                    {
                        "layer": layer_index,
                        "sequence_length": length,
                        "metric_sums": sums,
                        "metrics": _finalize_baseline(sums),
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
                layer_factors,
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
            "resident_payload": "frozen C1-V64; routing sidecar is metadata-only",
            "routing_code": "post-RoPE exact K projected once into R dimensions",
            "page_selector": "per-query-head proxy page log-sum-exp then GQA union",
            "exact_refinement": "exact post-RoPE QK over every token in fetched pages",
            "normalization": "softmax over all fetched pages; no second token Top-k",
            "exact_page_oracle": "same page budget selected by exact-QK log-sum-exp",
        },
        "model": {
            "path": str(model_path),
            "config_sha256": _sha256(model_path / "config.json"),
        },
        "windows": {
            "path": str(windows_path),
            "sha256": _sha256(windows_path),
            "manifest_sha256": _sha256(manifest_path),
            "indices": list(range(args.heldout_start, stop)),
            "nested_prefix_lengths": lengths,
        },
        "c1_export": {
            "path": str(c1_dir),
            "results_sha256": _sha256(c1_result_path),
            "value_rank": value_rank,
        },
        "routing_factors": {
            "path": str(factor_dir),
            "result_sha256": _sha256(factor_result_path),
            "calibration": factor_result["calibration"],
            "methods": methods,
            "ranks": ranks,
        },
        "configuration": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "records": records,
        "aggregate": _aggregate_records(records),
        "c1_full_baselines": baseline_records,
        "aggregate_c1_full_baselines": _aggregate_baselines(baseline_records),
        "runtime": {
            "seconds": time.perf_counter() - started,
            "peak_cuda_bytes": torch.cuda.max_memory_allocated(0),
            "torch_version": torch.__version__,
        },
    }
    _atomic_text(
        output_dir / "result.json",
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )
    _atomic_text(output_dir / "summary.md", _markdown(payload))
    print(f"[C1+R sweep] wrote {output_dir}", flush=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--c1-export", type=Path, required=True)
    parser.add_argument(
        "--routing-factors",
        type=Path,
        required=True,
        help=(
            "post-RoPE KQ-SVD export, or a pairwise KQ-SVD export whose "
            "independent per-layer factors are used with --methods kq_svd"
        ),
    )
    parser.add_argument("--layers", default=",".join(str(i) for i in range(36)))
    parser.add_argument("--lengths", default="1024,2048,4096,8192,16384,32768")
    parser.add_argument("--ranks", default="8,16,32,64")
    parser.add_argument("--methods", default="key_svd,kq_svd")
    parser.add_argument("--budgets", default="256,512,1024")
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument("--heldout-start", type=int, default=0)
    parser.add_argument("--heldout-windows", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--torch-num-threads", type=int, default=4)
    parser.add_argument("--model-dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument(
        "--attn-implementation",
        choices=("sdpa", "eager"),
        default="sdpa",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


if __name__ == "__main__":
    evaluate(_parser().parse_args())
