#!/usr/bin/env python3
"""Probe whether same-layer dense Value linearly contains Key information.

This is a frozen-model Stage-A diagnostic.  It compares dense V and the
existing C1-V latent as sources for two independent physical-head probes:

* ``pre_rope`` predicts pre-RoPE K and then applies the exact token RoPE;
* ``direct_post`` predicts post-RoPE K with one position-independent map.

The held-out evaluation measures Key reconstruction, centered QK score error,
teacher-to-proxy attention KL, Top-k recall, and retained teacher mass.  It does
not claim a deployable sparse-attention runtime.
"""

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

from safetensors.torch import load_file, save_file
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from transformers import AutoConfig, AutoModelForCausalLM


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basisserve.core.c1_v_k_index import apply_rotary  # noqa: E402
from basisserve.core.v_k_linear_probe import (  # noqa: E402
    HeadwiseLinearProbe,
    apply_headwise_linear_probe,
    fit_headwise_linear_probe,
)
from evaluation import eval_qwen3_32b_c1_wikitext as c1_evaluator  # noqa: E402
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


FORMAT = "basisserve.qwen3_8b.dense_v_k_proxy.v3"


def _parse_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_safetensors(path: Path, tensors: dict[str, Tensor]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    save_file(tensors, str(temporary), metadata={"format": FORMAT})
    os.replace(temporary, path)


@torch.inference_mode()
def _extract_layer_features(
    layer: nn.Module,
    hidden_bank: Tensor,
    *,
    value_encoder: Tensor,
    position_embeddings: tuple[Tensor, Tensor],
    batch_size: int,
) -> dict[str, Tensor]:
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

    attention = layer.self_attn
    device = attention.q_proj.weight.device
    model_dtype = attention.q_proj.weight.dtype
    encoder = value_encoder.to(device=device, dtype=model_dtype)
    chunks: dict[str, list[Tensor]] = defaultdict(list)
    for _, hidden in _batches(hidden_bank, 0, len(hidden_bank), batch_size):
        hidden = hidden.to(device=device, dtype=model_dtype, non_blocking=True)
        attention_input = layer.input_layernorm(hidden)
        batch, sequence, _ = attention_input.shape
        query = attention.q_norm(
            attention.q_proj(attention_input).view(
                batch, sequence, NUM_QUERY_HEADS, HEAD_DIM
            )
        ).transpose(1, 2)
        pre_key = attention.k_norm(
            attention.k_proj(attention_input).view(
                batch, sequence, NUM_KV_HEADS, HEAD_DIM
            )
        ).transpose(1, 2)
        dense_value = attention.v_proj(attention_input).view(
            batch, sequence, NUM_KV_HEADS, HEAD_DIM
        ).permute(0, 2, 1, 3)
        post_query, post_key = apply_rotary_pos_emb(
            query,
            pre_key,
            *position_embeddings,
        )
        c1_value = torch.einsum("bhsd,hdr->bhsr", dense_value, encoder)
        for name, value in (
            ("query", post_query),
            ("pre_key", pre_key),
            ("post_key", post_key),
            ("dense_value", dense_value),
            ("c1_value", c1_value),
        ):
            chunks[name].append(value.to(device="cpu").contiguous())
        del hidden, attention_input, query, pre_key, dense_value, post_query, post_key
    return {name: torch.cat(values) for name, values in chunks.items()}


def _fit_source_probes(
    source: Tensor,
    pre_key: Tensor,
    post_key: Tensor,
    *,
    relative_ridge: float,
) -> tuple[HeadwiseLinearProbe, HeadwiseLinearProbe]:
    return (
        fit_headwise_linear_probe(
            source,
            pre_key,
            relative_ridge=relative_ridge,
        ),
        fit_headwise_linear_probe(
            source,
            post_key,
            relative_ridge=relative_ridge,
        ),
    )


def _key_metric_sums(reference: Tensor, candidate: Tensor) -> dict[str, float | int]:
    if tuple(reference.shape) != tuple(candidate.shape):
        raise ValueError("Key reference and candidate geometry differ")
    reference_fp64 = reference.detach().to(device="cpu", dtype=torch.float64)
    candidate_fp64 = candidate.detach().to(device="cpu", dtype=torch.float64)
    mean = reference_fp64.mean(dim=(0, 2), keepdim=True)
    error = float((candidate_fp64 - reference_fp64).square().sum())
    energy = float((reference_fp64 - mean).square().sum())
    cosine = F.cosine_similarity(
        reference.detach().float(), candidate.detach().float(), dim=-1
    )
    return {
        "squared_error": error,
        "centered_energy": energy,
        "cosine_sum": float(cosine.double().sum()),
        "vectors": cosine.numel(),
    }


def _prediction(
    source: Tensor,
    probe: HeadwiseLinearProbe,
    *,
    rotate: bool,
    cos: Tensor,
    sin: Tensor,
) -> Tensor:
    candidate = apply_headwise_linear_probe(source, probe)
    return apply_rotary(candidate, cos, sin) if rotate else candidate


def _new_score_sums(budgets: list[int]) -> dict[str, Any]:
    return {
        "score_squared_error": 0.0,
        "score_centered_energy": 0.0,
        "attention_kl_sum": 0.0,
        "query_heads": 0,
        "budgets": {
            str(budget): {
                "top_k_recall_sum": 0.0,
                "teacher_mass_sum": 0.0,
                "oracle_teacher_mass_sum": 0.0,
            }
            for budget in budgets
        },
        "page": {
            "query_heads": 0,
            "groups": 0,
            "page_count": 0,
            "pages_per_head": 0,
            "candidate_pages_per_head": 0,
            "page_recall_sum": 0.0,
            "mass_page_recall_sum": 0.0,
            "top_k_covered_by_pages_sum": 0.0,
            "teacher_mass_sum": 0.0,
            "oracle_teacher_mass_sum": 0.0,
            "gqa_union_mass_page_recall_sum": 0.0,
            "gqa_union_teacher_mass_sum": 0.0,
            "gqa_oracle_union_teacher_mass_sum": 0.0,
            "gqa_union_amplification_sum": 0.0,
            "gqa_union_pages_sum": 0.0,
            "candidate_mass_page_recall_sum": 0.0,
            "candidate_teacher_mass_sum": 0.0,
            "candidate_gqa_union_mass_page_recall_sum": 0.0,
            "candidate_gqa_union_teacher_mass_sum": 0.0,
            "candidate_gqa_union_pages_sum": 0.0,
            "reranked_page_recall_sum": 0.0,
            "reranked_mass_page_recall_sum": 0.0,
            "reranked_top_k_covered_by_pages_sum": 0.0,
            "reranked_teacher_mass_sum": 0.0,
            "reranked_gqa_union_mass_page_recall_sum": 0.0,
            "reranked_gqa_union_teacher_mass_sum": 0.0,
            "reranked_gqa_union_pages_sum": 0.0,
        },
    }


def _page_selection_metrics(
    *,
    exact_scores: Tensor,
    proxy_scores: Tensor,
    teacher_probability: Tensor,
    page_size: int,
    token_budget: int,
    candidate_token_budget: int,
) -> dict[str, float | int]:
    heads, visible = map(int, exact_scores.shape)
    page_count = math.ceil(visible / page_size)
    pages_per_head = min(math.ceil(token_budget / page_size), page_count)
    candidate_pages_per_head = min(
        max(math.ceil(candidate_token_budget / page_size), pages_per_head),
        page_count,
    )
    padding = page_count * page_size - visible
    exact_padded = F.pad(exact_scores.float(), (0, padding), value=-torch.inf)
    proxy_padded = F.pad(proxy_scores.float(), (0, padding), value=-torch.inf)
    probability_padded = F.pad(teacher_probability.float(), (0, padding))
    exact_pages = exact_padded.reshape(heads, page_count, page_size)
    proxy_pages = proxy_padded.reshape(heads, page_count, page_size)
    page_probability = probability_padded.reshape(heads, page_count, page_size).sum(
        dim=-1
    )

    proxy_page_scores = proxy_pages.logsumexp(dim=-1)
    exact_page_scores = exact_pages.logsumexp(dim=-1)
    proxy_page_indices = proxy_page_scores.topk(
        pages_per_head, dim=-1
    ).indices
    candidate_page_indices = proxy_page_scores.topk(
        candidate_pages_per_head, dim=-1
    ).indices
    exact_max_page_indices = exact_pages.amax(dim=-1).topk(
        pages_per_head, dim=-1
    ).indices
    exact_mass_page_indices = page_probability.topk(
        pages_per_head, dim=-1
    ).indices
    selected_page_mask = torch.zeros(
        heads,
        page_count,
        dtype=torch.bool,
        device=exact_scores.device,
    )
    selected_page_mask.scatter_(1, proxy_page_indices, True)
    candidate_page_mask = torch.zeros_like(selected_page_mask)
    candidate_page_mask.scatter_(1, candidate_page_indices, True)
    reranked_page_indices = exact_page_scores.masked_fill(
        ~candidate_page_mask, -torch.inf
    ).topk(pages_per_head, dim=-1).indices
    reranked_page_mask = torch.zeros_like(selected_page_mask)
    reranked_page_mask.scatter_(1, reranked_page_indices, True)

    page_recall = selected_page_mask.gather(1, exact_max_page_indices).float().mean(
        dim=-1
    )
    mass_page_recall = selected_page_mask.gather(
        1, exact_mass_page_indices
    ).float().mean(dim=-1)
    selected_mass = page_probability.gather(1, proxy_page_indices).sum(dim=-1)
    oracle_mass = page_probability.gather(1, exact_mass_page_indices).sum(dim=-1)
    candidate_mass_page_recall = candidate_page_mask.gather(
        1, exact_mass_page_indices
    ).float().mean(dim=-1)
    candidate_mass = page_probability.gather(1, candidate_page_indices).sum(dim=-1)
    reranked_page_recall = reranked_page_mask.gather(
        1, exact_max_page_indices
    ).float().mean(dim=-1)
    reranked_mass_page_recall = reranked_page_mask.gather(
        1, exact_mass_page_indices
    ).float().mean(dim=-1)
    reranked_mass = page_probability.gather(1, reranked_page_indices).sum(dim=-1)

    selected_tokens = min(token_budget, visible)
    exact_token_indices = exact_scores.topk(selected_tokens, dim=-1).indices
    exact_token_page_indices = torch.div(
        exact_token_indices, page_size, rounding_mode="floor"
    )
    top_k_covered_by_pages = selected_page_mask.gather(
        1, exact_token_page_indices
    ).float().mean(dim=-1)
    reranked_top_k_covered_by_pages = reranked_page_mask.gather(
        1, exact_token_page_indices
    ).float().mean(dim=-1)

    groups = heads // HEADS_PER_GROUP
    selected_group_mask = selected_page_mask.reshape(
        groups, HEADS_PER_GROUP, page_count
    ).any(dim=1)
    candidate_group_mask = candidate_page_mask.reshape(
        groups, HEADS_PER_GROUP, page_count
    ).any(dim=1)
    reranked_group_mask = reranked_page_mask.reshape(
        groups, HEADS_PER_GROUP, page_count
    ).any(dim=1)
    oracle_page_mask = torch.zeros_like(selected_page_mask)
    oracle_page_mask.scatter_(1, exact_mass_page_indices, True)
    oracle_group_mask = oracle_page_mask.reshape(
        groups, HEADS_PER_GROUP, page_count
    ).any(dim=1)
    selected_union_mask = selected_group_mask.repeat_interleave(
        HEADS_PER_GROUP, dim=0
    )
    candidate_union_mask = candidate_group_mask.repeat_interleave(
        HEADS_PER_GROUP, dim=0
    )
    reranked_union_mask = reranked_group_mask.repeat_interleave(
        HEADS_PER_GROUP, dim=0
    )
    oracle_union_mask = oracle_group_mask.repeat_interleave(
        HEADS_PER_GROUP, dim=0
    )
    gqa_union_mass_page_recall = selected_union_mask.gather(
        1, exact_mass_page_indices
    ).float().mean(dim=-1)
    gqa_union_mass = (page_probability * selected_union_mask).sum(dim=-1)
    gqa_oracle_union_mass = (page_probability * oracle_union_mask).sum(dim=-1)
    gqa_union_amplification = (
        selected_group_mask.sum(dim=-1).float() / pages_per_head
    )
    candidate_gqa_union_mass_page_recall = candidate_union_mask.gather(
        1, exact_mass_page_indices
    ).float().mean(dim=-1)
    candidate_gqa_union_mass = (
        page_probability * candidate_union_mask
    ).sum(dim=-1)
    reranked_gqa_union_mass_page_recall = reranked_union_mask.gather(
        1, exact_mass_page_indices
    ).float().mean(dim=-1)
    reranked_gqa_union_mass = (
        page_probability * reranked_union_mask
    ).sum(dim=-1)
    return {
        "query_heads": heads,
        "groups": groups,
        "page_recall_sum": float(page_recall.double().sum()),
        "mass_page_recall_sum": float(mass_page_recall.double().sum()),
        "top_k_covered_by_pages_sum": float(
            top_k_covered_by_pages.double().sum()
        ),
        "teacher_mass_sum": float(selected_mass.double().sum()),
        "oracle_teacher_mass_sum": float(oracle_mass.double().sum()),
        "gqa_union_mass_page_recall_sum": float(
            gqa_union_mass_page_recall.double().sum()
        ),
        "gqa_union_teacher_mass_sum": float(gqa_union_mass.double().sum()),
        "gqa_oracle_union_teacher_mass_sum": float(
            gqa_oracle_union_mass.double().sum()
        ),
        "gqa_union_amplification_sum": float(
            gqa_union_amplification.double().sum()
        ),
        "gqa_union_pages_sum": float(selected_group_mask.sum().double()),
        "candidate_mass_page_recall_sum": float(
            candidate_mass_page_recall.double().sum()
        ),
        "candidate_teacher_mass_sum": float(candidate_mass.double().sum()),
        "candidate_gqa_union_mass_page_recall_sum": float(
            candidate_gqa_union_mass_page_recall.double().sum()
        ),
        "candidate_gqa_union_teacher_mass_sum": float(
            candidate_gqa_union_mass.double().sum()
        ),
        "candidate_gqa_union_pages_sum": float(
            candidate_group_mask.sum().double()
        ),
        "reranked_page_recall_sum": float(reranked_page_recall.double().sum()),
        "reranked_mass_page_recall_sum": float(
            reranked_mass_page_recall.double().sum()
        ),
        "reranked_top_k_covered_by_pages_sum": float(
            reranked_top_k_covered_by_pages.double().sum()
        ),
        "reranked_teacher_mass_sum": float(reranked_mass.double().sum()),
        "reranked_gqa_union_mass_page_recall_sum": float(
            reranked_gqa_union_mass_page_recall.double().sum()
        ),
        "reranked_gqa_union_teacher_mass_sum": float(
            reranked_gqa_union_mass.double().sum()
        ),
        "reranked_gqa_union_pages_sum": float(
            reranked_group_mask.sum().double()
        ),
        "page_count": page_count,
        "pages_per_head": pages_per_head,
        "candidate_pages_per_head": candidate_pages_per_head,
    }


def _accumulate_score_metrics(
    accumulator: dict[str, Any],
    *,
    exact_scores: Tensor,
    proxy_scores: Tensor,
    budgets: list[int],
    page_size: int,
    page_token_budget: int,
    page_candidate_token_budget: int,
) -> None:
    exact_centered = exact_scores - exact_scores.mean(dim=-1, keepdim=True)
    proxy_centered = proxy_scores - proxy_scores.mean(dim=-1, keepdim=True)
    accumulator["score_squared_error"] += float(
        (proxy_centered - exact_centered).double().square().sum()
    )
    accumulator["score_centered_energy"] += float(
        exact_centered.double().square().sum()
    )
    teacher_log_probability = F.log_softmax(exact_scores.float(), dim=-1)
    proxy_log_probability = F.log_softmax(proxy_scores.float(), dim=-1)
    teacher_probability = teacher_log_probability.exp()
    accumulator["attention_kl_sum"] += float(
        (
            teacher_probability
            * (teacher_log_probability - proxy_log_probability)
        ).double().sum()
    )
    heads, visible = map(int, exact_scores.shape)
    accumulator["query_heads"] += heads
    for budget in budgets:
        selected = min(budget, visible)
        exact_indices = exact_scores.topk(selected, dim=-1).indices
        proxy_indices = proxy_scores.topk(selected, dim=-1).indices
        exact_membership = torch.zeros(
            heads,
            visible,
            dtype=torch.bool,
            device=exact_scores.device,
        )
        exact_membership.scatter_(1, exact_indices, True)
        recall = exact_membership.gather(1, proxy_indices).float().mean(dim=-1)
        teacher_mass = teacher_probability.gather(1, proxy_indices).sum(dim=-1)
        oracle_mass = teacher_probability.gather(1, exact_indices).sum(dim=-1)
        row = accumulator["budgets"][str(budget)]
        row["top_k_recall_sum"] += float(recall.double().sum())
        row["teacher_mass_sum"] += float(teacher_mass.double().sum())
        row["oracle_teacher_mass_sum"] += float(oracle_mass.double().sum())
    page_metrics = _page_selection_metrics(
        exact_scores=exact_scores,
        proxy_scores=proxy_scores,
        teacher_probability=teacher_probability,
        page_size=page_size,
        token_budget=page_token_budget,
        candidate_token_budget=page_candidate_token_budget,
    )
    page_row = accumulator["page"]
    for name in (
        "query_heads",
        "groups",
        "page_recall_sum",
        "mass_page_recall_sum",
        "top_k_covered_by_pages_sum",
        "teacher_mass_sum",
        "oracle_teacher_mass_sum",
        "gqa_union_mass_page_recall_sum",
        "gqa_union_teacher_mass_sum",
        "gqa_oracle_union_teacher_mass_sum",
        "gqa_union_amplification_sum",
        "gqa_union_pages_sum",
        "candidate_mass_page_recall_sum",
        "candidate_teacher_mass_sum",
        "candidate_gqa_union_mass_page_recall_sum",
        "candidate_gqa_union_teacher_mass_sum",
        "candidate_gqa_union_pages_sum",
        "reranked_page_recall_sum",
        "reranked_mass_page_recall_sum",
        "reranked_top_k_covered_by_pages_sum",
        "reranked_teacher_mass_sum",
        "reranked_gqa_union_mass_page_recall_sum",
        "reranked_gqa_union_teacher_mass_sum",
        "reranked_gqa_union_pages_sum",
    ):
        page_row[name] += page_metrics[name]
    page_row["page_count"] = page_metrics["page_count"]
    page_row["pages_per_head"] = page_metrics["pages_per_head"]
    page_row["candidate_pages_per_head"] = page_metrics[
        "candidate_pages_per_head"
    ]


def _finalize_score_sums(sums: dict[str, Any]) -> dict[str, Any]:
    count = int(sums["query_heads"])
    result = {
        **{key: sums[key] for key in (
            "score_squared_error",
            "score_centered_energy",
            "attention_kl_sum",
            "query_heads",
        )},
        "centered_score_relative_mse": (
            sums["score_squared_error"] / max(sums["score_centered_energy"], 1.0e-300)
        ),
        "centered_score_relative_rmse": math.sqrt(
            sums["score_squared_error"] / max(sums["score_centered_energy"], 1.0e-300)
        ),
        "mean_attention_kl_teacher_to_proxy": sums["attention_kl_sum"] / count,
        "budgets": {},
    }
    for budget, row in sums["budgets"].items():
        result["budgets"][budget] = {
            "mean_top_k_recall": row["top_k_recall_sum"] / count,
            "mean_teacher_mass_selected": row["teacher_mass_sum"] / count,
            "mean_oracle_top_k_teacher_mass": (
                row["oracle_teacher_mass_sum"] / count
            ),
            **row,
        }
    page = sums["page"]
    page_heads = int(page["query_heads"])
    groups = int(page["groups"])
    result["page"] = {
        **page,
        "mean_page_recall": page["page_recall_sum"] / page_heads,
        "mean_mass_page_recall": page["mass_page_recall_sum"] / page_heads,
        "mean_top_k_covered_by_pages": (
            page["top_k_covered_by_pages_sum"] / page_heads
        ),
        "mean_teacher_mass_selected": page["teacher_mass_sum"] / page_heads,
        "mean_oracle_teacher_mass": (
            page["oracle_teacher_mass_sum"] / page_heads
        ),
        "mean_gqa_union_mass_page_recall": (
            page["gqa_union_mass_page_recall_sum"] / page_heads
        ),
        "mean_gqa_union_teacher_mass_selected": (
            page["gqa_union_teacher_mass_sum"] / page_heads
        ),
        "mean_gqa_oracle_union_teacher_mass": (
            page["gqa_oracle_union_teacher_mass_sum"] / page_heads
        ),
        "mean_gqa_union_amplification": (
            page["gqa_union_amplification_sum"] / groups
        ),
        "mean_gqa_union_pages": page["gqa_union_pages_sum"] / groups,
        "mean_candidate_mass_page_recall": (
            page["candidate_mass_page_recall_sum"] / page_heads
        ),
        "mean_candidate_teacher_mass": (
            page["candidate_teacher_mass_sum"] / page_heads
        ),
        "mean_candidate_gqa_union_mass_page_recall": (
            page["candidate_gqa_union_mass_page_recall_sum"] / page_heads
        ),
        "mean_candidate_gqa_union_teacher_mass": (
            page["candidate_gqa_union_teacher_mass_sum"] / page_heads
        ),
        "mean_candidate_gqa_union_pages": (
            page["candidate_gqa_union_pages_sum"] / groups
        ),
        "mean_reranked_page_recall": (
            page["reranked_page_recall_sum"] / page_heads
        ),
        "mean_reranked_mass_page_recall": (
            page["reranked_mass_page_recall_sum"] / page_heads
        ),
        "mean_reranked_top_k_covered_by_pages": (
            page["reranked_top_k_covered_by_pages_sum"] / page_heads
        ),
        "mean_reranked_teacher_mass": (
            page["reranked_teacher_mass_sum"] / page_heads
        ),
        "mean_reranked_gqa_union_mass_page_recall": (
            page["reranked_gqa_union_mass_page_recall_sum"] / page_heads
        ),
        "mean_reranked_gqa_union_teacher_mass": (
            page["reranked_gqa_union_teacher_mass_sum"] / page_heads
        ),
        "mean_reranked_gqa_union_pages": (
            page["reranked_gqa_union_pages_sum"] / groups
        ),
    }
    return result


def _merge_score_sums(rows: list[dict[str, Any]], budgets: list[int]) -> dict[str, Any]:
    merged = _new_score_sums(budgets)
    for row in rows:
        for name in (
            "score_squared_error",
            "score_centered_energy",
            "attention_kl_sum",
            "query_heads",
        ):
            merged[name] += row[name]
        for budget in budgets:
            target = merged["budgets"][str(budget)]
            source = row["budgets"][str(budget)]
            for name in target:
                target[name] += source[name]
        for name, value in row["page"].items():
            if name not in {
                "page_count",
                "pages_per_head",
                "candidate_pages_per_head",
            }:
                merged["page"][name] += value
        merged["page"]["page_count"] = row["page"]["page_count"]
        merged["page"]["pages_per_head"] = row["page"]["pages_per_head"]
        merged["page"]["candidate_pages_per_head"] = row["page"][
            "candidate_pages_per_head"
        ]
    return merged


def _record(
    *,
    layer: int,
    proxy: str,
    source: str,
    target: str,
    post_key_sums: dict[str, float | int],
    score_sums: dict[str, Any],
    pre_key_sums: dict[str, float | int] | None,
) -> dict[str, Any]:
    result = {
        "layer": layer,
        "proxy": proxy,
        "source": source,
        "target": target,
        "post_key": {
            **post_key_sums,
            "centered_relative_mse": (
                float(post_key_sums["squared_error"])
                / max(float(post_key_sums["centered_energy"]), 1.0e-300)
            ),
            "mean_cosine": (
                float(post_key_sums["cosine_sum"])
                / int(post_key_sums["vectors"])
            ),
        },
        "score": _finalize_score_sums(score_sums),
    }
    if pre_key_sums is not None:
        result["pre_key"] = {
            **pre_key_sums,
            "centered_relative_mse": (
                float(pre_key_sums["squared_error"])
                / max(float(pre_key_sums["centered_energy"]), 1.0e-300)
            ),
            "mean_cosine": (
                float(pre_key_sums["cosine_sum"])
                / int(pre_key_sums["vectors"])
            ),
        }
    return result


def _aggregate(records: list[dict[str, Any]], budgets: list[int]) -> list[dict[str, Any]]:
    by_proxy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_proxy[str(record["proxy"])].append(record)
    result = []
    for proxy, rows in sorted(by_proxy.items()):
        post_sums = {
            name: sum(float(row["post_key"][name]) for row in rows)
            for name in ("squared_error", "centered_energy", "cosine_sum", "vectors")
        }
        pre_rows = [row for row in rows if "pre_key" in row]
        pre_sums = None
        if pre_rows:
            pre_sums = {
                name: sum(float(row["pre_key"][name]) for row in pre_rows)
                for name in ("squared_error", "centered_energy", "cosine_sum", "vectors")
            }
        raw_score_rows = []
        for row in rows:
            score = row["score"]
            raw_page = {
                name: score["page"][name]
                for name in _new_score_sums(budgets)["page"]
            }
            raw_score_rows.append(
                {
                    **{
                        name: score[name]
                        for name in (
                            "score_squared_error",
                            "score_centered_energy",
                            "attention_kl_sum",
                            "query_heads",
                            "budgets",
                        )
                    },
                    "page": raw_page,
                }
            )
        result.append(
            _record(
                layer=-1,
                proxy=proxy,
                source=str(rows[0]["source"]),
                target=str(rows[0]["target"]),
                post_key_sums=post_sums,
                pre_key_sums=pre_sums,
                score_sums=_merge_score_sums(raw_score_rows, budgets),
            )
        )
    return result


def _table_rows(rows: list[dict[str, Any]], report_budget: int) -> list[str]:
    result = []
    for row in rows:
        pre = row.get("pre_key", {}).get("centered_relative_mse")
        score = row["score"]
        budget = score["budgets"][str(report_budget)]
        result.append(
            f"| {'all' if row['layer'] < 0 else row['layer']} | {row['proxy']} | "
            f"{'-' if pre is None else f'{pre:.6f}'} | "
            f"{row['post_key']['centered_relative_mse']:.6f} | "
            f"{row['post_key']['mean_cosine']:.6f} | "
            f"{score['centered_score_relative_rmse']:.6f} | "
            f"{score['mean_attention_kl_teacher_to_proxy']:.6f} | "
            f"{budget['mean_top_k_recall']:.6f} | "
            f"{budget['mean_teacher_mass_selected']:.6f} | "
            f"{budget['mean_oracle_top_k_teacher_mass']:.6f} |"
        )
    return result


def _page_table_rows(rows: list[dict[str, Any]]) -> list[str]:
    result = []
    for row in rows:
        page = row["score"]["page"]
        result.append(
            f"| {'all' if row['layer'] < 0 else row['layer']} | {row['proxy']} | "
            f"{page['mean_page_recall']:.6f} | "
            f"{page['mean_mass_page_recall']:.6f} | "
            f"{page['mean_top_k_covered_by_pages']:.6f} | "
            f"{page['mean_teacher_mass_selected']:.6f} | "
            f"{page['mean_oracle_teacher_mass']:.6f} | "
            f"{page['mean_gqa_union_mass_page_recall']:.6f} | "
            f"{page['mean_gqa_union_teacher_mass_selected']:.6f} | "
            f"{page['mean_gqa_oracle_union_teacher_mass']:.6f} | "
            f"{page['mean_gqa_union_amplification']:.6f} |"
        )
    return result


def _rerank_page_table_rows(rows: list[dict[str, Any]]) -> list[str]:
    result = []
    for row in rows:
        page = row["score"]["page"]
        result.append(
            f"| {'all' if row['layer'] < 0 else row['layer']} | {row['proxy']} | "
            f"{page['mean_mass_page_recall']:.6f} | "
            f"{page['mean_teacher_mass_selected']:.6f} | "
            f"{page['mean_candidate_mass_page_recall']:.6f} | "
            f"{page['mean_candidate_teacher_mass']:.6f} | "
            f"{page['mean_reranked_mass_page_recall']:.6f} | "
            f"{page['mean_reranked_teacher_mass']:.6f} | "
            f"{page['mean_oracle_teacher_mass']:.6f} | "
            f"{page['mean_candidate_gqa_union_pages']:.6f} | "
            f"{page['mean_reranked_gqa_union_pages']:.6f} | "
            f"{page['mean_reranked_gqa_union_mass_page_recall']:.6f} | "
            f"{page['mean_reranked_gqa_union_teacher_mass']:.6f} | "
            f"{page['mean_gqa_oracle_union_teacher_mass']:.6f} |"
        )
    return result


def _markdown(payload: dict[str, Any]) -> str:
    budget = max(int(item) for item in payload["configuration"]["budgets"].split(","))
    page_size = int(payload["configuration"]["page_size"])
    page_token_budget = int(payload["configuration"]["page_token_budget"])
    page_candidate_token_budget = int(
        payload["configuration"]["page_candidate_token_budget"]
    )
    pages_per_head = math.ceil(page_token_budget / page_size)
    candidate_pages_per_head = math.ceil(page_candidate_token_budget / page_size)
    value_rank = int(payload["c1_export"]["value_rank"])
    header = [
        "| layer | proxy | pre-K centered rel-MSE | post-K centered rel-MSE | "
        "post-K cosine | centered-score rel-RMSE | KL(P||P_proxy) | "
        f"R@{budget} | mass@{budget} | oracle-mass@{budget} |",
        "|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    return "\n".join(
        [
            "# Qwen3-8B dense-V to K linear probe",
            "",
            "The probe is fit on C4 windows that are disjoint from every reported "
            "held-out metric. `pre_rope` uses exact token positions only after the "
            "linear V-to-K prediction; `direct_post` is a single position-independent "
            f"map. C1-V{value_rank} is evaluated on the same examples as a controlled source "
            "ablation.",
            "",
            "## Aggregate",
            "",
            *header,
            *_table_rows(payload["aggregate"], budget),
            "",
            "## Per layer",
            "",
            *header,
            *_table_rows(payload["records"], budget),
            "",
            f"## Page{page_size} / B{page_token_budget} selector quality",
            "",
            f"Proxy token logits are reduced with page log-sum-exp and select "
            f"{pages_per_head} pages per Query head before the physical-GQA union.",
            "",
            "| layer | proxy | page recall | mass-page recall | exact-Top-k "
            "covered | mass | oracle mass | union mass-page recall | union mass | "
            "oracle union mass | union amp |",
            "|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            *_page_table_rows(payload["aggregate"]),
            *_page_table_rows(payload["records"]),
            "",
            f"## Page{page_size} exact rerank: {candidate_pages_per_head} candidates "
            f"to {pages_per_head} retained",
            "",
            "The proxy first selects candidate pages. Exact-QK page log-sum-exp "
            "then reranks only those candidates and retains the final pages. Candidate "
            "GQA-union pages model exact-K loads; reranked GQA-union pages model final "
            "V/attention loads.",
            "",
            "| layer | proxy | direct mass-page recall | direct mass | candidate "
            "oracle32 recall | candidate mass | rerank oracle32 recall | rerank mass | "
            "oracle mass | candidate "
            "union pages | rerank union pages | rerank union recall | rerank union "
            "mass | oracle union mass |",
            "|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            *_rerank_page_table_rows(payload["aggregate"]),
            *_rerank_page_table_rows(payload["records"]),
            "",
            "Token Top-k metrics use proxy logits for selection. The page-rerank table "
            "uses exact-QK only inside proxy-selected candidates. Oracle mass uses the "
            "true exact-QK Top-k on the identical query rows. This is a quality diagnostic, "
            "not a PCIe/runtime measurement.",
            "",
        ]
    )


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("dense-V to K probing requires CUDA")
    layers = _parse_ints(args.layers)
    budgets = _parse_ints(args.budgets)
    if not layers or not budgets or min(budgets) <= 0:
        raise ValueError("layers and positive budgets must be nonempty")
    if min(
        args.fit_windows,
        args.heldout_windows,
        args.sequence_length,
        args.batch_size,
        args.query_stride,
        args.torch_num_threads,
        args.page_size,
        args.page_token_budget,
        args.page_candidate_token_budget,
    ) <= 0:
        raise ValueError("window, sequence, batch, query, and thread sizes must be positive")
    fit_indices = set(range(args.fit_start, args.fit_start + args.fit_windows))
    heldout_indices = set(
        range(args.heldout_start, args.heldout_start + args.heldout_windows)
    )
    if fit_indices & heldout_indices:
        raise ValueError("fit and held-out windows must be disjoint")
    if args.query_start < max(
        max(budgets),
        args.page_token_budget,
        args.page_candidate_token_budget,
    ) - 1:
        raise ValueError("query start must expose at least the maximum Top-k budget")
    if args.relative_ridge < 0:
        raise ValueError("relative ridge must be nonnegative")

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
    if stored.ndim != 2 or int(stored.shape[1]) != args.sequence_length:
        raise ValueError("window bank geometry does not match the requested sequence")
    selected_indices = sorted(fit_indices) + sorted(heldout_indices)
    if max(selected_indices) >= len(stored):
        raise ValueError("requested windows exceed the stored bank")
    windows = stored.index_select(0, torch.tensor(selected_indices, dtype=torch.long))
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
    cos, sin = position_embeddings

    fit_count = args.fit_windows
    fit_slice = slice(0, fit_count)
    heldout_slice = slice(fit_count, fit_count + args.heldout_windows)
    query_positions = list(
        range(args.query_start, args.sequence_length, args.query_stride)
    )
    if query_positions[-1] != args.sequence_length - 1:
        query_positions.append(args.sequence_length - 1)

    records = []
    factor_artifacts: dict[str, Any] = {}
    requested_layers = set(layers)
    for layer_index, layer in enumerate(model.model.layers):
        if layer_index in requested_layers:
            encoder, _, c1_artifact = _load_layer_c1_factors(
                c1_dir, c1_result, layer_index, value_rank
            )
            features = _extract_layer_features(
                layer,
                hidden_bank,
                value_encoder=encoder,
                position_embeddings=position_embeddings,
                batch_size=args.batch_size,
            )
            dense_pre, dense_post = _fit_source_probes(
                features["dense_value"][fit_slice],
                features["pre_key"][fit_slice],
                features["post_key"][fit_slice],
                relative_ridge=args.relative_ridge,
            )
            c1_pre, c1_post = _fit_source_probes(
                features["c1_value"][fit_slice],
                features["pre_key"][fit_slice],
                features["post_key"][fit_slice],
                relative_ridge=args.relative_ridge,
            )
            c1_label = f"c1_v{value_rank}"
            probes = {
                "dense_v_pre_rope": ("dense_v128", features["dense_value"], dense_pre, True),
                "dense_v_direct_post": ("dense_v128", features["dense_value"], dense_post, False),
                f"{c1_label}_pre_rope": (c1_label, features["c1_value"], c1_pre, True),
                f"{c1_label}_direct_post": (c1_label, features["c1_value"], c1_post, False),
            }
            factor_path = output_dir / f"layer_{layer_index:03d}.safetensors"
            factor_tensors = {}
            for name, (_, _, probe, _) in probes.items():
                factor_tensors[f"{name}.weight"] = probe.weight.contiguous()
                factor_tensors[f"{name}.bias"] = probe.bias.contiguous()
            _atomic_safetensors(factor_path, factor_tensors)
            factor_artifacts[str(layer_index)] = {
                "file": factor_path.name,
                "sha256": _sha256(factor_path),
                "c1_factor_file": c1_artifact["file"],
                "c1_factor_sha256": c1_artifact["sha256"],
            }

            heldout_pre = features["pre_key"][heldout_slice]
            heldout_post = features["post_key"][heldout_slice]
            metric_state: dict[str, dict[str, Any]] = {}
            for name, (source_name, source, probe, rotate) in probes.items():
                source_heldout = source[heldout_slice]
                prediction = _prediction(
                    source_heldout,
                    probe,
                    rotate=rotate,
                    cos=cos.cpu(),
                    sin=sin.cpu(),
                )
                post_sums = _key_metric_sums(heldout_post, prediction)
                pre_sums = None
                if rotate:
                    predicted_pre = apply_headwise_linear_probe(source_heldout, probe)
                    pre_sums = _key_metric_sums(heldout_pre, predicted_pre)
                    del predicted_pre
                metric_state[name] = {
                    "source": source_name,
                    "target": "pre_rope_then_exact_rope" if rotate else "direct_post_rope",
                    "post_key_sums": post_sums,
                    "pre_key_sums": pre_sums,
                    "score_sums": _new_score_sums(budgets),
                }
                del prediction

            heldout_query = features["query"][heldout_slice]
            for example in range(args.heldout_windows):
                exact_key = heldout_post[example].to(device=device, dtype=torch.float32)
                query = heldout_query[example].to(device=device, dtype=torch.float32)
                proxy_keys = {}
                for name, (_, source, probe, rotate) in probes.items():
                    source_example = source[heldout_slice][example : example + 1].to(
                        device=device
                    )
                    proxy_keys[name] = _prediction(
                        source_example,
                        probe,
                        rotate=rotate,
                        cos=cos,
                        sin=sin,
                    )[0]
                for position in query_positions:
                    visible = position + 1
                    kv_head_index = torch.arange(
                        NUM_QUERY_HEADS, device=device
                    ) // HEADS_PER_GROUP
                    exact_repeated = exact_key.index_select(0, kv_head_index)[:, :visible]
                    query_row = query[:, position]
                    exact_scores = torch.einsum(
                        "hd,hld->hl", query_row, exact_repeated
                    ) / math.sqrt(HEAD_DIM)
                    for name, proxy_key in proxy_keys.items():
                        proxy_repeated = proxy_key.index_select(0, kv_head_index)[:, :visible]
                        proxy_scores = torch.einsum(
                            "hd,hld->hl", query_row, proxy_repeated
                        ) / math.sqrt(HEAD_DIM)
                        _accumulate_score_metrics(
                            metric_state[name]["score_sums"],
                            exact_scores=exact_scores,
                            proxy_scores=proxy_scores,
                            budgets=budgets,
                            page_size=args.page_size,
                            page_token_budget=args.page_token_budget,
                            page_candidate_token_budget=(
                                args.page_candidate_token_budget
                            ),
                        )
                print(
                    f"[dense-V K proxy] layer={layer_index} heldout="
                    f"{example + 1}/{args.heldout_windows}",
                    flush=True,
                )

            for name, state in metric_state.items():
                records.append(
                    _record(
                        layer=layer_index,
                        proxy=name,
                        source=state["source"],
                        target=state["target"],
                        post_key_sums=state["post_key_sums"],
                        pre_key_sums=state["pre_key_sums"],
                        score_sums=state["score_sums"],
                    )
                )
            del features, probes, metric_state, heldout_pre, heldout_post, heldout_query
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
            "probe": "per-physical-KV-head centered affine ridge regression",
            "source_controls": ["dense_v128", f"c1_v{value_rank}"],
            "targets": ["pre_rope_then_exact_token_rope", "direct_post_rope"],
            "score_centering": "subtract per-query mean over visible causal keys",
            "attention_kl": "KL(exact-QK teacher || proxy-QK)",
            "selection": "token Top-k from proxy logits",
            "page_selection": (
                "proxy page log-sum-exp candidates; exact-QK page log-sum-exp rerank; "
                "per-Query-head retained pages followed by physical-GQA union"
            ),
        },
        "model": {
            "path": str(model_path),
            "config_sha256": _sha256(model_path / "config.json"),
        },
        "windows": {
            "path": str(windows_path),
            "sha256": _sha256(windows_path),
            "manifest_sha256": _sha256(window_manifest_path),
            "fit_indices": sorted(fit_indices),
            "heldout_indices": sorted(heldout_indices),
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
        "factor_artifacts": factor_artifacts,
        "records": records,
        "aggregate": _aggregate(records, budgets),
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
    parser.add_argument("--fit-start", type=int, default=0)
    parser.add_argument("--fit-windows", type=int, default=8)
    parser.add_argument("--heldout-start", type=int, default=8)
    parser.add_argument("--heldout-windows", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--query-start", type=int, default=511)
    parser.add_argument("--query-stride", type=int, default=256)
    parser.add_argument("--budgets", default="64,256,512")
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument("--page-token-budget", type=int, default=512)
    parser.add_argument("--page-candidate-token-budget", type=int, default=512)
    parser.add_argument("--relative-ridge", type=float, default=1.0e-6)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--torch-num-threads", type=int, default=8)
    parser.add_argument("--model-dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


if __name__ == "__main__":
    evaluate(_parser().parse_args())
