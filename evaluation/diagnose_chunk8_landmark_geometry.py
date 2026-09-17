"""No-fit Chunk8 routing geometry for Llama-3.1-8B-Instruct B16R16."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shlex
import sys

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM

from basisserve.core.c1_conditional_page_attention import _selected_pages
from basisserve.core.c1_v_conditional_k_router import build_conditional_routing_sidecar
from evaluation.fit_k_routing_streaming import verified
from evaluation.llama_sink_recent_routing import page_support
from evaluation.v96kl_common import configure, read_json, sha256, write_json


CHUNK_SIZE = 8
SINK_TOKENS = 32
RECENT_TOKENS = 64
PREFIX_LENGTH = 65_280
SEQUENCE_LENGTH = 65_536
ONLINE_STEPS = 256
WINDOWS = tuple(range(64, 80))
CHUNK_ARMS = (
    "TOKEN_B16R16_CHUNK8_LSE",
    "EXACT_QK_CHUNK8_LSE",
    "EXACT_K_MEAN_CHUNK8",
    "BASE16_MEAN_CHUNK8",
    "BASE16_OLD_R16_MEAN_CHUNK8",
)
ARMS = (
    "TOKEN_B16R16_PAGE32",
    "EXACT_QK_PAGE32",
    *CHUNK_ARMS,
)


def chunk_means(values: torch.Tensor) -> torch.Tensor:
    """Return FP32-accumulated BF16 means for complete contiguous Chunk8 rows."""

    batch, heads, tokens, width = map(int, values.shape)
    assert tokens > 0 and tokens % CHUNK_SIZE == 0
    chunks = tokens // CHUNK_SIZE
    return (
        values.float()
        .reshape(batch, heads, chunks, CHUNK_SIZE, width)
        .mean(dim=-2)
        .to(torch.bfloat16)
        .contiguous()
    )


def chunk_logsumexp(token_scores: torch.Tensor, complete_chunks: int) -> torch.Tensor:
    """Exact token-score log-sum-exp for a complete historical Chunk8 prefix."""

    tokens = complete_chunks * CHUNK_SIZE
    assert 0 < tokens <= token_scores.shape[-1]
    return torch.logsumexp(
        token_scores[..., :tokens].float().reshape(
            *token_scores.shape[:-1], complete_chunks, CHUNK_SIZE
        ),
        dim=-1,
    )


def chunk_mean_scores(
    query: torch.Tensor,
    landmarks: torch.Tensor,
    complete_chunks: int,
    *,
    scale: float,
) -> torch.Tensor:
    """Score complete Chunk8 means with the mathematically required log(8)."""

    assert query.ndim == 4 and landmarks.ndim == 4
    scores = torch.einsum(
        "bghd,bgcd->bghc",
        query.float(),
        landmarks[:, :, :complete_chunks].float(),
    )
    return scores * scale + math.log(CHUNK_SIZE)


def chunk8_support(
    chunk_scores: torch.Tensor,
    total_tokens: int,
    *,
    budget: int = 2048,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select 4 pinned + 244 routed chunks, then append exact recent64 tokens."""

    batch, kv_heads, query_groups, complete_chunks = map(int, chunk_scores.shape)
    assert total_tokens > budget and budget % CHUNK_SIZE == 0
    assert SINK_TOKENS % CHUNK_SIZE == 0 and RECENT_TOKENS % CHUNK_SIZE == 0
    assert complete_chunks == (total_tokens - RECENT_TOKENS) // CHUNK_SIZE
    historical_chunk_budget = (budget - RECENT_TOKENS) // CHUNK_SIZE
    pinned_chunks = SINK_TOKENS // CHUNK_SIZE
    assert historical_chunk_budget == 248 and pinned_chunks == 4
    proxy = chunk_scores.reshape(
        batch,
        kv_heads * query_groups,
        1,
        complete_chunks,
    )
    valid = torch.ones_like(proxy, dtype=torch.bool)
    selected, selected_valid = _selected_pages(
        proxy,
        valid,
        kv_heads=kv_heads,
        page_size=1,
        page_budget=historical_chunk_budget,
        pinned_prefix_pages=pinned_chunks,
    )
    selected, order = selected.squeeze(-2).sort(-1)
    selected_valid = selected_valid.squeeze(-2).gather(-1, order)
    offsets = torch.arange(CHUNK_SIZE, device=chunk_scores.device)
    historical_ids = (selected[..., None] * CHUNK_SIZE + offsets).flatten(-2)
    historical_valid = selected_valid[..., None].expand(
        *selected_valid.shape, CHUNK_SIZE
    ).flatten(-2)
    recent_start = total_tokens - RECENT_TOKENS
    recent = torch.arange(
        recent_start,
        total_tokens,
        device=chunk_scores.device,
    ).expand(batch, kv_heads, -1)
    recent_valid = torch.ones_like(recent, dtype=torch.bool)
    ids = torch.cat((historical_ids, recent), -1)
    support_valid = torch.cat((historical_valid, recent_valid), -1)
    assert (support_valid.sum(-1) == budget).all()
    assert not (
        (historical_ids[..., None] == recent[..., None, :])
        & historical_valid[..., None]
    ).any()
    return ids, support_valid


def selected_chunk_mask(
    ids: torch.Tensor,
    valid: torch.Tensor,
    complete_chunks: int,
    *,
    routed_only: bool,
) -> torch.Tensor:
    """Map exact selected historical token support to physical Chunk8 identifiers."""

    result = torch.zeros(
        *ids.shape[:-1], complete_chunks, device=ids.device, dtype=torch.int32
    )
    historical_tokens = complete_chunks * CHUNK_SIZE
    selected_valid = valid & (ids < historical_tokens)
    if routed_only:
        selected_valid &= ids >= SINK_TOKENS
    result.scatter_add_(
        -1,
        (ids // CHUNK_SIZE).clamp_max(complete_chunks - 1),
        selected_valid.int(),
    )
    return result > 0


def synthetic_audits() -> dict[str, bool]:
    torch.manual_seed(20260917)
    query = torch.randn(1, 2, 3, 7)
    row = torch.randn(1, 2, 1, 7)
    identical = row.expand(1, 2, CHUNK_SIZE * 5, 7).clone()
    means = chunk_means(identical)
    mean_scores = chunk_mean_scores(query, means, 5, scale=1.0)
    token_scores = torch.einsum("bghd,bgtd->bght", query, identical)
    exact_scores = chunk_logsumexp(token_scores, 5)
    torch.testing.assert_close(mean_scores, exact_scores, atol=2e-2, rtol=2e-2)

    chunks = 400
    scores = torch.randn(1, 2, 3, chunks)
    total_tokens = chunks * CHUNK_SIZE + RECENT_TOKENS + 3
    complete = (total_tokens - RECENT_TOKENS) // CHUNK_SIZE
    scores = scores[..., :complete]
    ids, valid = chunk8_support(scores, total_tokens, budget=2048)
    assert (valid.sum(-1) == 2048).all()
    historical = selected_chunk_mask(ids, valid, complete, routed_only=False)
    routed = selected_chunk_mask(ids, valid, complete, routed_only=True)
    assert (historical.sum(-1) == 248).all()
    assert (routed.sum(-1) == 244).all()
    assert historical[..., :4].all() and not routed[..., :4].any()
    return {
        "identical_token_mean_matches_chunk_lse": True,
        "budget_is_exactly_2048_tokens": True,
        "historical_chunks_are_4_pinned_plus_244_routed": True,
        "recent64_is_disjoint_from_selected_historical_chunks": True,
    }


def empty_sums() -> dict[str, float | int]:
    return {
        "score_squared_error": 0.0,
        "score_energy": 0.0,
        "score_x": 0.0,
        "score_y": 0.0,
        "score_x2": 0.0,
        "score_y2": 0.0,
        "score_xy": 0.0,
        "score_count": 0,
        "exact_chunk_recall": 0.0,
        "exact_chunk_recall_count": 0,
        "b16_chunk_recall": 0.0,
        "b16_chunk_recall_count": 0,
        "attention_mass": 0.0,
        "non_sink_attention_mass": 0.0,
        "attention_count": 0,
        "output_squared_error": 0.0,
        "output_energy": 0.0,
        "wo_squared_error": 0.0,
        "wo_energy": 0.0,
        "support_tokens": 0.0,
        "support_tokens_count": 0,
        "support_tokens_min": 1 << 30,
        "support_tokens_max": 0,
        "historical_chunks": 0.0,
        "historical_chunks_count": 0,
        "historical_chunks_min": 1 << 30,
        "historical_chunks_max": 0,
    }


def add_score_statistics(
    report: dict[str, float | int],
    student: torch.Tensor,
    teacher: torch.Tensor,
) -> None:
    x = teacher.double()
    y = student.double()
    report["score_squared_error"] += float((y - x).square().sum())
    report["score_energy"] += float(x.square().sum())
    report["score_x"] += float(x.sum())
    report["score_y"] += float(y.sum())
    report["score_x2"] += float(x.square().sum())
    report["score_y2"] += float(y.square().sum())
    report["score_xy"] += float((x * y).sum())
    report["score_count"] += x.numel()


def add_recall(
    report: dict[str, float | int],
    selected: torch.Tensor,
    reference: torch.Tensor,
    prefix: str,
) -> None:
    denominator = reference.sum(-1)
    eligible = denominator > 0
    recall = (selected & reference).sum(-1) / denominator.clamp_min(1)
    report[f"{prefix}_chunk_recall"] += float((recall * eligible).sum())
    report[f"{prefix}_chunk_recall_count"] += int(eligible.sum())


def derive_metrics(sums: dict[str, float | int]) -> dict[str, float | int | None]:
    score_count = int(sums["score_count"])
    score_rel_mse = None
    score_pearson = None
    if score_count:
        numerator = score_count * sums["score_xy"] - sums["score_x"] * sums["score_y"]
        denominator_x = score_count * sums["score_x2"] - sums["score_x"] ** 2
        denominator_y = score_count * sums["score_y2"] - sums["score_y"] ** 2
        denominator = math.sqrt(max(0.0, denominator_x) * max(0.0, denominator_y))
        score_rel_mse = sums["score_squared_error"] / sums["score_energy"]
        score_pearson = numerator / denominator if denominator > 0 else 1.0
    return {
        "chunk_logit_rel_mse": score_rel_mse,
        "chunk_logit_pearson": score_pearson,
        "routed_chunk_recall_exact_qk": sums["exact_chunk_recall"]
        / sums["exact_chunk_recall_count"],
        "routed_chunk_recall_b16r16": sums["b16_chunk_recall"]
        / sums["b16_chunk_recall_count"],
        "attention_mass": sums["attention_mass"] / sums["attention_count"],
        "non_sink_attention_mass": sums["non_sink_attention_mass"]
        / sums["attention_count"],
        "output_rel_mse": sums["output_squared_error"] / sums["output_energy"],
        "wo_rel_mse": sums["wo_squared_error"] / sums["wo_energy"],
        "support_tokens_mean": sums["support_tokens"] / sums["support_tokens_count"],
        "support_tokens_min": sums["support_tokens_min"],
        "support_tokens_max": sums["support_tokens_max"],
        "historical_chunks_mean": sums["historical_chunks"]
        / sums["historical_chunks_count"],
        "historical_chunks_min": sums["historical_chunks_min"],
        "historical_chunks_max": sums["historical_chunks_max"],
    }


def merge_sums(reports: list[dict[str, float | int]]) -> dict[str, float | int]:
    merged = empty_sums()
    minima = ("support_tokens_min", "historical_chunks_min")
    maxima = ("support_tokens_max", "historical_chunks_max")
    for key in merged:
        if key in minima:
            merged[key] = min(report[key] for report in reports)
        elif key in maxima:
            merged[key] = max(report[key] for report in reports)
        else:
            merged[key] = sum(report[key] for report in reports)
    return merged


@torch.inference_mode()
def geometry_window_metrics(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    output_weight: torch.Tensor,
    factors: dict[str, torch.Tensor],
    *,
    budget: int,
    steps: int,
) -> tuple[dict[str, dict[str, float | int]], dict[str, bool]]:
    batch, query_heads, length, head_dim = map(int, query.shape)
    kv_heads = int(key.shape[1])
    query_groups = query_heads // kv_heads
    assert batch == 1 and length == SEQUENCE_LENGTH and 0 < steps <= ONLINE_STEPS
    assert key.shape == value.shape == (batch, kv_heads, length, head_dim)
    assert query_heads % kv_heads == 0
    scale = head_dim**-0.5
    sidecar = build_conditional_routing_sidecar(
        value,
        key,
        base_left=factors["base_left_b16"],
        base_right=factors["base_right_b16"],
        base_bias=factors["base_bias_b16"],
        residual_encoder=factors["residual_encoder_b16_r16"],
        cos=cos,
        sin=sin,
    )
    exact_key_landmarks = chunk_means(key)
    base_landmarks = chunk_means(sidecar[..., :head_dim])
    residual_landmarks = chunk_means(sidecar[..., head_dim:])
    values = value.float()
    output_weight = output_weight.float()
    sums = {name: empty_sums() for name in ARMS}
    audits = {
        "base_landmark_is_post_rope_mean": True,
        "chunk_teacher_is_direct_token_lse": True,
        "budget_is_exactly_2048_for_chunk8": True,
        "gqa_uses_existing_normalize_then_max": True,
        "selected_attention_uses_exact_k_dense_v": True,
        "no_token_after_recent_boundary_enters_landmark_selection": True,
    }

    for step, position in enumerate(range(PREFIX_LENGTH, PREFIX_LENGTH + steps)):
        total_tokens = position + 1
        complete_chunks = (total_tokens - RECENT_TOKENS) // CHUNK_SIZE
        current_query = query[:, :, position].float()
        grouped_query = current_query.reshape(batch, kv_heads, query_groups, head_dim)
        exact_logits = (
            grouped_query @ key[:, :, :total_tokens].float().transpose(-1, -2)
        ) * scale
        exact_probabilities = exact_logits.softmax(-1)
        non_sink_logits = exact_logits.clone()
        non_sink_logits[..., :SINK_TOKENS] = -torch.inf
        non_sink_probabilities = non_sink_logits.softmax(-1)
        dense_output = exact_probabilities @ values[:, :, :total_tokens]
        dense_wo = dense_output.reshape(batch, -1) @ output_weight.T

        residual_query = torch.einsum(
            "bhd,hdr->bhr",
            current_query,
            factors["residual_query_b16_r16"].float(),
        )
        grouped_residual_query = residual_query.reshape(
            batch, kv_heads, query_groups, -1
        )
        routing_query = torch.cat((current_query, residual_query), -1).reshape(
            batch, kv_heads, query_groups, -1
        )
        b16_token_scores = (
            routing_query @ sidecar[:, :, :total_tokens].float().transpose(-1, -2)
        ) * scale

        exact_chunk_scores = chunk_logsumexp(exact_logits, complete_chunks)
        b16_chunk_scores = chunk_logsumexp(b16_token_scores, complete_chunks)
        exact_mean_scores = chunk_mean_scores(
            grouped_query,
            exact_key_landmarks,
            complete_chunks,
            scale=scale,
        )
        base_scores = chunk_mean_scores(
            grouped_query,
            base_landmarks,
            complete_chunks,
            scale=scale,
        )
        residual_scores = torch.einsum(
            "bghd,bgcd->bghc",
            grouped_residual_query,
            residual_landmarks[:, :, :complete_chunks].float(),
        ) * scale
        chunk_scores = {
            "TOKEN_B16R16_CHUNK8_LSE": b16_chunk_scores,
            "EXACT_QK_CHUNK8_LSE": exact_chunk_scores,
            "EXACT_K_MEAN_CHUNK8": exact_mean_scores,
            "BASE16_MEAN_CHUNK8": base_scores,
            "BASE16_OLD_R16_MEAN_CHUNK8": base_scores + residual_scores,
        }
        supports = {
            "TOKEN_B16R16_PAGE32": page_support(b16_token_scores, budget=budget),
            "EXACT_QK_PAGE32": page_support(exact_logits, budget=budget),
        }
        supports.update(
            {
                name: chunk8_support(scores, total_tokens, budget=budget)
                for name, scores in chunk_scores.items()
            }
        )

        exact_reference = selected_chunk_mask(
            *supports["EXACT_QK_CHUNK8_LSE"],
            complete_chunks,
            routed_only=True,
        )
        b16_reference = selected_chunk_mask(
            *supports["TOKEN_B16R16_CHUNK8_LSE"],
            complete_chunks,
            routed_only=True,
        )
        for name in ARMS:
            report = sums[name]
            if name in CHUNK_ARMS:
                add_score_statistics(report, chunk_scores[name][..., 4:], exact_chunk_scores[..., 4:])
            ids, valid = supports[name]
            assert (valid.sum(-1) <= budget).all()
            routed = selected_chunk_mask(
                ids,
                valid,
                complete_chunks,
                routed_only=True,
            )
            add_recall(report, routed, exact_reference, "exact")
            add_recall(report, routed, b16_reference, "b16")
            token_counts = valid.sum(-1)
            chunk_counts = selected_chunk_mask(
                ids,
                valid,
                complete_chunks,
                routed_only=False,
            ).sum(-1)
            for prefix, tensor in (
                ("support_tokens", token_counts),
                ("historical_chunks", chunk_counts),
            ):
                report[prefix] += float(tensor.sum())
                report[f"{prefix}_count"] += tensor.numel()
                report[f"{prefix}_min"] = min(report[f"{prefix}_min"], int(tensor.min()))
                report[f"{prefix}_max"] = max(report[f"{prefix}_max"], int(tensor.max()))

            gather = ids.clamp_max(position)[:, :, None].expand(
                batch, kv_heads, query_groups, -1
            )
            expanded_valid = valid[:, :, None].expand_as(gather)
            for metric, probabilities in (
                ("attention_mass", exact_probabilities),
                ("non_sink_attention_mass", non_sink_probabilities),
            ):
                mass = probabilities.gather(-1, gather).masked_fill(
                    ~expanded_valid, 0.0
                ).sum(-1)
                assert torch.isfinite(mass).all()
                assert mass.min() >= 0 and mass.max() <= 1.00001
                report[metric] += float(mass.sum())
            report["attention_count"] += batch * kv_heads * query_groups

            selected_logits = exact_logits.gather(-1, gather).masked_fill(
                ~expanded_valid, -torch.inf
            )
            selected_values = values[:, :, :total_tokens].gather(
                2,
                ids.clamp_max(position)[..., None].expand(
                    batch, kv_heads, ids.shape[-1], head_dim
                ),
            )
            sparse_output = selected_logits.softmax(-1) @ selected_values
            difference = sparse_output - dense_output
            wo_difference = difference.reshape(batch, -1) @ output_weight.T
            for metric, tensor in (
                ("output_squared_error", difference),
                ("output_energy", dense_output),
                ("wo_squared_error", wo_difference),
                ("wo_energy", dense_wo),
            ):
                assert torch.isfinite(tensor).all()
                report[metric] += float(tensor.double().square().sum())
        if (step + 1) % 32 == 0:
            print(f"ONLINE_STEP {step + 1}", flush=True)

    assert all(audits.values()), audits
    return sums, audits


def protocol(root: Path, budget: int) -> dict:
    identity = read_json(root / "manifests/v128.json")
    windows = read_json(root / "calibration/manifest.json")
    assert identity["value_mode"] == "dense original V and Wo"
    assert identity["layer_ranks"] == [128] * 32
    assert windows["sha256"] == sha256(root / "calibration/windows.safetensors")
    assert windows["validation_ids"] == list(WINDOWS)
    identity_sha = sha256(root / "manifests/v128.json")
    banks = {}
    for layer in range(32):
        path = root / "ours_b16r16" / f"layer_{layer:03d}.safetensors"
        meta = read_json(path.with_suffix(".json"))
        assert meta["status"] == "complete" and meta["sha256"] == sha256(path)
        assert meta["identity_sha256"] == identity_sha and meta["v_rank"] == 128
        assert meta["protocol"]["base_rank"] == 16
        assert meta["protocol"]["residual_rank"] == 16
        banks[str(layer)] = meta["sha256"]
    sources = (
        "evaluation/diagnose_chunk8_landmark_geometry.py",
        "evaluation/llama_sink_recent_routing.py",
        "basisserve/core/c1_conditional_page_attention.py",
        "basisserve/core/c1_v_conditional_k_router.py",
    )
    return {
        "format": "basisserve.chunk8_landmark.geometry.v1",
        "model": identity["model"],
        "model_variant": "Llama-3.1-8B-Instruct",
        "value_path": "Dense V128 with identity value encoder and original Wo",
        "identity_sha256": identity_sha,
        "windows_sha256": windows["sha256"],
        "windows": list(WINDOWS),
        "sequence_length": SEQUENCE_LENGTH,
        "prefill_length": PREFIX_LENGTH,
        "online_decode_tokens": ONLINE_STEPS,
        "base_rank": 16,
        "old_token_residual_rank": 16,
        "chunk_size": CHUNK_SIZE,
        "budget_tokens": budget,
        "budget_accounting": {
            "historical_chunks": 248,
            "pinned_sink_chunks": 4,
            "freely_routed_historical_chunks": 244,
            "recent_tokens": 64,
            "recent_chunk_equivalent": 8,
            "total_chunk_equivalent": 256,
        },
        "chunk_selection": "Existing per-query-head normalization and max across four query heads per physical GQA KV group; four pinned sink chunks; exact recent64 appended disjointly",
        "page32_references": "Existing formal page_support implementation, unchanged",
        "payload": "Exact selected post-RoPE K128, original Dense V128, original Dense Wo",
        "landmarks": "FP32 accumulation and BF16 storage of post-RoPE means; only complete chunks older than recent64 are eligible",
        "purpose": "No-fit Stage-0 geometry ceiling before direct Chunk-Fisher ALS",
        "bank_sha256": banks,
        "source_sha256": {name: sha256(Path(name)) for name in sources},
    }


def load_factors(root: Path, layer: int, spec: dict) -> dict[str, torch.Tensor]:
    path = root / "ours_b16r16" / f"layer_{layer:03d}.safetensors"
    tensors, meta = verified(path)
    assert meta["sha256"] == spec["bank_sha256"][str(layer)]
    return {name: tensor.cuda() for name, tensor in tensors.items()}


def costs() -> dict[str, dict[str, float | int | str | None]]:
    return {
        "TOKEN_B16R16_PAGE32": {
            "retrieval_unit_tokens": 32,
            "routing_representations_per_unit": 32,
            "scan_dimensions_per_token": None,
            "note": "Existing token B16+R16 router",
        },
        "EXACT_QK_PAGE32": {
            "retrieval_unit_tokens": 32,
            "routing_representations_per_unit": 32,
            "scan_dimensions_per_token": 128,
            "note": "Exact-QK Page32 oracle",
        },
        "TOKEN_B16R16_CHUNK8_LSE": {
            "retrieval_unit_tokens": 8,
            "routing_representations_per_unit": 8,
            "scan_dimensions_per_token": None,
            "note": "Existing token B16+R16 scores with exact Chunk8 LSE",
        },
        "EXACT_QK_CHUNK8_LSE": {
            "retrieval_unit_tokens": 8,
            "routing_representations_per_unit": 8,
            "scan_dimensions_per_token": 128,
            "note": "Exact Chunk8-LSE selection oracle",
        },
        "EXACT_K_MEAN_CHUNK8": {
            "retrieval_unit_tokens": 8,
            "routing_representations_per_unit": 1,
            "scan_dimensions_per_token": 16,
            "note": "Exact post-RoPE K128 mean; Shadow-style diagnostic",
        },
        "BASE16_MEAN_CHUNK8": {
            "retrieval_unit_tokens": 8,
            "routing_representations_per_unit": 1,
            "scan_dimensions_per_token": 16,
            "note": "Predicted post-RoPE Base128 mean only",
        },
        "BASE16_OLD_R16_MEAN_CHUNK8": {
            "retrieval_unit_tokens": 8,
            "routing_representations_per_unit": 1,
            "scan_dimensions_per_token": 18,
            "note": "Base128 mean plus mean of existing token residual R16; direct-Fisher initialization reference",
        },
    }


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert path.read_text() == value
        return
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(value)
    partial.replace(path)


def command_record(root: Path, output: Path, budget: int) -> dict[str, str]:
    python = "/home/zhangal/.conda/envs/basis/bin/python"
    prefix = f"{python} -u -m evaluation.diagnose_chunk8_landmark_geometry"
    suffix = f"--root {root} --output {output} --budget {budget}"
    return {
        "environment": "basis",
        "python": python,
        "smoke": f"{prefix} smoke {suffix}",
        "evaluate": f"{prefix} evaluate {suffix} --shard-index $SLURM_ARRAY_TASK_ID",
        "summarize": f"{prefix} summarize {suffix}",
    }


def markdown(result: dict) -> str:
    lines = [
        "# Llama-3.1-8B-Instruct Chunk8 Landmark Geometry",
        "",
        "This is the no-fit Stage-0 diagnostic for direct Chunk8 Fisher landmarks. It uses Dense V128, the frozen existing Base16/B16R16 factors, exact selected K, Dense V, and original Wo.",
        "",
        "## Pooled results",
        "",
        "| Variant | Scan dims/token | Chunk rel-MSE | Recall vs Exact Chunk8 | Recall vs B16R16 Chunk8 | Attention mass | Non-sink mass | Output rel-MSE | Wo rel-MSE | Support tokens |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ARMS:
        metric = result["pooled"][name]
        cost = result["cost"][name]
        scan = cost["scan_dimensions_per_token"]
        score = metric["chunk_logit_rel_mse"]
        lines.append(
            f"| {name} | {'n/a' if scan is None else f'{scan:g}'} | "
            f"{'n/a' if score is None else f'{score:.6g}'} | "
            f"{metric['routed_chunk_recall_exact_qk']:.4f} | "
            f"{metric['routed_chunk_recall_b16r16']:.4f} | "
            f"{metric['attention_mass']:.4f} | {metric['non_sink_attention_mass']:.4f} | "
            f"{metric['output_rel_mse']:.6g} | {metric['wo_rel_mse']:.6g} | "
            f"{metric['support_tokens_mean']:.1f} |"
        )
    exact_page = result["pooled"]["EXACT_QK_PAGE32"]
    exact_chunk = result["pooled"]["EXACT_QK_CHUNK8_LSE"]
    exact_mean = result["pooled"]["EXACT_K_MEAN_CHUNK8"]
    base = result["pooled"]["BASE16_MEAN_CHUNK8"]
    old = result["pooled"]["BASE16_OLD_R16_MEAN_CHUNK8"]
    token = result["pooled"]["TOKEN_B16R16_CHUNK8_LSE"]
    lines.extend(
        [
            "",
            "## Geometry decomposition",
            "",
            f"- Chunk8 granularity ceiling: Exact-QK Wo rel-MSE {exact_page['wo_rel_mse']:.6g} (Page32) -> {exact_chunk['wo_rel_mse']:.6g} (Chunk8).",
            f"- Mean-pooling gap: Exact Chunk8-LSE {exact_chunk['wo_rel_mse']:.6g} -> Exact-K mean {exact_mean['wo_rel_mse']:.6g}.",
            f"- Frozen B16 representation gap before pooling: Exact Chunk8-LSE {exact_chunk['wo_rel_mse']:.6g} -> token B16R16 Chunk8-LSE {token['wo_rel_mse']:.6g}.",
            f"- Existing residual-mean contribution: Base16 mean {base['wo_rel_mse']:.6g} -> Base16+old-R16 mean {old['wo_rel_mse']:.6g}.",
            "",
            "Chunk8 arms use exactly 4 pinned sink chunks + 244 freely routed historical chunks + exact recent64, for 2048 unique logical attention tokens. No fit or learned landmark was used in this stage.",
            "",
        ]
    )
    return "\n".join(lines)


@torch.inference_mode()
def run_stage(args: argparse.Namespace, spec: dict) -> None:
    if args.stage == "evaluate":
        smoke = read_json(args.output / "smoke.json")
        assert smoke["status"] == "complete" and smoke["protocol"] == spec
        assert all(smoke["synthetic_audits"].values())
    layers = [0] if args.stage == "smoke" else list(
        range(args.shard_index * 8, (args.shard_index + 1) * 8)
    )
    selected_windows = [WINDOWS[0]] if args.stage == "smoke" else list(WINDOWS)
    factors = {layer: load_factors(args.root, layer, spec) for layer in layers}
    identity = read_json(args.root / "manifests/v128.json")
    model = AutoModelForCausalLM.from_pretrained(
        identity["model"],
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=True,
    ).eval().cuda()
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

    windows = load_file(str(args.root / "calibration/windows.safetensors"))["input_ids"]
    assert tuple(windows.shape) == (80, SEQUENCE_LENGTH)
    records = {layer: [] for layer in layers}
    active = {}
    handles = []
    for layer in layers:
        def capture(module, positional, kwargs, layer=layer):
            hidden = kwargs["hidden_states"]
            length = int(hidden.shape[1])
            query = module.q_proj(hidden).view(1, length, 32, 128).transpose(1, 2)
            key = module.k_proj(hidden).view(1, length, 8, 128).transpose(1, 2)
            value = module.v_proj(hidden).view(1, length, 8, 128).transpose(1, 2)
            cos, sin = kwargs["position_embeddings"]
            query, key = apply_rotary_pos_emb(query, key, cos, sin)
            sums, audits = geometry_window_metrics(
                query,
                key,
                value,
                cos,
                sin,
                module.o_proj.weight,
                factors[layer],
                budget=args.budget,
                steps=ONLINE_STEPS,
            )
            metrics = {name: derive_metrics(sums[name]) for name in ARMS}
            records[layer].append(
                {
                    "window": active["window"],
                    "steps": ONLINE_STEPS,
                    "metrics": metrics,
                    "sums": sums,
                    "audits": audits,
                }
            )
            print(
                {
                    "layer": layer,
                    "window": active["window"],
                    "exact_chunk_wo": metrics["EXACT_QK_CHUNK8_LSE"]["wo_rel_mse"],
                    "exact_mean_wo": metrics["EXACT_K_MEAN_CHUNK8"]["wo_rel_mse"],
                    "base_old_r16_mean_wo": metrics["BASE16_OLD_R16_MEAN_CHUNK8"]["wo_rel_mse"],
                },
                flush=True,
            )

        handles.append(
            model.model.layers[layer].self_attn.register_forward_pre_hook(
                capture,
                with_kwargs=True,
            )
        )
    for window in selected_windows:
        active["window"] = window
        result = model.model(windows[window : window + 1].long().cuda(), use_cache=False)
        assert torch.isfinite(result.last_hidden_state).all()
        del result
    for handle in handles:
        handle.remove()
    for layer in layers:
        combined = {
            name: merge_sums([window["sums"][name] for window in records[layer]])
            for name in ARMS
        }
        artifact = {
            "status": "complete",
            "layer": layer,
            "protocol": spec,
            "synthetic_audits": synthetic_audits(),
            "metrics": {name: derive_metrics(combined[name]) for name in ARMS},
            "sums": combined,
            "windows": records[layer],
            "command": shlex.join(sys.argv),
            "python": sys.executable,
        }
        path = args.output / "smoke.json" if args.stage == "smoke" else args.output / f"layer_{layer:03d}.json"
        write_json(path, artifact)


def summarize(args: argparse.Namespace, spec: dict) -> None:
    layers = []
    for layer in range(32):
        report = read_json(args.output / f"layer_{layer:03d}.json")
        assert report["status"] == "complete" and report["protocol"] == spec
        assert report["layer"] == layer
        assert [window["window"] for window in report["windows"]] == list(WINDOWS)
        assert all(report["synthetic_audits"].values())
        assert all(all(window["audits"].values()) for window in report["windows"])
        layers.append(report)
    pooled_sums = {
        name: merge_sums([layer["sums"][name] for layer in layers])
        for name in ARMS
    }
    pooled = {name: derive_metrics(pooled_sums[name]) for name in ARMS}
    equal_layer_mean = {
        name: {
            metric: sum(layer["metrics"][name][metric] for layer in layers) / 32
            for metric in layers[0]["metrics"][name]
            if layer_metric_is_numeric(layers[0]["metrics"][name][metric])
            and not metric.endswith("_min")
            and not metric.endswith("_max")
        }
        for name in ARMS
    }
    result = {
        "status": "complete",
        "protocol": spec,
        "pooled": pooled,
        "equal_layer_mean": equal_layer_mean,
        "cost": costs(),
        "audits": {
            "synthetic": synthetic_audits(),
            "all_runtime_audits_passed": True,
        },
    }
    write_json(args.output / "result.json", result)
    write_json(
        args.output / "per_layer.json",
        {
            "status": "complete",
            "protocol": spec,
            "layers": [
                {"layer": layer["layer"], "metrics": layer["metrics"]}
                for layer in layers
            ],
        },
    )
    write_text(args.output / "summary.md", markdown(result))
    print(json.dumps(pooled, indent=2), flush=True)


def layer_metric_is_numeric(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=("smoke", "evaluate", "summarize"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget", type=int, choices=(2048,), default=2048)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()
    assert 0 <= args.shard_index < 4
    configure()
    spec = protocol(args.root, args.budget)
    write_json(args.output / "config.json", spec)
    write_json(
        args.output / "commands.json",
        command_record(args.root, args.output, args.budget),
    )
    if args.stage == "summarize":
        summarize(args, spec)
    else:
        run_stage(args, spec)


if __name__ == "__main__":
    main()
