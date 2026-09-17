"""Parameter-free Page32 landmark diagnostic for the fitted Llama B16R16 router."""

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

from basisserve.core.c1_v_conditional_k_router import (
    build_conditional_routing_sidecar,
)
from evaluation.fit_k_routing_streaming import verified
from evaluation.llama_sink_recent_routing import page_support
from evaluation.v96kl_common import configure, read_json, sha256, write_json


PAGE_SIZE = 32
RECENT_TOKENS = 64
PREFIX_LENGTH = 65_280
SEQUENCE_LENGTH = 65_536
ONLINE_STEPS = 256
WINDOWS = tuple(range(64, 80))
LANDMARK_CHUNKS = {
    "LM32x1": 32,
    "LM16x2": 16,
    "LM8x4": 8,
    "LM4x8": 4,
}
ARMS = (
    "TOKEN_TEACHER",
    *LANDMARK_CHUNKS,
    "EXACT_K_LM8x4",
    "EXACT_QK",
)


def build_subpage_landmarks(
    token_state: torch.Tensor,
    subpage_size: int,
    *,
    storage_dtype: torch.dtype | None = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pool contiguous token-side states within fixed physical Page32 pages."""

    batch, heads, tokens, width = map(int, token_state.shape)
    assert tokens > 0 and PAGE_SIZE % subpage_size == 0
    pages = math.ceil(tokens / PAGE_SIZE)
    landmarks_per_page = PAGE_SIZE // subpage_size
    padded_tokens = pages * PAGE_SIZE
    padding = padded_tokens - tokens
    state = token_state.float()
    if padding:
        state = torch.nn.functional.pad(state, (0, 0, 0, padding))
    state = state.reshape(
        batch,
        heads,
        pages,
        landmarks_per_page,
        subpage_size,
        width,
    )
    token_indices = torch.arange(padded_tokens, device=token_state.device)
    valid = (token_indices < tokens).reshape(
        pages,
        landmarks_per_page,
        subpage_size,
    )
    counts = valid.sum(dim=-1)
    values = state.sum(dim=-2) / counts.clamp_min(1)[None, None, :, :, None]
    values.masked_fill_(~(counts > 0)[None, None, :, :, None], 0.0)
    if storage_dtype is not None:
        values = values.to(storage_dtype)
    return values.contiguous(), counts


def score_landmark_pages(
    query: torch.Tensor,
    landmarks: torch.Tensor,
    counts: torch.Tensor,
    *,
    scale: float,
) -> torch.Tensor:
    """Convert landmark scores and represented-token counts to Page32 LSEs."""

    batch, kv_heads, query_groups, width = map(int, query.shape)
    lm_batch, lm_heads, pages, landmarks_per_page, lm_width = map(
        int, landmarks.shape
    )
    assert batch == lm_batch and kv_heads == lm_heads and width == lm_width
    assert tuple(counts.shape) == (pages, landmarks_per_page)
    scores = torch.einsum(
        "bghd,bgpld->bghpl",
        query.float(),
        landmarks.float(),
    ) * scale
    weights = counts.to(device=scores.device, dtype=scores.dtype)
    weighted = scores + weights.log()[None, None, None]
    weighted.masked_fill_(~(weights > 0)[None, None, None], -torch.inf)
    return torch.logsumexp(weighted, dim=-1)


def prefix_landmark_page_scores(
    query: torch.Tensor,
    token_state: torch.Tensor,
    full_landmarks: torch.Tensor,
    full_counts: torch.Tensor,
    historical_tokens: int,
    subpage_size: int,
    *,
    scale: float,
) -> torch.Tensor:
    """Score cached landmarks without using tokens beyond the historical prefix."""

    assert 0 < historical_tokens <= token_state.shape[2]
    pages = math.ceil(historical_tokens / PAGE_SIZE)
    scores = score_landmark_pages(
        query,
        full_landmarks[:, :, :pages],
        full_counts[:pages],
        scale=scale,
    )
    remainder = historical_tokens % PAGE_SIZE
    if remainder:
        start = (pages - 1) * PAGE_SIZE
        partial, counts = build_subpage_landmarks(
            token_state[:, :, start:historical_tokens],
            subpage_size,
            storage_dtype=full_landmarks.dtype,
        )
        partial_scores = score_landmark_pages(
            query,
            partial,
            counts,
            scale=scale,
        )
        scores[..., -1:] = partial_scores
    return scores


def token_page_scores(scores: torch.Tensor, historical_tokens: int) -> torch.Tensor:
    """Exact Page32 token log-sum-exp, including a valid ragged final page."""

    assert 0 < historical_tokens <= scores.shape[-1]
    pages = math.ceil(historical_tokens / PAGE_SIZE)
    padding = pages * PAGE_SIZE - historical_tokens
    selected = scores[..., :historical_tokens].float()
    if padding:
        selected = torch.nn.functional.pad(selected, (0, padding), value=-torch.inf)
    return torch.logsumexp(selected.reshape(*selected.shape[:-1], pages, PAGE_SIZE), -1)


def page_scores_to_token_proxy(
    page_scores: torch.Tensor,
    historical_tokens: int,
    total_tokens: int,
) -> torch.Tensor:
    """Expand Page32 LSEs so the existing selector reconstructs them exactly."""

    pages = math.ceil(historical_tokens / PAGE_SIZE)
    assert page_scores.shape[-1] == pages and total_tokens >= historical_tokens
    page_indices = torch.arange(pages, device=page_scores.device)
    counts = (historical_tokens - page_indices * PAGE_SIZE).clamp(0, PAGE_SIZE)
    per_token = page_scores - counts.to(page_scores.dtype).log()
    historical = per_token.repeat_interleave(PAGE_SIZE, dim=-1)[..., :historical_tokens]
    if total_tokens == historical_tokens:
        return historical
    recent = torch.zeros(
        *historical.shape[:-1],
        total_tokens - historical_tokens,
        device=historical.device,
        dtype=historical.dtype,
    )
    return torch.cat((historical, recent), dim=-1)


def selected_page_mask(
    ids: torch.Tensor,
    valid: torch.Tensor,
    historical_tokens: int,
    *,
    routed_only: bool,
) -> torch.Tensor:
    """Represent selected historical physical pages as a Boolean mask."""

    pages = math.ceil(historical_tokens / PAGE_SIZE)
    selected = torch.zeros(
        *ids.shape[:-1], pages, device=ids.device, dtype=torch.int32
    )
    selected_valid = valid & (ids < historical_tokens)
    if routed_only:
        selected_valid &= ids >= PAGE_SIZE
    selected.scatter_add_(
        -1,
        (ids // PAGE_SIZE).clamp_max(pages - 1),
        selected_valid.int(),
    )
    return selected > 0


def synthetic_audits() -> dict[str, bool]:
    """Small deterministic checks for weighted LSE and ragged page handling."""

    torch.manual_seed(20260916)
    query = torch.randn(1, 2, 3, 7)
    identical = torch.randn(1, 2, 1, 7).expand(1, 2, 32, 7).clone()
    token_scores = torch.einsum("bghd,bgtd->bght", query, identical)
    teacher = token_page_scores(token_scores, 32)
    results = {}
    for chunk in (32, 16, 8, 4):
        landmarks, counts = build_subpage_landmarks(
            identical,
            chunk,
            storage_dtype=None,
        )
        student = score_landmark_pages(query, landmarks, counts, scale=1.0)
        torch.testing.assert_close(student, teacher, atol=2e-6, rtol=2e-6)
        results[f"identical_lm{chunk}"] = True

    ragged_parts = []
    for value, count in ((-0.75, 8), (0.5, 8), (1.25, 8), (-1.5, 5)):
        ragged_parts.append(torch.full((1, 1, count, 1), value))
    ragged = torch.cat(ragged_parts, dim=2)
    one_query = torch.ones(1, 1, 1, 1)
    direct = token_page_scores(ragged[:, :, None, :, 0], 29)
    landmarks, counts = build_subpage_landmarks(ragged, 8, storage_dtype=None)
    weighted = score_landmark_pages(one_query, landmarks, counts, scale=1.0)
    torch.testing.assert_close(weighted, direct, atol=2e-6, rtol=2e-6)
    results["ragged_weighted_lse"] = True

    proxy = page_scores_to_token_proxy(direct, 29, 93)
    rebuilt = token_page_scores(proxy, 29)
    torch.testing.assert_close(rebuilt, direct, atol=2e-6, rtol=2e-6)
    results["page_proxy_roundtrip"] = True
    return results


def empty_sums() -> dict[str, float | int]:
    return {
        "page_squared_error": 0.0,
        "page_energy": 0.0,
        "correlation_x": 0.0,
        "correlation_y": 0.0,
        "correlation_x2": 0.0,
        "correlation_y2": 0.0,
        "correlation_xy": 0.0,
        "correlation_count": 0,
        "teacher_page_recall": 0.0,
        "teacher_page_recall_count": 0,
        "teacher_routed_page_recall": 0.0,
        "teacher_routed_page_recall_count": 0,
        "oracle_page_recall": 0.0,
        "oracle_page_recall_count": 0,
        "oracle_routed_page_recall": 0.0,
        "oracle_routed_page_recall_count": 0,
        "attention_mass": 0.0,
        "non_sink_attention_mass": 0.0,
        "attention_count": 0,
        "output_squared_error": 0.0,
        "output_energy": 0.0,
        "wo_squared_error": 0.0,
        "wo_energy": 0.0,
        "physical_pages": 0.0,
        "physical_pages_count": 0,
        "physical_pages_min": 1 << 30,
        "physical_pages_max": 0,
        "physical_tokens": 0.0,
        "physical_tokens_count": 0,
        "physical_tokens_min": 1 << 30,
        "physical_tokens_max": 0,
    }


def add_tensor_statistics(
    report: dict[str, float | int],
    student: torch.Tensor,
    teacher: torch.Tensor,
) -> None:
    x = teacher.double()
    y = student.double()
    report["page_squared_error"] += float((y - x).square().sum())
    report["page_energy"] += float(x.square().sum())
    report["correlation_x"] += float(x.sum())
    report["correlation_y"] += float(y.sum())
    report["correlation_x2"] += float(x.square().sum())
    report["correlation_y2"] += float(y.square().sum())
    report["correlation_xy"] += float((x * y).sum())
    report["correlation_count"] += x.numel()


def add_recall(
    report: dict[str, float | int],
    selected: torch.Tensor,
    reference: torch.Tensor,
    prefix: str,
) -> None:
    denominator = reference.sum(dim=-1)
    eligible = denominator > 0
    recall = (selected & reference).sum(dim=-1) / denominator.clamp_min(1)
    report[f"{prefix}_page_recall"] += float((recall * eligible).sum())
    report[f"{prefix}_page_recall_count"] += int(eligible.sum())


def add_support_statistics(
    report: dict[str, float | int],
    ids: torch.Tensor,
    valid: torch.Tensor,
    historical_tokens: int,
) -> None:
    pages = selected_page_mask(ids, valid, historical_tokens, routed_only=False).sum(-1)
    tokens = valid.sum(-1)
    for name, values in (("physical_pages", pages), ("physical_tokens", tokens)):
        report[name] += float(values.sum())
        report[f"{name}_count"] += values.numel()
        report[f"{name}_min"] = min(report[f"{name}_min"], int(values.min()))
        report[f"{name}_max"] = max(report[f"{name}_max"], int(values.max()))


def derive_metrics(sums: dict[str, float | int]) -> dict[str, float | int]:
    count = int(sums["correlation_count"])
    numerator = count * sums["correlation_xy"] - sums["correlation_x"] * sums["correlation_y"]
    denominator_x = count * sums["correlation_x2"] - sums["correlation_x"] ** 2
    denominator_y = count * sums["correlation_y2"] - sums["correlation_y"] ** 2
    denominator = math.sqrt(max(0.0, denominator_x) * max(0.0, denominator_y))
    return {
        "page_logit_rel_mse": sums["page_squared_error"] / sums["page_energy"],
        "page_logit_pearson": numerator / denominator if denominator > 0 else 1.0,
        "page_recall_teacher": sums["teacher_page_recall"]
        / sums["teacher_page_recall_count"],
        "routed_page_recall_teacher": sums["teacher_routed_page_recall"]
        / sums["teacher_routed_page_recall_count"],
        "page_recall_exact_qk": sums["oracle_page_recall"]
        / sums["oracle_page_recall_count"],
        "routed_page_recall_exact_qk": sums["oracle_routed_page_recall"]
        / sums["oracle_routed_page_recall_count"],
        "attention_mass": sums["attention_mass"] / sums["attention_count"],
        "non_sink_attention_mass": sums["non_sink_attention_mass"]
        / sums["attention_count"],
        "output_rel_mse": sums["output_squared_error"] / sums["output_energy"],
        "wo_rel_mse": sums["wo_squared_error"] / sums["wo_energy"],
        "physical_pages_mean": sums["physical_pages"]
        / sums["physical_pages_count"],
        "physical_pages_min": sums["physical_pages_min"],
        "physical_pages_max": sums["physical_pages_max"],
        "physical_tokens_mean": sums["physical_tokens"]
        / sums["physical_tokens_count"],
        "physical_tokens_min": sums["physical_tokens_min"],
        "physical_tokens_max": sums["physical_tokens_max"],
    }


def merge_sums(reports: list[dict[str, float | int]]) -> dict[str, float | int]:
    merged = empty_sums()
    minima = ("physical_pages_min", "physical_tokens_min")
    maxima = ("physical_pages_max", "physical_tokens_max")
    for key in merged:
        if key in minima:
            merged[key] = min(report[key] for report in reports)
        elif key in maxima:
            merged[key] = max(report[key] for report in reports)
        else:
            merged[key] = sum(report[key] for report in reports)
    return merged


@torch.inference_mode()
def landmark_window_metrics(
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
    """Evaluate all teacher, landmark, and Exact-QK supports on one 64K window."""

    batch, query_heads, length, head_dim = map(int, query.shape)
    kv_heads = int(key.shape[1])
    query_groups = query_heads // kv_heads
    assert batch == 1 and length == SEQUENCE_LENGTH and 0 < steps <= ONLINE_STEPS
    assert key.shape == value.shape == (batch, kv_heads, length, head_dim)
    assert PREFIX_LENGTH + steps <= length and query_heads % kv_heads == 0
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
    landmarks = {
        name: build_subpage_landmarks(sidecar, chunk)
        for name, chunk in LANDMARK_CHUNKS.items()
    }
    exact_landmarks = build_subpage_landmarks(key, 8)
    scale = head_dim**-0.5
    values = value.float()
    output_weight = output_weight.float()
    sums = {name: empty_sums() for name in ARMS}
    audits = {
        "teacher_existing_selector": True,
        "gqa_existing_selector": True,
        "sink_recent_existing_selector": True,
        "exact_selected_k_dense_v": True,
        "no_future_token_state": True,
        "physical_page_count_equal": True,
    }

    for step, position in enumerate(range(PREFIX_LENGTH, PREFIX_LENGTH + steps)):
        total_tokens = position + 1
        historical_tokens = total_tokens - RECENT_TOKENS
        current_query = query[:, :, position].float()
        grouped_query = current_query.reshape(batch, kv_heads, query_groups, head_dim)
        exact_logits = (grouped_query @ key[:, :, :total_tokens].float().transpose(-1, -2)) * scale
        exact_probabilities = exact_logits.softmax(-1)
        non_sink_logits = exact_logits.clone()
        non_sink_logits[..., :PAGE_SIZE] = -torch.inf
        non_sink_probabilities = non_sink_logits.softmax(-1)
        dense_output = exact_probabilities @ values[:, :, :total_tokens]
        dense_wo = dense_output.reshape(batch, -1) @ output_weight.T

        residual_query = torch.einsum(
            "bhd,hdr->bhr",
            current_query,
            factors["residual_query_b16_r16"].float(),
        )
        routing_query = torch.cat((current_query, residual_query), -1).reshape(
            batch, kv_heads, query_groups, -1
        )
        teacher_scores = (
            routing_query @ sidecar[:, :, :total_tokens].float().transpose(-1, -2)
        ) * scale
        teacher_pages = token_page_scores(teacher_scores, historical_tokens)
        exact_pages = token_page_scores(exact_logits, historical_tokens)
        page_scores = {
            "TOKEN_TEACHER": teacher_pages,
            "EXACT_QK": exact_pages,
        }
        for name, chunk in LANDMARK_CHUNKS.items():
            full_landmarks, full_counts = landmarks[name]
            page_scores[name] = prefix_landmark_page_scores(
                routing_query,
                sidecar,
                full_landmarks,
                full_counts,
                historical_tokens,
                chunk,
                scale=scale,
            )
        page_scores["EXACT_K_LM8x4"] = prefix_landmark_page_scores(
            grouped_query,
            key,
            exact_landmarks[0],
            exact_landmarks[1],
            historical_tokens,
            8,
            scale=scale,
        )

        supports = {
            "TOKEN_TEACHER": page_support(teacher_scores, budget=budget),
            "EXACT_QK": page_support(exact_logits, budget=budget),
        }
        for name in (*LANDMARK_CHUNKS, "EXACT_K_LM8x4"):
            proxy = page_scores_to_token_proxy(
                page_scores[name],
                historical_tokens,
                total_tokens,
            )
            supports[name] = page_support(proxy, budget=budget)

        if step == 0:
            teacher_proxy = page_scores_to_token_proxy(
                teacher_pages,
                historical_tokens,
                total_tokens,
            )
            proxy_support = page_support(teacher_proxy, budget=budget)
            audits["teacher_existing_selector"] &= torch.equal(
                supports["TOKEN_TEACHER"][0], proxy_support[0]
            ) and torch.equal(supports["TOKEN_TEACHER"][1], proxy_support[1])

        teacher_mask = selected_page_mask(
            *supports["TOKEN_TEACHER"], historical_tokens, routed_only=False
        )
        teacher_routed = selected_page_mask(
            *supports["TOKEN_TEACHER"], historical_tokens, routed_only=True
        )
        oracle_mask = selected_page_mask(
            *supports["EXACT_QK"], historical_tokens, routed_only=False
        )
        oracle_routed = selected_page_mask(
            *supports["EXACT_QK"], historical_tokens, routed_only=True
        )
        page_counts = []
        for name in ARMS:
            report = sums[name]
            add_tensor_statistics(report, page_scores[name], teacher_pages)
            ids, valid = supports[name]
            assert (valid.sum(-1) <= budget).all()
            selected = selected_page_mask(
                ids, valid, historical_tokens, routed_only=False
            )
            selected_routed = selected_page_mask(
                ids, valid, historical_tokens, routed_only=True
            )
            add_recall(report, selected, teacher_mask, "teacher")
            add_recall(report, selected_routed, teacher_routed, "teacher_routed")
            add_recall(report, selected, oracle_mask, "oracle")
            add_recall(report, selected_routed, oracle_routed, "oracle_routed")
            add_support_statistics(report, ids, valid, historical_tokens)
            page_counts.append(selected.sum(-1))

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
        reference_count = page_counts[0]
        audits["physical_page_count_equal"] &= all(
            torch.equal(reference_count, count) for count in page_counts[1:]
        )
        if (step + 1) % 32 == 0:
            print(f"ONLINE_STEP {step + 1}", flush=True)

    assert all(audits.values()), audits
    return sums, audits


def protocol(root: Path, budget: int) -> dict:
    identity = read_json(root / "manifests/v128.json")
    window_manifest = read_json(root / "calibration/manifest.json")
    assert window_manifest["sha256"] == sha256(root / "calibration/windows.safetensors")
    assert window_manifest["validation_ids"] == list(WINDOWS)
    assert identity["layer_ranks"] == [128] * 32
    bank_hashes = {}
    for layer in range(32):
        path = root / "ours_b16r16" / f"layer_{layer:03d}.safetensors"
        meta = read_json(path.with_suffix(".json"))
        assert meta["status"] == "complete" and meta["sha256"] == sha256(path)
        assert meta["identity_sha256"] == sha256(root / "manifests/v128.json")
        assert meta["protocol"]["base_rank"] == 16
        assert meta["protocol"]["residual_rank"] == 16
        assert meta["protocol"]["windows_sha256"] == window_manifest["sha256"]
        assert not meta["protocol"]["smoke"]
        bank_hashes[str(layer)] = meta["sha256"]
    sources = [
        "evaluation/diagnose_page_landmarks.py",
        "evaluation/llama_sink_recent_routing.py",
        "basisserve/core/c1_conditional_page_attention.py",
        "basisserve/core/c1_v_conditional_k_router.py",
    ]
    return {
        "format": "basisserve.b16r16.parameter_free_page_landmarks.v1",
        "model": identity["model"],
        "identity_sha256": sha256(root / "manifests/v128.json"),
        "windows_sha256": window_manifest["sha256"],
        "windows": list(WINDOWS),
        "sequence_length": SEQUENCE_LENGTH,
        "prefill_length": PREFIX_LENGTH,
        "online_decode_tokens": ONLINE_STEPS,
        "base_rank": 16,
        "residual_rank": 16,
        "page_size": PAGE_SIZE,
        "budget_tokens": budget,
        "sink_tokens": PAGE_SIZE,
        "recent_tokens": RECENT_TOKENS,
        "pinned_prefix_pages": 1,
        "variants": {
            name: {
                "subpage_tokens": chunk,
                "landmarks_per_page": PAGE_SIZE // chunk,
            }
            for name, chunk in LANDMARK_CHUNKS.items()
        },
        "exact_k_lm8x4": True,
        "teacher": "Existing B16R16 token-side [predicted post-RoPE K128, residual R16] score and exact token Page-LSE",
        "landmark_construction": "Parameter-free contiguous means; FP32 accumulation, BF16 landmark storage; ragged historical boundary rebuilt only from currently historical cached tokens",
        "selection": "Existing page_support: per-query-head page softmax, max over four query heads in each physical GQA KV group, pinned Page0, exact recent64",
        "payload": "Exact selected post-RoPE K128 and original Dense V128; original Dense Wo",
        "arithmetic": "BF16 dense activations and routing sidecar; FP32 query projection, scores, exact QK softmax, sparse output, and Wo",
        "aggregation": "Pooled sufficient statistics over 256 continuous tail queries, 16 held-out 64K windows, and 32 layers; equal-layer means also reported",
        "bank_sha256": bank_hashes,
        "source_sha256": {name: sha256(Path(name)) for name in sources},
    }


def load_layer_factors(root: Path, layer: int, spec: dict) -> dict[str, torch.Tensor]:
    path = root / "ours_b16r16" / f"layer_{layer:03d}.safetensors"
    tensors, meta = verified(path)
    assert meta["sha256"] == spec["bank_sha256"][str(layer)]
    assert meta["sweeps"] == 40 and meta["pcg_iterations"] == 100
    return {name: value.cuda() for name, value in tensors.items()}


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert path.read_text() == value
        return
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(value)
    partial.replace(path)


def commands(root: Path, output: Path, budget: int) -> dict[str, str]:
    python = "/home/zhangal/.conda/envs/basis/bin/python"
    base = (
        f"{python} -u -m evaluation.diagnose_page_landmarks"
        f" --root {shlex.quote(str(root))} --output {shlex.quote(str(output))}"
        f" --budget {budget}"
    )
    return {
        "environment": "basis",
        "python": python,
        "smoke": f"{python} -u -m evaluation.diagnose_page_landmarks smoke --root {shlex.quote(str(root))} --output {shlex.quote(str(output))} --budget {budget}",
        "evaluate": f"{python} -u -m evaluation.diagnose_page_landmarks evaluate --root {shlex.quote(str(root))} --output {shlex.quote(str(output))} --budget {budget} --shard-index $SLURM_ARRAY_TASK_ID",
        "summarize": f"{python} -u -m evaluation.diagnose_page_landmarks summarize --root {shlex.quote(str(root))} --output {shlex.quote(str(output))} --budget {budget}",
    }


def cost_table() -> dict[str, dict[str, float | int | str | None]]:
    costs = {
        "TOKEN_TEACHER": {
            "routing_representations_per_page32": 32,
            "persistent_dimensions_per_token": 32,
            "landmark_dimensions_scanned_per_token": None,
            "note": "Base16+Residual16 persistent token state, plus transient Base reconstruction; this diagnostic materializes the 144D post-RoPE scoring state",
        },
        "EXACT_K_LM8x4": {
            "routing_representations_per_page32": 4,
            "persistent_dimensions_per_token": None,
            "landmark_dimensions_scanned_per_token": 16.0,
            "note": "Diagnostic exact-K pooling oracle only",
        },
        "EXACT_QK": {
            "routing_representations_per_page32": 32,
            "persistent_dimensions_per_token": 128,
            "landmark_dimensions_scanned_per_token": None,
            "note": "Exact-QK page-selection oracle",
        },
    }
    for name, chunk in LANDMARK_CHUNKS.items():
        count = PAGE_SIZE // chunk
        costs[name] = {
            "routing_representations_per_page32": count,
            "persistent_dimensions_per_token": 144 * count / PAGE_SIZE,
            "landmark_dimensions_scanned_per_token": 144 * count / PAGE_SIZE,
            "note": "BF16 post-RoPE Base128+Residual16 landmark coordinates",
        }
    return {name: costs[name] for name in ARMS}


def markdown_summary(result: dict) -> str:
    metrics = result["pooled"]
    costs = result["cost"]
    teacher = metrics["TOKEN_TEACHER"]
    lm8 = metrics["LM8x4"]
    close = result["decision"]["lm8x4_close_to_teacher"]
    lines = [
        "# Llama-3.1-8B-Instruct B16R16 Parameter-Free Page Landmarks",
        "",
        "## Conclusion",
        "",
        (
            "Yes under the declared diagnostic rule: LM8x4 preserves the fitted token router closely enough to justify the fixed 330-sample RULER follow-up."
            if close
            else "Not yet under the declared diagnostic rule: LM8x4 loses too much selection/output fidelity, so no RULER follow-up should be launched from this result."
        ),
        "",
        "This is a parameter-free diagnostic of the existing B16R16 fit. No Base/Residual refit, ALS, SVD/RRR landmark fit, or learned pooling was used.",
        "",
        "## Pooled diagnostic",
        "",
        "| Variant | Reps/Page32 | Scan dims/token | Page rel-MSE | Pearson | Recall vs teacher | Recall vs Exact-QK | Attention mass | Non-sink mass | Output rel-MSE | Wo rel-MSE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ARMS:
        metric = metrics[name]
        cost = costs[name]
        scan = cost["landmark_dimensions_scanned_per_token"]
        scan_text = "n/a" if scan is None else f"{scan:g}"
        lines.append(
            f"| {name} | {cost['routing_representations_per_page32']} | {scan_text} | "
            f"{metric['page_logit_rel_mse']:.6g} | {metric['page_logit_pearson']:.6f} | "
            f"{metric['routed_page_recall_teacher']:.4f} | {metric['routed_page_recall_exact_qk']:.4f} | "
            f"{metric['attention_mass']:.4f} | {metric['non_sink_attention_mass']:.4f} | "
            f"{metric['output_rel_mse']:.6g} | {metric['wo_rel_mse']:.6g} |"
        )
    lines.extend(
        [
            "",
            "The teacher's deployable persistent state is Base16+Residual16 (32 dimensions/token) plus transient Base reconstruction. The 144D materialized teacher state used by this diagnostic is not claimed as equivalent storage. Landmark scan costs are BF16 coordinates scanned per original token.",
            "",
            "## LM8x4 decision audit",
            "",
            "The diagnostic rule was declared in the result artifact: routed page recall against TOKEN_TEACHER >= 0.80, attention-mass loss <= 0.01, and Wo rel-MSE <= 1.25x TOKEN_TEACHER. It is only a gate for the specified RULER follow-up, not a general quality claim.",
            "",
            f"- Routed page recall vs teacher: {lm8['routed_page_recall_teacher']:.6f}",
            f"- Attention-mass change vs teacher: {lm8['attention_mass'] - teacher['attention_mass']:+.6f}",
            f"- Wo rel-MSE ratio vs teacher: {lm8['wo_rel_mse'] / teacher['wo_rel_mse']:.6f}x",
            f"- Follow-up gate: {'PASS' if close else 'FAIL'}",
            "",
            "## Protocol and audits",
            "",
            "- 32 layers, 16 held-out C4 64K windows, 256 continuous online tail queries per window.",
            "- Hard B=2048 support, Page32, pinned Page0/sink32, exact recent64, unchanged physical GQA aggregation.",
            "- Exact selected post-RoPE K and Dense V are used after routing; Wo is the original dense projection.",
            "- Full pages are pooled once; the moving ragged historical boundary is rebuilt using only currently historical cached tokens.",
            "- TOKEN_TEACHER uses the existing selector directly; landmark scores re-enter that selector through an exact Page-LSE-preserving proxy.",
            "- Synthetic identical-token, weighted ragged-LSE, and proxy round-trip audits passed.",
            "",
        ]
    )
    return "\n".join(lines)


@torch.inference_mode()
def run_model_stage(args: argparse.Namespace, spec: dict) -> None:
    root, output = args.root, args.output
    if args.stage == "evaluate":
        smoke = read_json(output / "smoke.json")
        assert smoke["status"] == "complete" and smoke["protocol"] == spec
        assert all(smoke["synthetic_audits"].values())
    layers = [0] if args.stage == "smoke" else list(
        range(args.shard_index * 8, (args.shard_index + 1) * 8)
    )
    selected_windows = [WINDOWS[0]] if args.stage == "smoke" else list(WINDOWS)
    factors = {layer: load_layer_factors(root, layer, spec) for layer in layers}
    identity = read_json(root / "manifests/v128.json")
    model = AutoModelForCausalLM.from_pretrained(
        identity["model"],
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=True,
    ).eval().cuda()
    assert model.config.model_type == "llama"
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

    windows = load_file(str(root / "calibration/windows.safetensors"))["input_ids"]
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
            sums, audits = landmark_window_metrics(
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
            record = {
                "window": active["window"],
                "steps": ONLINE_STEPS,
                "metrics": metrics,
                "sums": sums,
                "audits": audits,
            }
            records[layer].append(record)
            print(
                {
                    "layer": layer,
                    "window": active["window"],
                    "teacher_wo_rel_mse": metrics["TOKEN_TEACHER"]["wo_rel_mse"],
                    "lm8x4_wo_rel_mse": metrics["LM8x4"]["wo_rel_mse"],
                    "lm8x4_teacher_recall": metrics["LM8x4"]["routed_page_recall_teacher"],
                },
                flush=True,
            )

        handle = model.model.layers[layer].self_attn.register_forward_pre_hook(
            capture,
            with_kwargs=True,
        )
        handles.append(handle)
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
        metrics = {name: derive_metrics(combined[name]) for name in ARMS}
        artifact = {
            "status": "complete",
            "layer": layer,
            "protocol": spec,
            "synthetic_audits": synthetic_audits(),
            "metrics": metrics,
            "sums": combined,
            "windows": records[layer],
            "command": shlex.join(sys.argv),
            "python": sys.executable,
        }
        path = output / "smoke.json" if args.stage == "smoke" else output / f"layer_{layer:03d}.json"
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
            if not metric.endswith("_min") and not metric.endswith("_max")
        }
        for name in ARMS
    }
    teacher = pooled["TOKEN_TEACHER"]
    lm8 = pooled["LM8x4"]
    decision_rule = {
        "routed_page_recall_teacher_min": 0.80,
        "attention_mass_loss_max": 0.01,
        "wo_rel_mse_ratio_max": 1.25,
    }
    close = (
        lm8["routed_page_recall_teacher"] >= decision_rule["routed_page_recall_teacher_min"]
        and teacher["attention_mass"] - lm8["attention_mass"]
        <= decision_rule["attention_mass_loss_max"]
        and lm8["wo_rel_mse"]
        <= decision_rule["wo_rel_mse_ratio_max"] * teacher["wo_rel_mse"]
    )
    result = {
        "status": "complete",
        "protocol": spec,
        "pooled": pooled,
        "equal_layer_mean": equal_layer_mean,
        "cost": cost_table(),
        "decision": {
            "rule": decision_rule,
            "lm8x4_close_to_teacher": close,
            "launch_ruler_followup": close,
        },
        "audits": {
            "synthetic": synthetic_audits(),
            "all_runtime_audits_passed": True,
        },
    }
    per_layer = {
        "status": "complete",
        "protocol": spec,
        "layers": [
            {"layer": layer["layer"], "metrics": layer["metrics"]}
            for layer in layers
        ],
    }
    write_json(args.output / "result.json", result)
    write_json(args.output / "per_layer.json", per_layer)
    write_text(args.output / "summary.md", markdown_summary(result))
    print(json.dumps(result["pooled"], indent=2), flush=True)


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
    write_json(args.output / "commands.json", commands(args.root, args.output, args.budget))
    if args.stage == "summarize":
        summarize(args, spec)
    else:
        run_model_stage(args, spec)


if __name__ == "__main__":
    main()
