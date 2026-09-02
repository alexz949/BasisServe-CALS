#!/usr/bin/env python3
"""Fit and evaluate V-conditioned residual-Key routers on Qwen3-8B."""

from __future__ import annotations

import argparse
from dataclasses import asdict
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
from transformers import AutoConfig
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basisserve.core.c1_v_conditional_k_router import (  # noqa: E402
    AffineReducedRankMap,
    fit_affine_reduced_rank_map,
    residual_page_fisher_gram,
)
from basisserve.core.c1_v_k_index import apply_rotary, invert_rotary  # noqa: E402
from basisserve.core.exact_qk_v_offload import (  # noqa: E402
    gqa_group_max_page_mass_mask,
)
from basisserve.core.gqa_joint_routing_payload_s80_ablation import (  # noqa: E402
    fit_page_fisher_router,
)
from basisserve.core.gqa_joint_routing_payload_s80_fisher import (  # noqa: E402
    S80CompactSoftmaxFisherRouting,
    compact_softmax_fisher_loss,
    softmax_fisher_transform,
)


FORMAT = "basisserve.qwen3_8b.v80_conditional_residual_router.v2"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--calibration-root", required=True)
    parser.add_argument("--fresh-direct-dir", required=True)
    parser.add_argument("--c1-v80", required=True)
    parser.add_argument("--reference-json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--layers", default="0,17,35")
    parser.add_argument("--base-ranks", default="16,32,48,64")
    parser.add_argument("--residual-ranks", default="0,8,16,24,32")
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument("--exact-token-budget", type=int, default=2048)
    parser.add_argument("--router-sweeps", type=int, default=10)
    parser.add_argument("--relative-damping", type=float, default=1e-5)
    parser.add_argument("--iterative-tolerance", type=float, default=1e-5)
    parser.add_argument("--iterative-max-iterations", type=int, default=100)
    parser.add_argument("--work-device", default="cuda")
    parser.add_argument("--torch-num-threads", type=int, default=4)
    return parser.parse_args()


def _parse_ints(specification: str) -> tuple[int, ...]:
    return tuple(sorted({int(item) for item in specification.split(",") if item}))


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_safetensors(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    save_file(tensors, str(temporary), metadata={"format": FORMAT})
    os.replace(temporary, path)


def _mmap_tensor(root: Path, record: dict[str, Any]) -> torch.Tensor:
    shape = tuple(int(size) for size in record["shape"])
    values = math.prod(shape)
    return torch.from_file(
        str(root / record["file"]),
        shared=False,
        size=values,
        dtype=torch.bfloat16,
    ).reshape(shape)


def _discover_capture(
    calibration_root: Path,
    *,
    split: str,
    layer: int,
) -> tuple[Path, dict[str, Any]]:
    pattern = f"qwen3_8b_s80_c4_64f16h_s32768_chunk_*/direct/{split}/manifest.json"
    matches = []
    for path in sorted(calibration_root.glob(pattern)):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if str(layer) in manifest["artifacts"]:
            matches.append((path.parent, manifest))
    assert len(matches) == 1
    return matches[0]


def _load_direct(
    root: Path,
    manifest: dict[str, Any],
    layer: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    records = manifest["artifacts"][str(layer)]
    return (
        _mmap_tensor(root, records["routing_queries"]),
        _mmap_tensor(root, records["routing_joint_rows"]),
    )


def _rotary_embeddings(
    model_root: Path,
    *,
    sequence: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    config = AutoConfig.from_pretrained(str(model_root), local_files_only=True)
    rotary = Qwen3RotaryEmbedding(config, device=device)
    positions = torch.arange(sequence, device=device).unsqueeze(0)
    example = torch.empty(1, device=device, dtype=torch.float32)
    return rotary(example, positions)


def _pre_rope_rows(
    post_rope: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    head_major = post_rope.permute(1, 0, 2).unsqueeze(0)
    return invert_rotary(head_major, cos, sin)[0].permute(1, 0, 2)


def _post_rope_rows(
    pre_rope: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    head_major = pre_rope.permute(1, 0, 2).unsqueeze(0)
    return apply_rotary(head_major, cos, sin)[0].permute(1, 0, 2)


def _value_codes(
    dense_value: torch.Tensor,
    value_encoder: torch.Tensor,
) -> torch.Tensor:
    return torch.einsum("tgd,gdr->tgr", dense_value, value_encoder)


def _fit_base_maps(
    rows: torch.Tensor,
    *,
    value_encoder: torch.Tensor,
    base_ranks: tuple[int, ...],
    cos: torch.Tensor,
    sin: torch.Tensor,
    device: torch.device,
) -> dict[int, tuple[AffineReducedRankMap, ...]]:
    documents, tokens, groups, joint_dim = map(int, rows.shape)
    head_dim = joint_dim // 2
    value_rank = int(value_encoder.shape[-1])
    input_sum = torch.zeros(groups, value_rank, dtype=torch.float64)
    target_sum = torch.zeros(groups, head_dim, dtype=torch.float64)
    input_gram = torch.zeros(groups, value_rank, value_rank, dtype=torch.float64)
    cross_gram = torch.zeros(groups, value_rank, head_dim, dtype=torch.float64)
    encoder = value_encoder.to(device=device, dtype=torch.float32)
    for document in range(documents):
        print(
            f"    base moments document={document + 1}/{documents}",
            flush=True,
        )
        current = rows[document].to(device=device, dtype=torch.float32)
        dense_value = current[..., :head_dim]
        exact_pre_key = _pre_rope_rows(
            current[..., head_dim:],
            cos,
            sin,
        )
        codes = _value_codes(dense_value, encoder)
        group_codes = codes.permute(1, 0, 2)
        group_keys = exact_pre_key.permute(1, 0, 2)
        input_sum += group_codes.sum(dim=1).double().cpu()
        target_sum += group_keys.sum(dim=1).double().cpu()
        input_gram += torch.bmm(group_codes.mT, group_codes).double().cpu()
        cross_gram += torch.bmm(group_codes.mT, group_keys).double().cpu()
        del current, dense_value, exact_pre_key, codes, group_codes, group_keys
    row_count = documents * tokens
    fitted: dict[int, tuple[AffineReducedRankMap, ...]] = {}
    for rank in base_ranks:
        fitted[rank] = tuple(
            fit_affine_reduced_rank_map(
                row_count=row_count,
                input_sum=input_sum[group],
                target_sum=target_sum[group],
                input_gram=input_gram[group],
                input_target_gram=cross_gram[group],
                rank=rank,
            )
            for group in range(groups)
        )
    return fitted


def _stack_base_map(
    maps: tuple[AffineReducedRankMap, ...],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    left = torch.stack([item.left for item in maps]).to(
        device=device,
        dtype=torch.float32,
    )
    right = torch.stack([item.right for item in maps]).to(
        device=device,
        dtype=torch.float32,
    )
    bias = torch.stack([item.bias for item in maps]).to(
        device=device,
        dtype=torch.float32,
    )
    return left, right, bias


def _apply_base(
    codes: torch.Tensor,
    factors: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    left, right, bias = factors
    return torch.einsum("tgi,gir,grd->tgd", codes, left, right) + bias


def _build_residual_statistics(
    queries_raw: torch.Tensor,
    rows_raw: torch.Tensor,
    *,
    value_encoder: torch.Tensor,
    base_maps: dict[int, tuple[AffineReducedRankMap, ...]],
    cos: torch.Tensor,
    sin: torch.Tensor,
    page_size: int,
    device: torch.device,
) -> tuple[dict[int, S80CompactSoftmaxFisherRouting], dict[int, dict[str, float]]]:
    documents, query_heads, head_dim = map(int, queries_raw.shape)
    _, _, groups, _ = map(int, rows_raw.shape)
    heads_per_group = query_heads // groups
    mapping = torch.arange(query_heads, dtype=torch.long) // heads_per_group
    queries_cpu = queries_raw.permute(1, 0, 2).contiguous().float()
    grams = {
        rank: torch.empty(
            query_heads,
            documents,
            head_dim,
            head_dim,
            dtype=torch.float32,
        )
        for rank in base_maps
    }
    energy = {rank: 0.0 for rank in base_maps}
    residual_error = {rank: 0.0 for rank in base_maps}
    key_energy = 0.0
    encoder = value_encoder.to(device=device, dtype=torch.float32)
    factors = {
        rank: _stack_base_map(maps, device=device)
        for rank, maps in base_maps.items()
    }
    scaling = head_dim**-0.5
    for document in range(documents):
        print(
            f"    residual Page-Fisher document={document + 1}/{documents}",
            flush=True,
        )
        current = rows_raw[document].to(device=device, dtype=torch.float32)
        queries = queries_raw[document].to(device=device, dtype=torch.float32)
        dense_value = current[..., :head_dim]
        exact_key = current[..., head_dim:]
        codes = _value_codes(dense_value, encoder)
        key_energy += float(exact_key.square().sum())
        for rank, base_factor in factors.items():
            base_pre = _apply_base(codes, base_factor)
            base_post = _post_rope_rows(base_pre, cos, sin)
            residual = exact_key - base_post
            residual_error[rank] += float(residual.square().sum())
            for group in range(groups):
                first = group * heads_per_group
                stop = first + heads_per_group
                group_grams, group_energy = residual_page_fisher_gram(
                    queries[first:stop],
                    exact_key[:, group],
                    residual[:, group],
                    scaling=scaling,
                    page_size=page_size,
                )
                grams[rank][first:stop, document].copy_(
                    group_grams.float().cpu()
                )
                energy[rank] += group_energy
        del current, queries, dense_value, exact_key, codes
    statistics = {
        rank: S80CompactSoftmaxFisherRouting(
            queries_by_head=queries_cpu,
            fisher_grams_by_head=grams[rank],
            head_to_kv_group=mapping,
            value_dim=0,
            key_dim=head_dim,
            scaling=scaling,
            teacher_fisher_energy=energy[rank],
        )
        for rank in base_maps
    }
    reconstruction = {
        rank: {
            "post_rope_relative_mse": residual_error[rank] / key_energy,
            "residual_squared_error": residual_error[rank],
            "exact_key_squared_energy": key_energy,
        }
        for rank in base_maps
    }
    return statistics, reconstruction


def _initial_residual_factors(
    statistics: S80CompactSoftmaxFisherRouting,
    *,
    rank: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    mapping = statistics.head_to_kv_group
    groups = int(mapping.max()) + 1
    encoders = []
    for group in range(groups):
        heads = torch.nonzero(mapping == group, as_tuple=False).flatten()
        gram = statistics.fisher_grams_by_head.index_select(0, heads).sum(
            dim=(0, 1)
        )
        _, eigenvectors = torch.linalg.eigh(0.5 * (gram + gram.mT))
        encoders.append(eigenvectors[:, -rank:])
    encoder = torch.stack(encoders).to(device=device, dtype=torch.float32)
    query = encoder.index_select(0, mapping.to(device=device))
    return encoder, query


def _fit_residual_grid(
    fit_statistics: dict[int, S80CompactSoftmaxFisherRouting],
    validation_statistics: dict[int, S80CompactSoftmaxFisherRouting],
    *,
    residual_ranks: tuple[int, ...],
    sweeps: int,
    relative_damping: float,
    iterative_tolerance: float,
    iterative_max_iterations: int,
    device: torch.device,
) -> tuple[
    dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]],
    dict[str, dict[str, Any]],
]:
    factor_bank: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}
    diagnostics: dict[str, dict[str, Any]] = {}
    for base_rank, fit_cpu in fit_statistics.items():
        fit = S80CompactSoftmaxFisherRouting(
            queries_by_head=fit_cpu.queries_by_head.to(device),
            fisher_grams_by_head=fit_cpu.fisher_grams_by_head.to(device),
            head_to_kv_group=fit_cpu.head_to_kv_group.to(device),
            value_dim=0,
            key_dim=fit_cpu.key_dim,
            scaling=fit_cpu.scaling,
            teacher_fisher_energy=fit_cpu.teacher_fisher_energy,
        )
        validation_cpu = validation_statistics[base_rank]
        validation = S80CompactSoftmaxFisherRouting(
            queries_by_head=validation_cpu.queries_by_head.to(device),
            fisher_grams_by_head=validation_cpu.fisher_grams_by_head.to(device),
            head_to_kv_group=validation_cpu.head_to_kv_group.to(device),
            value_dim=0,
            key_dim=validation_cpu.key_dim,
            scaling=validation_cpu.scaling,
            teacher_fisher_energy=validation_cpu.teacher_fisher_energy,
        )
        diagnostics[f"b{base_rank}_r0"] = {
            "base_rank": base_rank,
            "residual_rank": 0,
            "fit_page_fisher_nmse": 1.0,
            "validation_page_fisher_nmse": 1.0,
        }
        for residual_rank in residual_ranks:
            if residual_rank == 0:
                continue
            print(
                f"    fit residual router base={base_rank} rank={residual_rank}",
                flush=True,
            )
            initial_encoder, initial_query = _initial_residual_factors(
                fit,
                rank=residual_rank,
                device=device,
            )
            fitted = fit_page_fisher_router(
                fit,
                initial_routing_encoders=initial_encoder,
                initial_query_factors=initial_query,
                active_joint_rows=torch.arange(fit.key_dim, device=device),
                sweeps=sweeps,
                relative_damping=relative_damping,
                relative_tolerance=iterative_tolerance,
                max_iterations=iterative_max_iterations,
            )
            fit_loss = compact_softmax_fisher_loss(
                fit,
                routing_payload_encoders=fitted.routing_encoders,
                routing_query_factors=fitted.routing_query_factors,
            )
            validation_loss = compact_softmax_fisher_loss(
                validation,
                routing_payload_encoders=fitted.routing_encoders,
                routing_query_factors=fitted.routing_query_factors,
            )
            factor_bank[(base_rank, residual_rank)] = (
                fitted.routing_encoders.float().cpu(),
                fitted.routing_query_factors.float().cpu(),
            )
            diagnostics[f"b{base_rank}_r{residual_rank}"] = {
                "base_rank": base_rank,
                "residual_rank": residual_rank,
                "fit_page_fisher_nmse": fit_loss / fit.teacher_fisher_energy,
                "validation_page_fisher_nmse": (
                    validation_loss / validation.teacher_fisher_energy
                ),
                "sweeps": [asdict(item) for item in fitted.sweeps],
                "final_query_maximum_iterations": max(
                    item.iterations for item in fitted.final_query_diagnostics
                ),
                "final_query_maximum_relative_residual": max(
                    item.relative_residual for item in fitted.final_query_diagnostics
                ),
            }
        del fit, validation
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return factor_bank, diagnostics


def _new_metric_accumulator() -> dict[str, Any]:
    return {
        "raw_error": 0.0,
        "raw_energy": 0.0,
        "fisher_error": 0.0,
        "teacher_fisher_energy": 0.0,
        "output_error": 0.0,
        "output_energy": 0.0,
        "page_recalls": [],
        "mass_recalls": [],
        "selected_fractions": [],
    }


def _update_selection_metrics(
    accumulator: dict[str, Any],
    *,
    proxy_scores: torch.Tensor,
    exact_scores: torch.Tensor,
    probabilities: torch.Tensor,
    value_codes: torch.Tensor,
    output_decoder: torch.Tensor,
    head_to_group: torch.Tensor,
    page_size: int,
    exact_token_budget: int,
) -> None:
    query_heads, tokens = map(int, exact_scores.shape)
    groups = int(value_codes.shape[1])
    heads_per_group = query_heads // groups
    pages_per_group = math.ceil(exact_token_budget / page_size)
    selected_tokens, selected_pages = gqa_group_max_page_mass_mask(
        proxy_scores,
        num_kv_heads=groups,
        page_size=page_size,
        pages_per_kv_head=pages_per_group,
    )
    _, teacher_pages = gqa_group_max_page_mass_mask(
        exact_scores,
        num_kv_heads=groups,
        page_size=page_size,
        pages_per_kv_head=pages_per_group,
    )
    page_recall = (
        (selected_pages & teacher_pages).sum(dim=-1)
        / teacher_pages.sum(dim=-1).clamp_min(1)
    )
    query_mask = selected_tokens.repeat_interleave(heads_per_group, dim=0)
    mass_recall = (probabilities * query_mask).sum(dim=-1)
    accumulator["page_recalls"].extend(page_recall.tolist())
    accumulator["mass_recalls"].extend(mass_recall.tolist())
    accumulator["selected_fractions"].extend(
        selected_tokens.float().mean(dim=-1).tolist()
    )
    delta = proxy_scores - exact_scores
    accumulator["raw_error"] += float(delta.square().sum())
    accumulator["raw_energy"] += float(exact_scores.square().sum())
    accumulator["fisher_error"] += float(
        softmax_fisher_transform(delta, probabilities).square().sum()
    )
    accumulator["teacher_fisher_energy"] += float(
        softmax_fisher_transform(exact_scores, probabilities).square().sum()
    )

    head_codes = value_codes.permute(1, 0, 2).index_select(0, head_to_group)
    dense_latent = torch.einsum("ht,htr->hr", probabilities, head_codes)
    sparse_probabilities = torch.softmax(
        exact_scores.masked_fill(~query_mask, -torch.inf),
        dim=-1,
    )
    sparse_latent = torch.einsum(
        "ht,htr->hr",
        sparse_probabilities,
        head_codes,
    )
    dense_output = torch.einsum("hr,hro->o", dense_latent, output_decoder)
    sparse_output = torch.einsum("hr,hro->o", sparse_latent, output_decoder)
    accumulator["output_error"] += float(
        (sparse_output - dense_output).square().sum()
    )
    accumulator["output_energy"] += float(dense_output.square().sum())


def _finish_metrics(accumulator: dict[str, Any], *, tokens: int) -> dict[str, float]:
    selected_fraction = sum(accumulator["selected_fractions"]) / len(
        accumulator["selected_fractions"]
    )
    return {
        "raw_score_nmse": accumulator["raw_error"] / accumulator["raw_energy"],
        "softmax_fisher_nmse": (
            accumulator["fisher_error"] / accumulator["teacher_fisher_energy"]
        ),
        "physical_page_recall_mean": sum(accumulator["page_recalls"])
        / len(accumulator["page_recalls"]),
        "physical_page_recall_minimum": min(accumulator["page_recalls"]),
        "attention_mass_recall_mean": sum(accumulator["mass_recalls"])
        / len(accumulator["mass_recalls"]),
        "attention_mass_recall_minimum": min(accumulator["mass_recalls"]),
        "selected_token_fraction_mean": selected_fraction,
        "physical_token_budget_mean": selected_fraction * tokens,
        "exact_refined_output_relative_mse": (
            accumulator["output_error"] / accumulator["output_energy"]
        ),
    }


def _evaluate_fresh(
    queries_raw: torch.Tensor,
    rows_raw: torch.Tensor,
    *,
    value_encoder: torch.Tensor,
    output_decoder: torch.Tensor,
    base_maps: dict[int, tuple[AffineReducedRankMap, ...]],
    factor_bank: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]],
    residual_ranks: tuple[int, ...],
    cos: torch.Tensor,
    sin: torch.Tensor,
    page_size: int,
    exact_token_budget: int,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    documents, query_heads, head_dim = map(int, queries_raw.shape)
    _, tokens, groups, _ = map(int, rows_raw.shape)
    heads_per_group = query_heads // groups
    head_to_group = torch.arange(query_heads, device=device) // heads_per_group
    scaling = head_dim**-0.5
    encoder = value_encoder.to(device=device, dtype=torch.float32)
    decoder = output_decoder.to(device=device, dtype=torch.float32)
    base_factors = {
        rank: _stack_base_map(maps, device=device)
        for rank, maps in base_maps.items()
    }
    route_factors = {
        key: (
            values[0].to(device=device, dtype=torch.float32),
            values[1].to(device=device, dtype=torch.float32),
        )
        for key, values in factor_bank.items()
    }
    accumulators = {
        f"b{base_rank}_r{residual_rank}": _new_metric_accumulator()
        for base_rank in base_maps
        for residual_rank in residual_ranks
    }
    for document in range(documents):
        print(f"    fresh evaluation document={document + 1}/{documents}", flush=True)
        current = rows_raw[document].to(device=device, dtype=torch.float32)
        queries = queries_raw[document].to(device=device, dtype=torch.float32)
        dense_value = current[..., :head_dim]
        exact_key = current[..., head_dim:]
        codes = _value_codes(dense_value, encoder)
        exact_scores = torch.empty(
            query_heads,
            tokens,
            device=device,
            dtype=torch.float32,
        )
        for group in range(groups):
            first = group * heads_per_group
            stop = first + heads_per_group
            exact_scores[first:stop] = scaling * (
                queries[first:stop] @ exact_key[:, group].mT
            )
        probabilities = torch.softmax(exact_scores, dim=-1)
        for base_rank, base_factor in base_factors.items():
            base_pre = _apply_base(codes, base_factor)
            base_post = _post_rope_rows(base_pre, cos, sin)
            residual = exact_key - base_post
            base_scores = torch.empty_like(exact_scores)
            for group in range(groups):
                first = group * heads_per_group
                stop = first + heads_per_group
                base_scores[first:stop] = scaling * (
                    queries[first:stop] @ base_post[:, group].mT
                )
            for residual_rank in residual_ranks:
                proxy_scores = base_scores
                if residual_rank:
                    residual_encoder, query_factor = route_factors[
                        (base_rank, residual_rank)
                    ]
                    proxy_scores = torch.empty_like(exact_scores)
                    for group in range(groups):
                        first = group * heads_per_group
                        stop = first + heads_per_group
                        query_code = torch.einsum(
                            "hd,hdr->hr",
                            queries[first:stop],
                            query_factor[first:stop],
                        )
                        token_code = residual[:, group] @ residual_encoder[group]
                        proxy_scores[first:stop] = base_scores[first:stop] + scaling * (
                            query_code @ token_code.mT
                        )
                _update_selection_metrics(
                    accumulators[f"b{base_rank}_r{residual_rank}"],
                    proxy_scores=proxy_scores,
                    exact_scores=exact_scores,
                    probabilities=probabilities,
                    value_codes=codes,
                    output_decoder=decoder,
                    head_to_group=head_to_group,
                    page_size=page_size,
                    exact_token_budget=exact_token_budget,
                )
        del current, queries, dense_value, exact_key, codes, exact_scores
    return {
        name: _finish_metrics(accumulator, tokens=tokens)
        for name, accumulator in accumulators.items()
    }


def _reference_by_layer(path: Path | None) -> dict[int, dict[str, Any]]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        int(record["layer"]): record["arms"]["independent_v80_k32"]["selection"]
        for record in payload["layers"]
    }


def _aggregate(layer_records: list[dict[str, Any]]) -> dict[str, Any]:
    arms = sorted(layer_records[0]["fresh"])
    result: dict[str, Any] = {}
    for arm in arms:
        fresh_rows = [record["fresh"][arm] for record in layer_records]
        fit_rows = [record["router_fit"][arm] for record in layer_records]
        result[arm] = {
            "base_rank": fit_rows[0]["base_rank"],
            "residual_rank": fit_rows[0]["residual_rank"],
            "extra_k_storage_rank": fit_rows[0]["residual_rank"],
            "logical_base_compute_width": fit_rows[0]["base_rank"] * 128,
            "fit_page_fisher_nmse": sum(
                float(row["fit_page_fisher_nmse"]) for row in fit_rows
            )
            / len(fit_rows),
            "validation_page_fisher_nmse": sum(
                float(row["validation_page_fisher_nmse"]) for row in fit_rows
            )
            / len(fit_rows),
        }
        for key in fresh_rows[0]:
            values = [float(row[key]) for row in fresh_rows]
            result[arm][key] = min(values) if key.endswith("minimum") else sum(values) / len(values)
    references = [record.get("reference_k_only_pagef32") for record in layer_records]
    references = [record for record in references if record is not None]
    if references:
        result["reference_k_only_pagef32"] = {
            key: min(float(row[key]) for row in references)
            if key.endswith("minimum")
            else sum(float(row[key]) for row in references) / len(references)
            for key in references[0]
            if isinstance(references[0][key], (int, float))
        }
    return result


@torch.inference_mode()
def main() -> None:
    args = _parse_args()
    torch.set_num_threads(args.torch_num_threads)
    torch.backends.cuda.matmul.allow_tf32 = True
    started = time.monotonic()
    model_root = Path(args.model).expanduser().resolve()
    calibration_root = Path(args.calibration_root).expanduser().resolve()
    fresh_root = Path(args.fresh_direct_dir).expanduser().resolve()
    c1_root = Path(args.c1_v80).expanduser().resolve()
    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.work_device)
    layers = _parse_ints(args.layers)
    base_ranks = _parse_ints(args.base_ranks)
    residual_ranks = _parse_ints(args.residual_ranks)
    config = json.loads((model_root / "config.json").read_text(encoding="utf-8"))
    sequence = 32768
    cos, sin = _rotary_embeddings(
        model_root,
        sequence=sequence,
        device=device,
    )
    c1_manifest = json.loads((c1_root / "results.json").read_text(encoding="utf-8"))
    fresh_manifest = json.loads((fresh_root / "manifest.json").read_text(encoding="utf-8"))
    reference = _reference_by_layer(
        None if args.reference_json is None else Path(args.reference_json).expanduser().resolve()
    )
    layer_records: list[dict[str, Any]] = []
    progress_path = output_root / "result.json"

    for ordinal, layer in enumerate(layers, start=1):
        layer_started = time.monotonic()
        print(
            f"[conditional residual] layer={layer} ({ordinal}/{len(layers)})",
            flush=True,
        )
        fit_root, fit_manifest = _discover_capture(
            calibration_root,
            split="fit",
            layer=layer,
        )
        validation_root, validation_manifest = _discover_capture(
            calibration_root,
            split="validation",
            layer=layer,
        )
        fit_queries, fit_rows = _load_direct(fit_root, fit_manifest, layer)
        validation_queries, validation_rows = _load_direct(
            validation_root,
            validation_manifest,
            layer,
        )
        fresh_queries, fresh_rows = _load_direct(fresh_root, fresh_manifest, layer)
        c1_artifact = c1_manifest["artifacts"][str(layer)]
        c1_tensors = load_file(str(c1_root / c1_artifact["file"]), device="cpu")
        value_encoder = c1_tensors["value_coordinate_encoders"]
        output_decoder = c1_tensors["head_output_decoders"]

        base_maps = _fit_base_maps(
            fit_rows,
            value_encoder=value_encoder,
            base_ranks=base_ranks,
            cos=cos,
            sin=sin,
            device=device,
        )
        fit_statistics, fit_reconstruction = _build_residual_statistics(
            fit_queries,
            fit_rows,
            value_encoder=value_encoder,
            base_maps=base_maps,
            cos=cos,
            sin=sin,
            page_size=args.page_size,
            device=device,
        )
        validation_statistics, validation_reconstruction = _build_residual_statistics(
            validation_queries,
            validation_rows,
            value_encoder=value_encoder,
            base_maps=base_maps,
            cos=cos,
            sin=sin,
            page_size=args.page_size,
            device=device,
        )
        factor_bank, router_diagnostics = _fit_residual_grid(
            fit_statistics,
            validation_statistics,
            residual_ranks=residual_ranks,
            sweeps=args.router_sweeps,
            relative_damping=args.relative_damping,
            iterative_tolerance=args.iterative_tolerance,
            iterative_max_iterations=args.iterative_max_iterations,
            device=device,
        )
        fresh = _evaluate_fresh(
            fresh_queries,
            fresh_rows,
            value_encoder=value_encoder,
            output_decoder=output_decoder,
            base_maps=base_maps,
            factor_bank=factor_bank,
            residual_ranks=residual_ranks,
            cos=cos,
            sin=sin,
            page_size=args.page_size,
            exact_token_budget=args.exact_token_budget,
            device=device,
        )
        factor_tensors: dict[str, torch.Tensor] = {}
        for base_rank, maps in base_maps.items():
            factor_tensors[f"base_left_b{base_rank}"] = torch.stack(
                [item.left for item in maps]
            ).float()
            factor_tensors[f"base_right_b{base_rank}"] = torch.stack(
                [item.right for item in maps]
            ).float()
            factor_tensors[f"base_bias_b{base_rank}"] = torch.stack(
                [item.bias for item in maps]
            ).float()
        for (base_rank, residual_rank), (encoder, query) in factor_bank.items():
            factor_tensors[
                f"residual_encoder_b{base_rank}_r{residual_rank}"
            ] = encoder
            factor_tensors[
                f"residual_query_b{base_rank}_r{residual_rank}"
            ] = query
        artifact_name = f"layer_{layer:03d}.safetensors"
        _atomic_safetensors(output_root / artifact_name, factor_tensors)
        layer_record = {
            "layer": layer,
            "artifact": artifact_name,
            "fit_base_reconstruction": fit_reconstruction,
            "validation_base_reconstruction": validation_reconstruction,
            "router_fit": router_diagnostics,
            "fresh": fresh,
            "reference_k_only_pagef32": reference.get(layer),
            "seconds": time.monotonic() - layer_started,
        }
        layer_records.append(layer_record)
        _atomic_json(
            progress_path,
            {
                "format": FORMAT,
                "status": "running",
                "command": shlex.join(sys.argv),
                "layers": layer_records,
                "aggregate": _aggregate(layer_records),
                "elapsed_seconds": time.monotonic() - started,
            },
        )
        del fit_statistics, validation_statistics, factor_bank
        if device.type == "cuda":
            torch.cuda.empty_cache()

    result = {
        "format": FORMAT,
        "status": "complete",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(sys.argv),
        "protocol": {
            "model": str(model_root),
            "fixed_payload": "C1-V80 ALS5",
            "predictive_base": "unregularized affine reduced-rank regression from C1-V80 to pre-RoPE K",
            "residual": "exact post-RoPE K minus exact-RoPE predictive base",
            "residual_objective": (
                f"exact-teacher Page{args.page_size} Fisher BCD"
            ),
            "fit_capture": "64 independent C4 documents x 32768 tokens; last-token query",
            "validation_capture": "16 independent C4 documents x 32768 tokens; last-token query",
            "final_capture": "fresh 4 independent C4 documents x 32768 tokens; last-token query",
            "page_size": args.page_size,
            "physical_token_budget_per_kv_group": args.exact_token_budget,
            "physical_accounting": (
                "per-query-head normalized page mass, max across each GQA "
                "group, then one fixed physical Top-page set"
            ),
            "base_ranks": list(base_ranks),
            "residual_ranks": list(residual_ranks),
            "router_sweeps": args.router_sweeps,
            "relative_damping": args.relative_damping,
            "iterative_tolerance": args.iterative_tolerance,
            "iterative_max_iterations": args.iterative_max_iterations,
        },
        "layers": layer_records,
        "aggregate": _aggregate(layer_records),
        "elapsed_seconds": time.monotonic() - started,
        "environment": {
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_devices": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ],
        },
    }
    _atomic_json(progress_path, result)
    print(f"[conditional residual] wrote {progress_path}", flush=True)


if __name__ == "__main__":
    main()
