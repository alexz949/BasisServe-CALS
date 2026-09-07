#!/usr/bin/env python3
"""Compare MSE, query-weighted, and Page-Fisher C1-V80 routing bases."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import sys
import time
from typing import Any

from safetensors import safe_open
from safetensors.torch import load_file, save_file
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basisserve.core.c1_v_conditional_k_router import (  # noqa: E402
    AffineReducedRankMap,
    fit_affine_metric_reduced_rank_map,
    fit_affine_reduced_rank_map,
)
from basisserve.core.gqa_joint_routing_payload_s80_ablation import (  # noqa: E402
    fit_page_fisher_router,
)
from basisserve.core.gqa_joint_routing_payload_s80_fisher import (  # noqa: E402
    S80CompactSoftmaxFisherRouting,
    compact_softmax_fisher_map_loss,
    page_softmax_fisher_gram,
)
from evaluation.eval_qwen3_8b_v80_conditional_residual_router import (  # noqa: E402
    _apply_base,
    _discover_capture,
    _load_direct,
    _parse_ints,
    _post_rope_rows,
    _pre_rope_rows,
    _rotary_embeddings,
    _stack_base_map,
    _value_codes,
)
from evaluation.fit_qwen3_8b_v80_base16_r8_nonsink_page32 import (  # noqa: E402
    _finish_metrics,
    _new_metrics,
    _update_metrics,
)


FORMAT = "basisserve.qwen3_8b.v80_fisher_base.v1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--calibration-root", required=True)
    parser.add_argument("--query-statistics-root", required=True)
    parser.add_argument("--fresh-direct-dir", required=True)
    parser.add_argument("--c1-v80", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--layers", default="0,13,33")
    parser.add_argument("--base-rank", type=int, default=16)
    parser.add_argument("--residual-ranks", default="0,4,8")
    parser.add_argument("--page-size", type=int, default=32)
    parser.add_argument("--pinned-prefix-pages", type=int, default=1)
    parser.add_argument("--physical-token-budget", type=int, default=2048)
    parser.add_argument("--fit-documents", type=int, default=64)
    parser.add_argument("--validation-documents", type=int, default=16)
    parser.add_argument("--fresh-documents", type=int, default=4)
    parser.add_argument("--router-sweeps", type=int, default=40)
    parser.add_argument("--relative-damping", type=float, default=1e-5)
    parser.add_argument("--iterative-tolerance", type=float, default=1e-5)
    parser.add_argument("--iterative-max-iterations", type=int, default=100)
    parser.add_argument("--query-metric-position-chunk", type=int, default=2048)
    parser.add_argument("--work-device", default="cuda")
    parser.add_argument("--torch-num-threads", type=int, default=2)
    return parser.parse_args()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_text(path: Path, contents: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(contents, encoding="utf-8")
    os.replace(temporary, path)


def _write_factors(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    save_file(tensors, str(temporary), metadata={"format": FORMAT})
    os.replace(temporary, path)


def _rotate_half(values: torch.Tensor) -> torch.Tensor:
    half = int(values.shape[-1]) // 2
    return torch.cat((-values[..., half:], values[..., :half]), dim=-1)


def _discover_query_statistics(
    root: Path,
    *,
    split: str,
    layer: int,
) -> tuple[Path, dict[str, Any]]:
    pattern = f"qwen3_8b_c1_v80_page_output_pullback_q8_s32768_chunk_*/{split}/manifest.json"
    matches = []
    for path in sorted(root.glob(pattern)):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if str(layer) in manifest["artifacts"]:
            matches.append((path.parent, manifest))
    assert len(matches) == 1
    return matches[0]


def _load_query_observations(
    root: Path,
    manifest: dict[str, Any],
    *,
    layer: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    calibration = manifest["calibration"]
    documents = int(calibration["documents"])
    queries_per_document = int(calibration["queries_per_document"])
    query_heads = int(manifest["artifacts"][str(layer)]["query_heads"])
    artifact = root / manifest["artifacts"][str(layer)]["file"]
    with safe_open(artifact, framework="pt", device="cpu") as handle:
        queries = handle.get_tensor("queries_by_head")
    assert tuple(queries.shape[:2]) == (
        query_heads,
        documents * queries_per_document,
    )
    queries = queries.permute(1, 0, 2).reshape(
        documents,
        queries_per_document,
        query_heads,
        queries.shape[-1],
    )
    positions = torch.tensor(calibration["query_positions"], dtype=torch.long)
    assert int(positions.numel()) == queries_per_document
    return queries, positions


def _check_query_alignment(
    direct_manifest: dict[str, Any],
    query_manifest: dict[str, Any],
) -> None:
    direct = direct_manifest["calibration"]
    query = query_manifest["calibration"]
    assert direct["windows_sha256"] == query_manifest["source"]["windows_sha256"]
    assert int(direct["sequence_length"]) == int(query["sequence_length"])
    assert direct["split"] == query["split"]


def _fit_predictive_bases(
    queries_raw: torch.Tensor,
    rows_raw: torch.Tensor,
    *,
    query_positions: torch.Tensor,
    value_encoder: torch.Tensor,
    base_rank: int,
    cos: torch.Tensor,
    sin: torch.Tensor,
    pinned_prefix_tokens: int,
    position_chunk: int,
    device: torch.device,
) -> tuple[
    dict[str, tuple[AffineReducedRankMap, ...]],
    torch.Tensor,
]:
    documents, tokens, groups, joint_dim = map(int, rows_raw.shape)
    query_heads = int(queries_raw.shape[2])
    heads_per_group = query_heads // groups
    head_dim = joint_dim // 2
    value_rank = int(value_encoder.shape[-1])
    input_sum = torch.zeros(groups, value_rank, dtype=torch.float64)
    target_sum = torch.zeros(groups, head_dim, dtype=torch.float64)
    input_gram = torch.zeros(groups, value_rank, value_rank, dtype=torch.float64)
    cross_gram = torch.zeros(groups, value_rank, head_dim, dtype=torch.float64)
    query_metric = torch.zeros(groups, head_dim, head_dim, dtype=torch.float64)
    query_metric_rows = torch.zeros(groups, dtype=torch.int64)
    encoder = value_encoder.to(device=device, dtype=torch.float32)
    rotary_cos = cos[0] if cos.ndim == 3 else cos
    rotary_sin = sin[0] if sin.ndim == 3 else sin

    for document in range(documents):
        print(
            f"    predictive moments document={document + 1}/{documents}",
            flush=True,
        )
        current = rows_raw[document].to(device=device, dtype=torch.float32)
        dense_value = current[..., :head_dim]
        exact_pre_key = _pre_rope_rows(current[..., head_dim:], cos, sin)
        codes = _value_codes(dense_value, encoder)
        group_codes = codes.permute(1, 0, 2)
        group_keys = exact_pre_key.permute(1, 0, 2)
        input_sum += group_codes.sum(dim=1).double().cpu()
        target_sum += group_keys.sum(dim=1).double().cpu()
        input_gram += torch.bmm(group_codes.mT, group_codes).double().cpu()
        cross_gram += torch.bmm(group_codes.mT, group_keys).double().cpu()

        for sample, query_position in enumerate(query_positions.tolist()):
            queries = queries_raw[document, sample].to(
                device=device,
                dtype=torch.float32,
            )
            rotated_queries = _rotate_half(queries)
            causal_stop = int(query_position) + 1
            for first in range(pinned_prefix_tokens, causal_stop, position_chunk):
                stop = min(causal_stop, first + position_chunk)
                effective = (
                    rotary_cos[first:stop, None, :] * queries[None, :, :]
                    - rotary_sin[first:stop, None, :] * rotated_queries[None, :, :]
                )
                for group in range(groups):
                    head_first = group * heads_per_group
                    head_stop = head_first + heads_per_group
                    selected = effective[:, head_first:head_stop].reshape(
                        -1,
                        head_dim,
                    )
                    query_metric[group] += (selected.mT @ selected).double().cpu()
                    query_metric_rows[group] += int(selected.shape[0])
        del current, dense_value, exact_pre_key, codes, group_codes, group_keys
        del queries, rotated_queries, effective, selected

    query_metric /= query_metric_rows[:, None, None]
    row_count = documents * tokens
    bases: dict[str, tuple[AffineReducedRankMap, ...]] = {}
    bases["mse16"] = tuple(
        fit_affine_reduced_rank_map(
            row_count=row_count,
            input_sum=input_sum[group],
            target_sum=target_sum[group],
            input_gram=input_gram[group],
            input_target_gram=cross_gram[group],
            rank=base_rank,
        )
        for group in range(groups)
    )
    bases["mse80"] = tuple(
        fit_affine_reduced_rank_map(
            row_count=row_count,
            input_sum=input_sum[group],
            target_sum=target_sum[group],
            input_gram=input_gram[group],
            input_target_gram=cross_gram[group],
            rank=value_rank,
        )
        for group in range(groups)
    )
    bases["q16"] = tuple(
        fit_affine_metric_reduced_rank_map(
            row_count=row_count,
            input_sum=input_sum[group],
            target_sum=target_sum[group],
            input_gram=input_gram[group],
            input_target_gram=cross_gram[group],
            target_metric=query_metric[group],
            rank=base_rank,
        )
        for group in range(groups)
    )
    bases["q80"] = tuple(
        fit_affine_metric_reduced_rank_map(
            row_count=row_count,
            input_sum=input_sum[group],
            target_sum=target_sum[group],
            input_gram=input_gram[group],
            input_target_gram=cross_gram[group],
            target_metric=query_metric[group],
            rank=value_rank,
        )
        for group in range(groups)
    )
    return bases, query_metric


def _predictive_sources(
    codes: torch.Tensor,
    *,
    base_factors: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> dict[str, torch.Tensor]:
    return {
        name: _post_rope_rows(_apply_base(codes, factors), cos, sin)
        for name, factors in base_factors.items()
    }


def _build_statistics(
    queries_raw: torch.Tensor,
    rows_raw: torch.Tensor,
    *,
    query_positions: torch.Tensor,
    source_names: tuple[str, ...],
    value_encoder: torch.Tensor,
    base_maps: dict[str, tuple[AffineReducedRankMap, ...]],
    cos: torch.Tensor,
    sin: torch.Tensor,
    page_size: int,
    pinned_prefix_pages: int,
    device: torch.device,
) -> dict[str, S80CompactSoftmaxFisherRouting]:
    documents, tokens, groups, joint_dim = map(int, rows_raw.shape)
    queries_per_document = int(queries_raw.shape[1])
    query_heads = int(queries_raw.shape[2])
    heads_per_group = query_heads // groups
    head_dim = joint_dim // 2
    first_token = pinned_prefix_pages * page_size
    joint_width = 2 * head_dim
    names = source_names
    assert names
    assert all(name in base_maps for name in names)
    grams = {
        name: torch.empty(
            query_heads,
            documents * queries_per_document,
            joint_width,
            joint_width,
            dtype=torch.float32,
        )
        for name in names
    }
    energies = {name: 0.0 for name in names}
    encoder = value_encoder.to(device=device, dtype=torch.float32)
    factors = {
        name: _stack_base_map(base_maps[name], device=device)
        for name in names
    }
    scaling = head_dim**-0.5

    for document in range(documents):
        print(
            f"    Page-Fisher statistics document={document + 1}/{documents}",
            flush=True,
        )
        current = rows_raw[document].to(device=device, dtype=torch.float32)
        codes = _value_codes(current[..., :head_dim], encoder)
        exact_key = current[..., head_dim:]
        sources = _predictive_sources(
            codes,
            base_factors=factors,
            cos=cos,
            sin=sin,
        )
        for sample, query_position in enumerate(query_positions.tolist()):
            observation = document * queries_per_document + sample
            queries = queries_raw[document, sample].to(
                device=device,
                dtype=torch.float32,
            )
            causal_stop = int(query_position) + 1
            for name in names:
                for group in range(groups):
                    head_first = group * heads_per_group
                    head_stop = head_first + heads_per_group
                    joint_rows = torch.cat(
                        (
                            sources[name][first_token:causal_stop, group],
                            exact_key[first_token:causal_stop, group],
                        ),
                        dim=-1,
                    )
                    current_grams, current_energy = page_softmax_fisher_gram(
                        queries[head_first:head_stop],
                        joint_rows,
                        value_dim=head_dim,
                        scaling=scaling,
                        page_size=page_size,
                    )
                    grams[name][head_first:head_stop, observation].copy_(
                        current_grams.float().cpu()
                    )
                    energies[name] += current_energy
        del current, queries, codes, exact_key, sources, joint_rows, current_grams

    mapping = torch.arange(query_heads, dtype=torch.long) // heads_per_group
    queries_cpu = (
        queries_raw.permute(2, 0, 1, 3)
        .reshape(query_heads, documents * queries_per_document, head_dim)
        .contiguous()
        .float()
    )
    return {
        name: S80CompactSoftmaxFisherRouting(
            queries_by_head=queries_cpu,
            fisher_grams_by_head=grams[name],
            head_to_kv_group=mapping,
            value_dim=head_dim,
            key_dim=head_dim,
            scaling=scaling,
            teacher_fisher_energy=energies[name],
        )
        for name in names
    }


def _statistics_to(
    statistics: S80CompactSoftmaxFisherRouting,
    device: torch.device,
) -> S80CompactSoftmaxFisherRouting:
    return S80CompactSoftmaxFisherRouting(
        queries_by_head=statistics.queries_by_head.to(device),
        fisher_grams_by_head=statistics.fisher_grams_by_head.to(device),
        head_to_kv_group=statistics.head_to_kv_group.to(device),
        value_dim=statistics.value_dim,
        key_dim=statistics.key_dim,
        scaling=statistics.scaling,
        teacher_fisher_energy=statistics.teacher_fisher_energy,
    )


def _selector(statistics: S80CompactSoftmaxFisherRouting) -> torch.Tensor:
    heads = int(statistics.queries_by_head.shape[0])
    result = statistics.queries_by_head.new_zeros(
        heads,
        statistics.key_dim,
        statistics.joint_dim,
    )
    result[:, :, statistics.value_dim :] = torch.eye(
        statistics.key_dim,
        dtype=result.dtype,
        device=result.device,
    )
    return result


def _identity_source_maps(
    statistics: S80CompactSoftmaxFisherRouting,
) -> torch.Tensor:
    heads = int(statistics.queries_by_head.shape[0])
    result = statistics.queries_by_head.new_zeros(
        heads,
        statistics.key_dim,
        statistics.joint_dim,
    )
    result[:, :, : statistics.value_dim] = torch.eye(
        statistics.key_dim,
        dtype=result.dtype,
        device=result.device,
    )
    return result


def _initial_router(
    statistics: S80CompactSoftmaxFisherRouting,
    *,
    rank: int,
    active_first: int,
    active_stop: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    mapping = statistics.head_to_kv_group
    groups = int(mapping.max()) + 1
    encoders = statistics.queries_by_head.new_zeros(
        groups,
        statistics.joint_dim,
        rank,
    )
    group_basis = []
    for group in range(groups):
        heads = torch.nonzero(mapping == group, as_tuple=False).flatten()
        gram = statistics.fisher_grams_by_head.index_select(0, heads).sum(dim=(0, 1))
        active = gram[active_first:active_stop, active_first:active_stop]
        _, eigenvectors = torch.linalg.eigh(0.5 * (active + active.mT))
        basis = eigenvectors[:, -rank:]
        encoders[group, active_first:active_stop] = basis
        group_basis.append(basis)
    basis_by_group = torch.stack(group_basis)
    queries = basis_by_group.index_select(0, mapping)
    return encoders, queries


def _factor_maps(
    statistics: S80CompactSoftmaxFisherRouting,
    encoder: torch.Tensor,
    query: torch.Tensor,
) -> torch.Tensor:
    expanded = encoder.index_select(0, statistics.head_to_kv_group)
    return torch.bmm(query, expanded.transpose(1, 2))


def _fit_routers(
    fit_statistics: dict[str, S80CompactSoftmaxFisherRouting],
    validation_statistics: dict[str, S80CompactSoftmaxFisherRouting],
    *,
    base_rank: int,
    residual_ranks: tuple[int, ...],
    sweeps: int,
    relative_damping: float,
    tolerance: float,
    max_iterations: int,
    device: torch.device,
) -> tuple[
    dict[str, tuple[torch.Tensor, torch.Tensor]],
    dict[tuple[str, int], tuple[torch.Tensor, torch.Tensor]],
    dict[str, dict[str, Any]],
]:
    base_factors: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    residual_factors: dict[
        tuple[str, int], tuple[torch.Tensor, torch.Tensor]
    ] = {}
    diagnostics: dict[str, dict[str, Any]] = {}

    for objective, source_name in (
        ("mse", "mse16"),
        ("q_rrr", "q16"),
        ("page_fisher", "mse80"),
    ):
        fit = _statistics_to(fit_statistics[source_name], device)
        validation = _statistics_to(validation_statistics[source_name], device)
        selector = _selector(fit)
        if objective == "page_fisher":
            initial_encoder, initial_query = _initial_router(
                fit,
                rank=base_rank,
                active_first=0,
                active_stop=fit.value_dim,
            )
            fitted_base = fit_page_fisher_router(
                fit,
                initial_routing_encoders=initial_encoder,
                initial_query_factors=initial_query,
                active_joint_rows=torch.arange(fit.value_dim, device=device),
                sweeps=sweeps,
                relative_damping=relative_damping,
                relative_tolerance=tolerance,
                max_iterations=max_iterations,
            )
            base_encoder = fitted_base.routing_encoders
            base_query = fitted_base.routing_query_factors
            base_maps = _factor_maps(fit, base_encoder, base_query)
            base_factors[objective] = (
                base_encoder.float().cpu(),
                base_query.float().cpu(),
            )
            base_sweeps = [asdict(item) for item in fitted_base.sweeps]
        else:
            base_maps = _identity_source_maps(fit)
            base_sweeps = []

        fit_base_loss = compact_softmax_fisher_map_loss(
            fit,
            proxy_maps=base_maps,
        )
        validation_base_loss = compact_softmax_fisher_map_loss(
            validation,
            proxy_maps=base_maps,
        )
        diagnostics[f"{objective}_r0"] = {
            "base_objective": objective,
            "residual_rank": 0,
            "fit_page_fisher_nmse": fit_base_loss / fit.teacher_fisher_energy,
            "validation_page_fisher_nmse": (
                validation_base_loss / validation.teacher_fisher_energy
            ),
            "base_sweeps": base_sweeps,
        }

        residual_target = selector - base_maps
        for residual_rank in residual_ranks:
            if residual_rank == 0:
                continue
            print(
                f"    fit {objective} residual rank={residual_rank}",
                flush=True,
            )
            initial_encoder, initial_query = _initial_router(
                fit,
                rank=residual_rank,
                active_first=fit.value_dim,
                active_stop=fit.joint_dim,
            )
            fitted = fit_page_fisher_router(
                fit,
                initial_routing_encoders=initial_encoder,
                initial_query_factors=initial_query,
                active_joint_rows=torch.arange(
                    fit.value_dim,
                    fit.joint_dim,
                    device=device,
                ),
                sweeps=sweeps,
                relative_damping=relative_damping,
                relative_tolerance=tolerance,
                max_iterations=max_iterations,
                target_maps=residual_target,
            )
            residual_encoder = fitted.routing_encoders
            residual_query = fitted.routing_query_factors
            total_maps = base_maps + _factor_maps(
                fit,
                residual_encoder,
                residual_query,
            )
            validation_total_maps = base_maps + _factor_maps(
                validation,
                residual_encoder,
                residual_query,
            )
            fit_loss = compact_softmax_fisher_map_loss(
                fit,
                proxy_maps=total_maps,
            )
            validation_loss = compact_softmax_fisher_map_loss(
                validation,
                proxy_maps=validation_total_maps,
            )
            residual_factors[(objective, residual_rank)] = (
                residual_encoder.float().cpu(),
                residual_query.float().cpu(),
            )
            diagnostics[f"{objective}_r{residual_rank}"] = {
                "base_objective": objective,
                "residual_rank": residual_rank,
                "fit_page_fisher_nmse": fit_loss / fit.teacher_fisher_energy,
                "validation_page_fisher_nmse": (
                    validation_loss / validation.teacher_fisher_energy
                ),
                "residual_sweeps": [asdict(item) for item in fitted.sweeps],
            }
        del fit, validation, selector, base_maps, residual_target
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return base_factors, residual_factors, diagnostics


def _factor_scores(
    queries: torch.Tensor,
    token_rows: torch.Tensor,
    *,
    encoder: torch.Tensor,
    query_factor: torch.Tensor,
) -> torch.Tensor:
    tokens, groups, _ = map(int, token_rows.shape)
    query_heads, head_dim = map(int, queries.shape)
    heads_per_group = query_heads // groups
    scores = torch.empty(
        query_heads,
        tokens,
        device=queries.device,
        dtype=torch.float32,
    )
    scaling = head_dim**-0.5
    for group in range(groups):
        first = group * heads_per_group
        stop = first + heads_per_group
        query_code = torch.einsum(
            "hd,hdr->hr",
            queries[first:stop],
            query_factor[first:stop],
        )
        token_code = token_rows[:, group] @ encoder[group]
        scores[first:stop] = scaling * (query_code @ token_code.mT)
    return scores


def _direct_scores(
    queries: torch.Tensor,
    keys: torch.Tensor,
) -> torch.Tensor:
    tokens, groups, head_dim = map(int, keys.shape)
    query_heads = int(queries.shape[0])
    heads_per_group = query_heads // groups
    scores = torch.empty(
        query_heads,
        tokens,
        device=queries.device,
        dtype=torch.float32,
    )
    scaling = head_dim**-0.5
    for group in range(groups):
        first = group * heads_per_group
        stop = first + heads_per_group
        scores[first:stop] = scaling * (
            queries[first:stop] @ keys[:, group].mT
        )
    return scores


def _evaluate_fresh(
    queries_raw: torch.Tensor,
    rows_raw: torch.Tensor,
    *,
    value_encoder: torch.Tensor,
    output_decoder: torch.Tensor,
    base_maps: dict[str, tuple[AffineReducedRankMap, ...]],
    page_fisher_base: tuple[torch.Tensor, torch.Tensor],
    residual_factors: dict[
        tuple[str, int], tuple[torch.Tensor, torch.Tensor]
    ],
    residual_ranks: tuple[int, ...],
    cos: torch.Tensor,
    sin: torch.Tensor,
    page_size: int,
    page_budget: int,
    pinned_prefix_pages: int,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    documents, _, groups, joint_dim = map(int, rows_raw.shape)
    query_heads = int(queries_raw.shape[1])
    head_dim = joint_dim // 2
    heads_per_group = query_heads // groups
    head_to_group = torch.arange(query_heads, device=device) // heads_per_group
    encoder = value_encoder.to(device=device, dtype=torch.float32)
    decoder = output_decoder.to(device=device, dtype=torch.float32)
    predictive_factors = {
        name: _stack_base_map(maps, device=device)
        for name, maps in base_maps.items()
    }
    pf_encoder, pf_query = (
        item.to(device=device, dtype=torch.float32)
        for item in page_fisher_base
    )
    pf_source_encoder = pf_encoder[:, :head_dim]
    route_factors = {
        key: tuple(
            item.to(device=device, dtype=torch.float32) for item in values
        )
        for key, values in residual_factors.items()
    }
    arms = [
        f"{objective}_r{rank}"
        for objective in ("mse", "q_rrr", "page_fisher")
        for rank in residual_ranks
    ] + ["mse80_r0", "q_rrr80_r0"]
    accumulators = {name: _new_metrics() for name in arms}

    for document in range(documents):
        print(
            f"    fresh evaluation document={document + 1}/{documents}",
            flush=True,
        )
        current = rows_raw[document].to(device=device, dtype=torch.float32)
        queries = queries_raw[document].to(device=device, dtype=torch.float32)
        dense_value = current[..., :head_dim]
        exact_key = current[..., head_dim:]
        codes = _value_codes(dense_value, encoder)
        sources = _predictive_sources(
            codes,
            base_factors=predictive_factors,
            cos=cos,
            sin=sin,
        )
        exact_scores = _direct_scores(queries, exact_key)
        exact_probabilities = torch.softmax(exact_scores, dim=-1)
        head_codes = codes.permute(1, 0, 2).index_select(0, head_to_group)
        dense_latent = torch.einsum("ht,htr->hr", exact_probabilities, head_codes)
        dense_output = torch.einsum("hr,hro->o", dense_latent, decoder)
        base_scores = {
            "mse": _direct_scores(queries, sources["mse16"]),
            "q_rrr": _direct_scores(queries, sources["q16"]),
            "page_fisher": _factor_scores(
                queries,
                sources["mse80"],
                encoder=pf_source_encoder,
                query_factor=pf_query,
            ),
        }
        for objective in ("mse", "q_rrr", "page_fisher"):
            for residual_rank in residual_ranks:
                proxy_scores = base_scores[objective]
                if residual_rank:
                    residual_encoder, residual_query = route_factors[
                        (objective, residual_rank)
                    ]
                    proxy_scores = proxy_scores + _factor_scores(
                        queries,
                        exact_key,
                        encoder=residual_encoder[:, head_dim:],
                        query_factor=residual_query,
                    )
                _update_metrics(
                    accumulators[f"{objective}_r{residual_rank}"],
                    proxy_scores=proxy_scores,
                    exact_scores=exact_scores,
                    exact_probabilities=exact_probabilities,
                    head_codes=head_codes,
                    output_decoder=decoder,
                    dense_output=dense_output,
                    num_kv_heads=groups,
                    page_size=page_size,
                    page_budget=page_budget,
                    pinned_prefix_pages=pinned_prefix_pages,
                )
        for name, source_name in (("mse80_r0", "mse80"), ("q_rrr80_r0", "q80")):
            _update_metrics(
                accumulators[name],
                proxy_scores=_direct_scores(queries, sources[source_name]),
                exact_scores=exact_scores,
                exact_probabilities=exact_probabilities,
                head_codes=head_codes,
                output_decoder=decoder,
                dense_output=dense_output,
                num_kv_heads=groups,
                page_size=page_size,
                page_budget=page_budget,
                pinned_prefix_pages=pinned_prefix_pages,
            )
        del current, queries, dense_value, exact_key, codes, sources
        del exact_scores, exact_probabilities, head_codes, dense_latent, dense_output
    return {
        name: _finish_metrics(accumulator, tokens=int(rows_raw.shape[1]))
        for name, accumulator in accumulators.items()
    }


def _markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Qwen3-8B C1-V80 Fisher-Base Diagnostic",
        "",
        "| Layer | Base | Residual | Validation Fisher NMSE | Selected mass | P01 mass | Page recall | Output rel-MSE |",
        "|---:|:---|---:|---:|---:|---:|---:|---:|",
    ]
    for layer in result.get("layers", []):
        for name in sorted(layer["fresh"]):
            metric = layer["fresh"][name]
            fit = layer.get("router_fit", {}).get(name, {})
            objective, rank = name.rsplit("_r", 1)
            validation_nmse = fit.get("validation_page_fisher_nmse")
            validation_cell = (
                "—"
                if validation_nmse is None
                else f"{float(validation_nmse):.6f}"
            )
            lines.append(
                f"| {layer['layer']} | {objective} | {rank} | "
                f"{validation_cell} | "
                f"{metric['attention_mass_recall_mean']:.6f} | "
                f"{metric['attention_mass_recall_p01']:.6f} | "
                f"{metric['non_sink_physical_page_recall_mean']:.6f} | "
                f"{metric['exact_refined_output_relative_mse']:.6f} |"
            )
    lines.append("")
    return "\n".join(lines)


@torch.inference_mode()
def main() -> None:
    args = _parse_args()
    torch.set_num_threads(args.torch_num_threads)
    torch.backends.cuda.matmul.allow_tf32 = True
    started = time.monotonic()
    model_root = Path(args.model).expanduser().resolve()
    calibration_root = Path(args.calibration_root).expanduser().resolve()
    query_statistics_root = Path(args.query_statistics_root).expanduser().resolve()
    fresh_root = Path(args.fresh_direct_dir).expanduser().resolve()
    c1_root = Path(args.c1_v80).expanduser().resolve()
    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.work_device)
    layers = _parse_ints(args.layers)
    residual_ranks = _parse_ints(args.residual_ranks)
    page_budget = args.physical_token_budget // args.page_size
    sequence = 32768
    cos, sin = _rotary_embeddings(
        model_root,
        sequence=sequence,
        device=device,
    )
    c1_manifest = json.loads((c1_root / "results.json").read_text(encoding="utf-8"))
    fresh_manifest = json.loads((fresh_root / "manifest.json").read_text(encoding="utf-8"))
    layer_records = []
    result_path = output_root / "result.json"
    summary_path = output_root / "summary.md"

    for ordinal, layer in enumerate(layers, start=1):
        layer_started = time.monotonic()
        print(
            f"[Fisher Base] layer={layer} ({ordinal}/{len(layers)})",
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
        _, fit_rows = _load_direct(fit_root, fit_manifest, layer)
        _, validation_rows = _load_direct(
            validation_root,
            validation_manifest,
            layer,
        )
        fresh_queries, fresh_rows = _load_direct(fresh_root, fresh_manifest, layer)
        fit_query_root, fit_query_manifest = _discover_query_statistics(
            query_statistics_root,
            split="fit",
            layer=layer,
        )
        validation_query_root, validation_query_manifest = _discover_query_statistics(
            query_statistics_root,
            split="validation",
            layer=layer,
        )
        _check_query_alignment(fit_manifest, fit_query_manifest)
        _check_query_alignment(validation_manifest, validation_query_manifest)
        fit_queries, fit_query_positions = _load_query_observations(
            fit_query_root,
            fit_query_manifest,
            layer=layer,
        )
        validation_queries, validation_query_positions = _load_query_observations(
            validation_query_root,
            validation_query_manifest,
            layer=layer,
        )
        assert torch.equal(fit_query_positions, validation_query_positions)
        fit_queries = fit_queries[: args.fit_documents]
        fit_rows = fit_rows[: args.fit_documents]
        validation_queries = validation_queries[: args.validation_documents]
        validation_rows = validation_rows[: args.validation_documents]
        fresh_queries = fresh_queries[: args.fresh_documents]
        fresh_rows = fresh_rows[: args.fresh_documents]
        c1_artifact = c1_manifest["artifacts"][str(layer)]
        c1_tensors = load_file(str(c1_root / c1_artifact["file"]), device="cpu")
        value_encoder = c1_tensors["value_coordinate_encoders"]
        output_decoder = c1_tensors["head_output_decoders"]

        base_maps, query_metric = _fit_predictive_bases(
            fit_queries,
            fit_rows,
            query_positions=fit_query_positions,
            value_encoder=value_encoder,
            base_rank=args.base_rank,
            cos=cos,
            sin=sin,
            pinned_prefix_tokens=args.pinned_prefix_pages * args.page_size,
            position_chunk=args.query_metric_position_chunk,
            device=device,
        )
        fit_statistics = _build_statistics(
            fit_queries,
            fit_rows,
            query_positions=fit_query_positions,
            source_names=("mse16", "q16", "mse80"),
            value_encoder=value_encoder,
            base_maps=base_maps,
            cos=cos,
            sin=sin,
            page_size=args.page_size,
            pinned_prefix_pages=args.pinned_prefix_pages,
            device=device,
        )
        validation_statistics = _build_statistics(
            validation_queries,
            validation_rows,
            query_positions=validation_query_positions,
            source_names=("mse16", "q16", "mse80"),
            value_encoder=value_encoder,
            base_maps=base_maps,
            cos=cos,
            sin=sin,
            page_size=args.page_size,
            pinned_prefix_pages=args.pinned_prefix_pages,
            device=device,
        )
        fisher_base, residual_factors, router_fit = _fit_routers(
            fit_statistics,
            validation_statistics,
            base_rank=args.base_rank,
            residual_ranks=residual_ranks,
            sweeps=args.router_sweeps,
            relative_damping=args.relative_damping,
            tolerance=args.iterative_tolerance,
            max_iterations=args.iterative_max_iterations,
            device=device,
        )
        fresh = _evaluate_fresh(
            fresh_queries,
            fresh_rows,
            value_encoder=value_encoder,
            output_decoder=output_decoder,
            base_maps=base_maps,
            page_fisher_base=fisher_base["page_fisher"],
            residual_factors=residual_factors,
            residual_ranks=residual_ranks,
            cos=cos,
            sin=sin,
            page_size=args.page_size,
            page_budget=page_budget,
            pinned_prefix_pages=args.pinned_prefix_pages,
            device=device,
        )

        factor_tensors: dict[str, torch.Tensor] = {"query_metric": query_metric.float()}
        for name, maps in base_maps.items():
            factor_tensors[f"{name}_left"] = torch.stack(
                [item.left for item in maps]
            ).float()
            factor_tensors[f"{name}_right"] = torch.stack(
                [item.right for item in maps]
            ).float()
            factor_tensors[f"{name}_bias"] = torch.stack(
                [item.bias for item in maps]
            ).float()
        for objective, (route_encoder, route_query) in fisher_base.items():
            factor_tensors[f"{objective}_base_encoder"] = route_encoder
            factor_tensors[f"{objective}_base_query"] = route_query
        for (objective, rank), (route_encoder, route_query) in residual_factors.items():
            factor_tensors[f"{objective}_residual_r{rank}_encoder"] = route_encoder
            factor_tensors[f"{objective}_residual_r{rank}_query"] = route_query
        artifact = f"layer_{layer:03d}.safetensors"
        _write_factors(output_root / artifact, factor_tensors)
        layer_records.append(
            {
                "layer": layer,
                "artifact": artifact,
                "router_fit": router_fit,
                "fresh": fresh,
                "seconds": time.monotonic() - layer_started,
            }
        )
        partial = {
            "format": FORMAT,
            "status": "running",
            "command": shlex.join(sys.argv),
            "layers": layer_records,
            "elapsed_seconds": time.monotonic() - started,
        }
        _write_json(result_path, partial)
        _write_text(summary_path, _markdown(partial))
        del fit_statistics, validation_statistics, residual_factors, c1_tensors
        if device.type == "cuda":
            torch.cuda.empty_cache()

    result = {
        "format": FORMAT,
        "status": "complete",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(sys.argv),
        "protocol": {
            "model": str(model_root),
            "fixed_payload": str(c1_root),
            "base_objectives": {
                "mse": "rank-16 affine K-MSE RRR to pre-RoPE K plus exact RoPE",
                "q_rrr": "rank-16 affine RRR in the mean inverse-RoPE query covariance",
                "page_fisher": (
                    "rank-16 score factorization of the full V-conditioned Key "
                    "under exact-teacher non-sink Page32 Fisher"
                ),
            },
            "residual": (
                "rank-r exact-K score sidecar fitted additively after each frozen base"
            ),
            "residual_ranks": list(residual_ranks),
            "fit_documents": int(fit_queries.shape[0]),
            "validation_documents": int(validation_queries.shape[0]),
            "fresh_documents": int(fresh_queries.shape[0]),
            "queries_per_fit_document": int(fit_queries.shape[1]),
            "queries_per_validation_document": int(validation_queries.shape[1]),
            "query_positions": fit_query_positions.tolist(),
            "sequence_length": sequence,
            "page_size": args.page_size,
            "pinned_prefix_pages": args.pinned_prefix_pages,
            "physical_token_budget_per_kv_group": args.physical_token_budget,
            "router_sweeps": args.router_sweeps,
            "relative_damping": args.relative_damping,
            "iterative_tolerance": args.iterative_tolerance,
            "iterative_max_iterations": args.iterative_max_iterations,
        },
        "layers": layer_records,
        "elapsed_seconds": time.monotonic() - started,
        "environment": {
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "python_executable": sys.executable,
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_devices": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ],
        },
    }
    _write_json(result_path, result)
    _write_text(summary_path, _markdown(result))
    print(f"[Fisher Base] wrote {result_path} and {summary_path}", flush=True)


if __name__ == "__main__":
    main()
