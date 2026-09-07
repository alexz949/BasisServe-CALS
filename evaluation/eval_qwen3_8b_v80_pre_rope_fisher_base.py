#!/usr/bin/env python3
"""Fit a fair pre-RoPE C1-V80 Base16 with multi-query Page-Fisher."""

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

from safetensors.torch import load_file, save_file
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basisserve.core.c1_v_conditional_k_router import (  # noqa: E402
    AffineReducedRankMap,
)
from basisserve.core.gqa_joint_routing_payload_s80_ablation import (  # noqa: E402
    fit_page_fisher_router,
)
from basisserve.core.gqa_joint_routing_payload_s80_fisher import (  # noqa: E402
    S80CompactSoftmaxFisherRouting,
    compact_softmax_fisher_map_loss,
)
from evaluation.eval_qwen3_8b_v80_conditional_residual_router import (  # noqa: E402
    _discover_capture,
    _load_direct,
    _parse_ints,
    _post_rope_rows,
    _rotary_embeddings,
    _stack_base_map,
    _value_codes,
)
from evaluation.eval_qwen3_8b_v80_fisher_base import (  # noqa: E402
    _build_statistics,
    _check_query_alignment,
    _discover_query_statistics,
    _factor_maps,
    _identity_source_maps,
    _initial_router,
    _load_query_observations,
    _selector,
    _statistics_to,
)
from evaluation.fit_qwen3_8b_v80_base16_r8_nonsink_page32 import (  # noqa: E402
    _finish_metrics,
    _new_metrics,
    _update_metrics,
)


FORMAT = "basisserve.qwen3_8b.v80_pre_rope_fisher_base.v1"
SOURCE_NAME = "pre_rope_fisher16"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--calibration-root", required=True)
    parser.add_argument("--query-statistics-root", required=True)
    parser.add_argument("--fresh-direct-dir", required=True)
    parser.add_argument("--c1-v80", required=True)
    parser.add_argument("--mse-initialization-root", required=True)
    parser.add_argument("--reference-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--layers", default="0,13,33")
    parser.add_argument("--residual-ranks", default="0,4,8")
    parser.add_argument("--page-size", type=int, default=32)
    parser.add_argument("--pinned-prefix-pages", type=int, default=1)
    parser.add_argument("--physical-token-budget", type=int, default=2048)
    parser.add_argument("--fit-documents", type=int, default=64)
    parser.add_argument("--validation-documents", type=int, default=16)
    parser.add_argument("--fresh-documents", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--documents-per-step", type=int, default=4)
    parser.add_argument("--factor-learning-rate", type=float, default=2e-3)
    parser.add_argument("--bias-learning-rate", type=float, default=5e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--early-stopping-patience", type=int, default=4)
    parser.add_argument("--router-sweeps", type=int, default=40)
    parser.add_argument("--relative-damping", type=float, default=1e-5)
    parser.add_argument("--iterative-tolerance", type=float, default=1e-5)
    parser.add_argument("--iterative-max-iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=73)
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


def _initial_maps(root: Path, layer: int) -> tuple[AffineReducedRankMap, ...]:
    artifact = root / f"layer_{layer}" / f"layer_{layer:03d}.safetensors"
    tensors = load_file(str(artifact), device="cpu")
    left = tensors["mse16_left"].float()
    right = tensors["mse16_right"].float()
    bias = tensors["mse16_bias"].float()
    return tuple(
        AffineReducedRankMap(
            left=left[group],
            right=right[group],
            bias=bias[group],
        )
        for group in range(left.shape[0])
    )


def _balanced_factors(
    maps: tuple[AffineReducedRankMap, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    left_parts = []
    right_parts = []
    for item in maps:
        matrix = item.left.double() @ item.right.double()
        left_singular, singular_values, right_singular = torch.linalg.svd(
            matrix,
            full_matrices=False,
        )
        rank = int(item.left.shape[-1])
        roots = torch.sqrt(singular_values[:rank].clamp_min(0))
        left_parts.append((left_singular[:, :rank] * roots).float())
        right_parts.append((roots[:, None] * right_singular[:rank]).float())
    return (
        torch.stack(left_parts),
        torch.stack(right_parts),
        torch.stack([item.bias.float() for item in maps]),
    )


def _gqa_scores(queries: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
    tokens, groups, head_dim = map(int, keys.shape)
    query_heads = int(queries.shape[0])
    heads_per_group = query_heads // groups
    grouped_queries = queries.reshape(groups, heads_per_group, head_dim)
    return (
        torch.einsum("ghd,tgd->ght", grouped_queries, keys)
        .reshape(query_heads, tokens)
        .mul(head_dim**-0.5)
    )


def _page_fisher_document_loss(
    queries: torch.Tensor,
    rows: torch.Tensor,
    *,
    query_positions: torch.Tensor,
    value_encoder: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    bias: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    page_size: int,
    pinned_prefix_pages: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    head_dim = int(rows.shape[-1]) // 2
    first_token = int(page_size) * int(pinned_prefix_pages)
    codes = _value_codes(rows[..., :head_dim], value_encoder)
    base_codes = torch.einsum("tgv,gvr->tgr", codes, left)
    predicted_pre_key = torch.einsum("tgr,grd->tgd", base_codes, right)
    predicted_pre_key = predicted_pre_key + bias.unsqueeze(0)
    predicted_key = _post_rope_rows(predicted_pre_key, cos, sin)
    exact_key = rows[..., head_dim:]
    total_loss = predicted_key.new_zeros(())
    total_energy = predicted_key.new_zeros(())

    for sample, query_position in enumerate(query_positions.tolist()):
        causal_stop = int(query_position) + 1
        current_queries = queries[sample]
        predicted_scores = _gqa_scores(
            current_queries,
            predicted_key[first_token:causal_stop],
        )
        with torch.no_grad():
            exact_scores = _gqa_scores(
                current_queries,
                exact_key[first_token:causal_stop],
            )
            probabilities = torch.softmax(exact_scores, dim=-1)
        visible = int(predicted_scores.shape[-1])
        pages = (visible + int(page_size) - 1) // int(page_size)
        padding = pages * int(page_size) - visible
        if padding:
            probabilities = F.pad(probabilities, (0, padding))
            exact_scores = F.pad(exact_scores, (0, padding))
            predicted_scores = F.pad(predicted_scores, (0, padding))
        probabilities = probabilities.reshape(-1, pages, int(page_size))
        exact_scores = exact_scores.reshape(-1, pages, int(page_size))
        score_error = (predicted_scores - exact_scores.reshape_as(predicted_scores))
        score_error = score_error.reshape(-1, pages, int(page_size))
        page_mass = probabilities.sum(dim=-1)
        inverse_mass = page_mass.clamp_min(
            torch.finfo(page_mass.dtype).tiny
        ).reciprocal()
        page_error = (probabilities * score_error).sum(dim=-1) * inverse_mass
        page_score = (probabilities * exact_scores).sum(dim=-1) * inverse_mass
        mean_error = (page_mass * page_error).sum(dim=-1, keepdim=True)
        mean_score = (page_mass * page_score).sum(dim=-1, keepdim=True)
        total_loss = total_loss + 0.5 * (
            page_mass * (page_error - mean_error).square()
        ).sum()
        total_energy = total_energy + 0.5 * (
            page_mass * (page_score - mean_score).square()
        ).sum()
    return total_loss, total_energy


@torch.no_grad()
def _objective(
    queries: torch.Tensor,
    rows: torch.Tensor,
    *,
    query_positions: torch.Tensor,
    value_encoder: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    bias: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    page_size: int,
    pinned_prefix_pages: int,
    device: torch.device,
    document_loss: Any = _page_fisher_document_loss,
) -> tuple[float, float]:
    loss = 0.0
    energy = 0.0
    for document in range(int(rows.shape[0])):
        current_loss, current_energy = document_loss(
            queries[document].to(device=device, dtype=torch.float32),
            rows[document].to(device=device, dtype=torch.float32),
            query_positions=query_positions,
            value_encoder=value_encoder,
            left=left,
            right=right,
            bias=bias,
            cos=cos,
            sin=sin,
            page_size=page_size,
            pinned_prefix_pages=pinned_prefix_pages,
        )
        loss += float(current_loss)
        energy += float(current_energy)
    return loss, energy


def _train_base(
    fit_queries: torch.Tensor,
    fit_rows: torch.Tensor,
    validation_queries: torch.Tensor,
    validation_rows: torch.Tensor,
    *,
    query_positions: torch.Tensor,
    value_encoder: torch.Tensor,
    initial_maps: tuple[AffineReducedRankMap, ...],
    cos: torch.Tensor,
    sin: torch.Tensor,
    page_size: int,
    pinned_prefix_pages: int,
    epochs: int,
    documents_per_step: int,
    factor_learning_rate: float,
    bias_learning_rate: float,
    gradient_clip: float,
    patience: int,
    seed: int,
    device: torch.device,
    document_loss: Any = _page_fisher_document_loss,
    metric_name: str = "page_fisher",
) -> tuple[tuple[AffineReducedRankMap, ...], list[dict[str, float]], int]:
    initial_left, initial_right, initial_bias = _balanced_factors(initial_maps)
    left = torch.nn.Parameter(initial_left.to(device))
    right = torch.nn.Parameter(initial_right.to(device))
    bias = torch.nn.Parameter(initial_bias.to(device))
    encoder = value_encoder.to(device=device, dtype=torch.float32)
    optimizer = torch.optim.Adam(
        (
            {"params": (left, right), "lr": float(factor_learning_rate)},
            {"params": (bias,), "lr": float(bias_learning_rate)},
        )
    )
    fit_loss, fit_energy = _objective(
        fit_queries,
        fit_rows,
        query_positions=query_positions,
        value_encoder=encoder,
        left=left,
        right=right,
        bias=bias,
        cos=cos,
        sin=sin,
        page_size=page_size,
        pinned_prefix_pages=pinned_prefix_pages,
        device=device,
        document_loss=document_loss,
    )
    validation_loss, validation_energy = _objective(
        validation_queries,
        validation_rows,
        query_positions=query_positions,
        value_encoder=encoder,
        left=left,
        right=right,
        bias=bias,
        cos=cos,
        sin=sin,
        page_size=page_size,
        pinned_prefix_pages=pinned_prefix_pages,
        device=device,
        document_loss=document_loss,
    )
    history = [
        {
            "epoch": 0,
            f"fit_{metric_name}_nmse": fit_loss / fit_energy,
            f"validation_{metric_name}_nmse": validation_loss / validation_energy,
        }
    ]
    best_epoch = 0
    best_validation = validation_loss / validation_energy
    best = (
        left.detach().cpu().clone(),
        right.detach().cpu().clone(),
        bias.detach().cpu().clone(),
    )
    stale = 0
    fit_documents = int(fit_rows.shape[0])
    assert fit_documents % int(documents_per_step) == 0
    random = torch.Generator().manual_seed(int(seed))
    gradient_scale = fit_documents / max(fit_energy, torch.finfo(torch.float64).tiny)

    for epoch in range(1, int(epochs) + 1):
        print(f"    pre-RoPE {metric_name} epoch={epoch}/{epochs}", flush=True)
        ordering = torch.randperm(fit_documents, generator=random).tolist()
        optimizer.zero_grad(set_to_none=True)
        for ordinal, document in enumerate(ordering, start=1):
            current_loss, _ = document_loss(
                fit_queries[document].to(device=device, dtype=torch.float32),
                fit_rows[document].to(device=device, dtype=torch.float32),
                query_positions=query_positions,
                value_encoder=encoder,
                left=left,
                right=right,
                bias=bias,
                cos=cos,
                sin=sin,
                page_size=page_size,
                pinned_prefix_pages=pinned_prefix_pages,
            )
            scaled = (
                current_loss
                * gradient_scale
                / float(documents_per_step)
            )
            scaled.backward()
            if ordinal % int(documents_per_step) == 0:
                torch.nn.utils.clip_grad_norm_(
                    (left, right, bias),
                    max_norm=float(gradient_clip),
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        validation_loss, validation_energy = _objective(
            validation_queries,
            validation_rows,
            query_positions=query_positions,
            value_encoder=encoder,
            left=left,
            right=right,
            bias=bias,
            cos=cos,
            sin=sin,
            page_size=page_size,
            pinned_prefix_pages=pinned_prefix_pages,
            device=device,
            document_loss=document_loss,
        )
        validation_nmse = validation_loss / validation_energy
        history.append(
            {
                "epoch": epoch,
                f"validation_{metric_name}_nmse": validation_nmse,
            }
        )
        print(f"      validation {metric_name} NMSE={validation_nmse:.6f}", flush=True)
        if validation_nmse < best_validation * (1.0 - 1e-5):
            best_validation = validation_nmse
            best_epoch = epoch
            best = (
                left.detach().cpu().clone(),
                right.detach().cpu().clone(),
                bias.detach().cpu().clone(),
            )
            stale = 0
        else:
            stale += 1
            if stale >= int(patience):
                break

    best_left, best_right, best_bias = best
    left.data.copy_(best_left.to(device))
    right.data.copy_(best_right.to(device))
    bias.data.copy_(best_bias.to(device))
    final_fit_loss, final_fit_energy = _objective(
        fit_queries,
        fit_rows,
        query_positions=query_positions,
        value_encoder=encoder,
        left=left,
        right=right,
        bias=bias,
        cos=cos,
        sin=sin,
        page_size=page_size,
        pinned_prefix_pages=pinned_prefix_pages,
        device=device,
        document_loss=document_loss,
    )
    history[best_epoch][f"fit_{metric_name}_nmse"] = (
        final_fit_loss / final_fit_energy
    )
    maps = tuple(
        AffineReducedRankMap(
            left=best_left[group],
            right=best_right[group],
            bias=best_bias[group],
        )
        for group in range(best_left.shape[0])
    )
    return maps, history, best_epoch


@torch.inference_mode()
def _fit_residuals(
    fit: S80CompactSoftmaxFisherRouting,
    validation: S80CompactSoftmaxFisherRouting,
    *,
    residual_ranks: tuple[int, ...],
    sweeps: int,
    relative_damping: float,
    tolerance: float,
    max_iterations: int,
    device: torch.device,
) -> tuple[
    dict[int, tuple[torch.Tensor, torch.Tensor]],
    dict[str, dict[str, Any]],
]:
    fit_device = _statistics_to(fit, device)
    validation_device = _statistics_to(validation, device)
    base_maps = _identity_source_maps(fit_device)
    selector = _selector(fit_device)
    target = selector - base_maps
    factors = {}
    diagnostics = {
        "r0": {
            "fit_page_fisher_nmse": compact_softmax_fisher_map_loss(
                fit_device,
                proxy_maps=base_maps,
            )
            / fit_device.teacher_fisher_energy,
            "validation_page_fisher_nmse": compact_softmax_fisher_map_loss(
                validation_device,
                proxy_maps=base_maps,
            )
            / validation_device.teacher_fisher_energy,
        }
    }
    for rank in residual_ranks:
        if rank == 0:
            continue
        print(f"    fit exact post-RoPE K residual rank={rank}", flush=True)
        initial_encoder, initial_query = _initial_router(
            fit_device,
            rank=rank,
            active_first=fit_device.value_dim,
            active_stop=fit_device.joint_dim,
        )
        fitted = fit_page_fisher_router(
            fit_device,
            initial_routing_encoders=initial_encoder,
            initial_query_factors=initial_query,
            active_joint_rows=torch.arange(
                fit_device.value_dim,
                fit_device.joint_dim,
                device=device,
            ),
            sweeps=sweeps,
            relative_damping=relative_damping,
            relative_tolerance=tolerance,
            max_iterations=max_iterations,
            target_maps=target,
        )
        residual_maps = _factor_maps(
            fit_device,
            fitted.routing_encoders,
            fitted.routing_query_factors,
        )
        total_maps = base_maps + residual_maps
        validation_maps = base_maps + _factor_maps(
            validation_device,
            fitted.routing_encoders,
            fitted.routing_query_factors,
        )
        factors[rank] = (
            fitted.routing_encoders.float().cpu(),
            fitted.routing_query_factors.float().cpu(),
        )
        diagnostics[f"r{rank}"] = {
            "fit_page_fisher_nmse": compact_softmax_fisher_map_loss(
                fit_device,
                proxy_maps=total_maps,
            )
            / fit_device.teacher_fisher_energy,
            "validation_page_fisher_nmse": compact_softmax_fisher_map_loss(
                validation_device,
                proxy_maps=validation_maps,
            )
            / validation_device.teacher_fisher_energy,
            "sweeps": [asdict(item) for item in fitted.sweeps],
        }
    return factors, diagnostics


@torch.inference_mode()
def _evaluate_fresh(
    queries_raw: torch.Tensor,
    rows_raw: torch.Tensor,
    *,
    value_encoder: torch.Tensor,
    output_decoder: torch.Tensor,
    base_maps: tuple[AffineReducedRankMap, ...],
    residual_factors: dict[int, tuple[torch.Tensor, torch.Tensor]],
    residual_ranks: tuple[int, ...],
    cos: torch.Tensor,
    sin: torch.Tensor,
    page_size: int,
    page_budget: int,
    pinned_prefix_pages: int,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    documents, query_heads, head_dim = map(int, queries_raw.shape)
    _, tokens, groups, _ = map(int, rows_raw.shape)
    heads_per_group = query_heads // groups
    head_to_group = torch.arange(query_heads, device=device) // heads_per_group
    encoder = value_encoder.to(device=device, dtype=torch.float32)
    decoder = output_decoder.to(device=device, dtype=torch.float32)
    base_factors = _stack_base_map(base_maps, device=device)
    route_factors = {
        rank: tuple(item.to(device=device, dtype=torch.float32) for item in values)
        for rank, values in residual_factors.items()
    }
    accumulators = {f"r{rank}": _new_metrics() for rank in residual_ranks}

    for document in range(documents):
        print(f"    fresh evaluation document={document + 1}/{documents}", flush=True)
        current = rows_raw[document].to(device=device, dtype=torch.float32)
        queries = queries_raw[document].to(device=device, dtype=torch.float32)
        dense_value = current[..., :head_dim]
        exact_key = current[..., head_dim:]
        codes = _value_codes(dense_value, encoder)
        left, right, bias = base_factors
        base_pre = torch.einsum("tgv,gvr,grd->tgd", codes, left, right)
        base_pre = base_pre + bias.unsqueeze(0)
        base_key = _post_rope_rows(base_pre, cos, sin)
        exact_scores = _gqa_scores(queries, exact_key)
        base_scores = _gqa_scores(queries, base_key)
        exact_probabilities = torch.softmax(exact_scores, dim=-1)
        head_codes = codes.permute(1, 0, 2).index_select(0, head_to_group)
        dense_latent = torch.einsum("ht,htr->hr", exact_probabilities, head_codes)
        dense_output = torch.einsum("hr,hro->o", dense_latent, decoder)
        for rank in residual_ranks:
            proxy_scores = base_scores
            if rank:
                residual_encoder, residual_query = route_factors[rank]
                query_code = torch.einsum("hd,hdr->hr", queries, residual_query)
                token_code = torch.einsum(
                    "tgd,gdr->tgr",
                    exact_key,
                    residual_encoder[:, head_dim:],
                )
                head_token_code = token_code.permute(1, 0, 2).index_select(
                    0,
                    head_to_group,
                )
                proxy_scores = base_scores + head_dim**-0.5 * torch.einsum(
                    "hr,htr->ht",
                    query_code,
                    head_token_code,
                )
            _update_metrics(
                accumulators[f"r{rank}"],
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
    return {
        name: _finish_metrics(accumulator, tokens=tokens)
        for name, accumulator in accumulators.items()
    }


def _markdown(result: dict[str, Any]) -> str:
    layer = result["layers"][0]
    lines = [
        "# Qwen3-8B C1-V80 Fair Pre-RoPE Fisher Base",
        "",
        f"Layer: {layer['layer']}",
        "",
        f"Best epoch: {layer['base_fit']['best_epoch']}",
        "",
        "| Epoch | Fit Page-Fisher NMSE | Validation Page-Fisher NMSE |",
        "|---:|---:|---:|",
    ]
    for row in layer["base_fit"]["history"]:
        fit = row.get("fit_page_fisher_nmse")
        fit_cell = "—" if fit is None else f"{fit:.6f}"
        lines.append(
            f"| {row['epoch']} | {fit_cell} | "
            f"{row['validation_page_fisher_nmse']:.6f} |"
        )
    lines.extend(
        (
            "",
            "| Method | Residual | Selected mass | P01 mass | Page recall | Output rel-MSE |",
            "|:---|---:|---:|---:|---:|---:|",
        )
    )
    reference = layer["reference"]
    for name in ("mse_r0", "mse_r4", "mse_r8", "page_fisher_r0", "page_fisher_r4", "page_fisher_r8"):
        if name not in reference:
            continue
        metric = reference[name]
        method, rank = name.rsplit("_r", 1)
        lines.append(
            f"| Reference {method} | {rank} | "
            f"{metric['attention_mass_recall_mean']:.6f} | "
            f"{metric['attention_mass_recall_p01']:.6f} | "
            f"{metric['non_sink_physical_page_recall_mean']:.6f} | "
            f"{metric['exact_refined_output_relative_mse']:.6f} |"
        )
    for name, metric in sorted(layer["fresh"].items()):
        rank = name.removeprefix("r")
        lines.append(
            f"| Fair pre-RoPE Fisher | {rank} | "
            f"{metric['attention_mass_recall_mean']:.6f} | "
            f"{metric['attention_mass_recall_p01']:.6f} | "
            f"{metric['non_sink_physical_page_recall_mean']:.6f} | "
            f"{metric['exact_refined_output_relative_mse']:.6f} |"
        )
    lines.append("")
    return "\n".join(lines)


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
    initialization_root = Path(args.mse_initialization_root).expanduser().resolve()
    reference_root = Path(args.reference_root).expanduser().resolve()
    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.work_device)
    layers = _parse_ints(args.layers)
    residual_ranks = _parse_ints(args.residual_ranks)
    assert 0 in residual_ranks
    assert len(layers) == 1
    layer = layers[0]
    page_budget = args.physical_token_budget // args.page_size
    cos, sin = _rotary_embeddings(model_root, sequence=32768, device=device)

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
    _, validation_rows = _load_direct(validation_root, validation_manifest, layer)
    fresh_manifest = json.loads((fresh_root / "manifest.json").read_text(encoding="utf-8"))
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
    fit_queries, fit_positions = _load_query_observations(
        fit_query_root,
        fit_query_manifest,
        layer=layer,
    )
    validation_queries, validation_positions = _load_query_observations(
        validation_query_root,
        validation_query_manifest,
        layer=layer,
    )
    assert torch.equal(fit_positions, validation_positions)
    fit_queries = fit_queries[: args.fit_documents]
    fit_rows = fit_rows[: args.fit_documents]
    validation_queries = validation_queries[: args.validation_documents]
    validation_rows = validation_rows[: args.validation_documents]
    fresh_queries = fresh_queries[: args.fresh_documents]
    fresh_rows = fresh_rows[: args.fresh_documents]

    c1_manifest = json.loads((c1_root / "results.json").read_text(encoding="utf-8"))
    c1_record = c1_manifest["artifacts"][str(layer)]
    c1_tensors = load_file(str(c1_root / c1_record["file"]), device="cpu")
    value_encoder = c1_tensors["value_coordinate_encoders"]
    output_decoder = c1_tensors["head_output_decoders"]
    initial_maps = _initial_maps(initialization_root, layer)

    fitted_maps, history, best_epoch = _train_base(
        fit_queries,
        fit_rows,
        validation_queries,
        validation_rows,
        query_positions=fit_positions,
        value_encoder=value_encoder,
        initial_maps=initial_maps,
        cos=cos,
        sin=sin,
        page_size=args.page_size,
        pinned_prefix_pages=args.pinned_prefix_pages,
        epochs=args.epochs,
        documents_per_step=args.documents_per_step,
        factor_learning_rate=args.factor_learning_rate,
        bias_learning_rate=args.bias_learning_rate,
        gradient_clip=args.gradient_clip,
        patience=args.early_stopping_patience,
        seed=args.seed,
        device=device,
    )

    source_maps = {SOURCE_NAME: fitted_maps}
    fit_statistics = _build_statistics(
        fit_queries,
        fit_rows,
        query_positions=fit_positions,
        source_names=(SOURCE_NAME,),
        value_encoder=value_encoder,
        base_maps=source_maps,
        cos=cos,
        sin=sin,
        page_size=args.page_size,
        pinned_prefix_pages=args.pinned_prefix_pages,
        device=device,
    )[SOURCE_NAME]
    validation_statistics = _build_statistics(
        validation_queries,
        validation_rows,
        query_positions=validation_positions,
        source_names=(SOURCE_NAME,),
        value_encoder=value_encoder,
        base_maps=source_maps,
        cos=cos,
        sin=sin,
        page_size=args.page_size,
        pinned_prefix_pages=args.pinned_prefix_pages,
        device=device,
    )[SOURCE_NAME]
    residual_factors, residual_fit = _fit_residuals(
        fit_statistics,
        validation_statistics,
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
        base_maps=fitted_maps,
        residual_factors=residual_factors,
        residual_ranks=residual_ranks,
        cos=cos,
        sin=sin,
        page_size=args.page_size,
        page_budget=page_budget,
        pinned_prefix_pages=args.pinned_prefix_pages,
        device=device,
    )
    reference = json.loads(
        (reference_root / f"layer_{layer}" / "result.json").read_text(
            encoding="utf-8"
        )
    )["layers"][0]["fresh"]

    factor_tensors = {
        "base_left": torch.stack([item.left for item in fitted_maps]),
        "base_right": torch.stack([item.right for item in fitted_maps]),
        "base_bias": torch.stack([item.bias for item in fitted_maps]),
    }
    for rank, (route_encoder, route_query) in residual_factors.items():
        factor_tensors[f"residual_r{rank}_encoder"] = route_encoder
        factor_tensors[f"residual_r{rank}_query"] = route_query
    artifact = f"layer_{layer:03d}.safetensors"
    temporary = output_root / f"{artifact}.tmp"
    save_file(factor_tensors, str(temporary), metadata={"format": FORMAT})
    os.replace(temporary, output_root / artifact)

    result = {
        "format": FORMAT,
        "status": "complete",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(sys.argv),
        "protocol": {
            "model": str(model_root),
            "fixed_payload": str(c1_root),
            "initialization": str(initialization_root),
            "fit_documents": int(fit_rows.shape[0]),
            "validation_documents": int(validation_rows.shape[0]),
            "fresh_documents": int(fresh_rows.shape[0]),
            "queries_per_document": int(fit_queries.shape[1]),
            "query_positions": fit_positions.tolist(),
            "base_parameterization": "C1-V80 -> rank16 pre-RoPE K -> exact RoPE",
            "base_objective": "exact-teacher non-sink Page32 Fisher",
            "page_size": args.page_size,
            "pinned_prefix_pages": args.pinned_prefix_pages,
            "physical_token_budget_per_kv_group": args.physical_token_budget,
            "epochs": args.epochs,
            "documents_per_step": args.documents_per_step,
            "factor_learning_rate": args.factor_learning_rate,
            "bias_learning_rate": args.bias_learning_rate,
            "gradient_clip": args.gradient_clip,
            "early_stopping_patience": args.early_stopping_patience,
            "residual_ranks": list(residual_ranks),
            "router_sweeps": args.router_sweeps,
        },
        "layers": [
            {
                "layer": layer,
                "artifact": artifact,
                "base_fit": {"best_epoch": best_epoch, "history": history},
                "residual_fit": residual_fit,
                "fresh": fresh,
                "reference": reference,
            }
        ],
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
    _write_json(output_root / "result.json", result)
    _write_text(output_root / "summary.md", _markdown(result))
    print(
        f"[Pre-RoPE Fisher Base] wrote {output_root / 'result.json'} and "
        f"{output_root / 'summary.md'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
