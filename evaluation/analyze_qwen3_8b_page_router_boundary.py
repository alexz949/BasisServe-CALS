#!/usr/bin/env python3
"""Decompose Page-Fisher and Top-page behavior for two Qwen3-8B routers."""

from __future__ import annotations

import argparse
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


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basisserve.core.gqa_joint_routing_payload_s80_ablation import (  # noqa: E402
    fit_page_fisher_router,
)
from basisserve.core.page_routing_decomposition import (  # noqa: E402
    budget_recovery,
    fisher_cross_loss,
    fisher_partition,
    page_lse_terms,
    top_page_masks,
)
from evaluation.eval_qwen3_8b_v80_conditional_residual_router import (  # noqa: E402
    _apply_base,
    _load_direct,
    _post_rope_rows,
    _rotary_embeddings,
    _value_codes,
)
from scripts.build_qwen3_gqa_joint_routing_payload_s80 import (  # noqa: E402
    _load_compact_fisher_layer,
)


FORMAT = "basisserve.qwen3_8b.page_router_boundary_decomposition.v1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--fresh-direct-dir", required=True)
    parser.add_argument("--page-fisher-calibration-root", required=True)
    parser.add_argument("--kq-init", required=True)
    parser.add_argument("--c1-v80", required=True)
    parser.add_argument("--conditional-checkpoint", required=True)
    parser.add_argument("--layers", default="0,17,35")
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument("--pages", type=int, default=32)
    parser.add_argument("--cutoff-band-pages", type=int, default=16)
    parser.add_argument("--router-sweeps", type=int, default=10)
    parser.add_argument("--relative-damping", type=float, default=1e-5)
    parser.add_argument("--iterative-tolerance", type=float, default=1e-5)
    parser.add_argument("--iterative-max-iterations", type=int, default=100)
    parser.add_argument("--work-device", default="cuda")
    parser.add_argument("--fit-dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--torch-num-threads", type=int, default=4)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _parse_ints(specification: str) -> tuple[int, ...]:
    return tuple(sorted({int(item) for item in specification.split(",") if item}))


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _discover_page_fisher(
    calibration_root: Path,
    *,
    layer: int,
) -> tuple[Path, dict[str, Any]]:
    pattern = "qwen3_8b_s80_c4_q8_s32768_page_fisher_chunk_*/fit/manifest.json"
    matches = []
    for path in sorted(calibration_root.glob(pattern)):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if str(layer) in manifest["artifacts"]:
            matches.append((path.parent, manifest))
    assert len(matches) == 1
    return matches[0]


def _fit_k_only_router(
    *,
    calibration_root: Path,
    layer: int,
    key_bank: torch.Tensor,
    query_bank: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    sweeps: int,
    relative_damping: float,
    iterative_tolerance: float,
    iterative_max_iterations: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float | int]]:
    root, manifest = _discover_page_fisher(calibration_root, layer=layer)
    statistics = _load_compact_fisher_layer(
        root,
        manifest,
        layer,
        device=device,
        dtype=dtype,
    )
    rank = 32
    value_dim = int(statistics.value_dim)
    groups = int(statistics.head_to_kv_group.max()) + 1
    initial_encoder = torch.zeros(
        groups,
        statistics.joint_dim,
        rank,
        device=device,
        dtype=dtype,
    )
    initial_encoder[:, value_dim:] = key_bank[layer, :, :, :rank].to(
        device=device,
        dtype=dtype,
    )
    initial_query = query_bank[layer, :, :, :rank].to(device=device, dtype=dtype)
    if initial_query.shape[0] == groups:
        initial_query = initial_query.repeat_interleave(
            int(statistics.queries_by_head.shape[0]) // groups,
            dim=0,
        )
    fitted = fit_page_fisher_router(
        statistics,
        initial_routing_encoders=initial_encoder,
        initial_query_factors=initial_query,
        active_joint_rows=torch.arange(
            value_dim,
            statistics.joint_dim,
            device=device,
        ),
        sweeps=sweeps,
        relative_damping=relative_damping,
        relative_tolerance=iterative_tolerance,
        max_iterations=iterative_max_iterations,
    )
    diagnostics = {
        "sweeps": sweeps,
        "final_query_maximum_iterations": max(
            item.iterations for item in fitted.final_query_diagnostics
        ),
        "final_query_maximum_relative_residual": max(
            item.relative_residual for item in fitted.final_query_diagnostics
        ),
    }
    return (
        fitted.routing_encoders[:, value_dim:].float(),
        fitted.routing_query_factors.float(),
        diagnostics,
    )


def _score_k_only(
    queries: torch.Tensor,
    exact_key: torch.Tensor,
    *,
    encoder: torch.Tensor,
    query_factor: torch.Tensor,
) -> torch.Tensor:
    query_heads, head_dim = map(int, queries.shape)
    groups = int(exact_key.shape[1])
    heads_per_group = query_heads // groups
    scaling = head_dim**-0.5
    scores = torch.empty(
        query_heads,
        exact_key.shape[0],
        device=queries.device,
        dtype=queries.dtype,
    )
    for group in range(groups):
        first = group * heads_per_group
        stop = first + heads_per_group
        query_code = torch.einsum(
            "hd,hdr->hr",
            queries[first:stop],
            query_factor[first:stop],
        )
        token_code = exact_key[:, group] @ encoder[group]
        scores[first:stop] = scaling * (query_code @ token_code.mT)
    return scores


def _score_conditional(
    queries: torch.Tensor,
    exact_key: torch.Tensor,
    value_codes: torch.Tensor,
    *,
    base_left: torch.Tensor,
    base_right: torch.Tensor,
    base_bias: torch.Tensor,
    residual_encoder: torch.Tensor,
    residual_query: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    query_heads, head_dim = map(int, queries.shape)
    groups = int(exact_key.shape[1])
    heads_per_group = query_heads // groups
    scaling = head_dim**-0.5
    base_pre = _apply_base(value_codes, (base_left, base_right, base_bias))
    base_post = _post_rope_rows(base_pre, cos, sin)
    residual = exact_key - base_post
    scores = torch.empty(
        query_heads,
        exact_key.shape[0],
        device=queries.device,
        dtype=queries.dtype,
    )
    for group in range(groups):
        first = group * heads_per_group
        stop = first + heads_per_group
        base_score = scaling * (queries[first:stop] @ base_post[:, group].mT)
        query_code = torch.einsum(
            "hd,hdr->hr",
            queries[first:stop],
            residual_query[first:stop],
        )
        token_code = residual[:, group] @ residual_encoder[group]
        scores[first:stop] = base_score + scaling * (query_code @ token_code.mT)
    return scores


def _rank_masks(
    teacher_logits: torch.Tensor,
    *,
    pages: int,
    band: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    order = teacher_logits.argsort(dim=-1, descending=True)
    ranks = torch.empty_like(order)
    positions = torch.arange(
        teacher_logits.shape[-1],
        device=teacher_logits.device,
    ).expand_as(order)
    ranks.scatter_(-1, order, positions)
    inside = ranks < int(pages)
    cutoff = (ranks >= max(0, int(pages) - int(band))) & (
        ranks < int(pages) + int(band)
    )
    inner_boundary = (ranks >= max(0, int(pages) - int(band))) & inside
    outer_boundary = (ranks >= int(pages)) & (
        ranks < int(pages) + int(band)
    )
    return inside, cutoff, inner_boundary, outer_boundary


def _append_tensor(store: dict[str, list[torch.Tensor]], key: str, value: torch.Tensor) -> None:
    store.setdefault(key, []).append(value.detach().float().cpu().reshape(-1))


def _pearson(values: torch.Tensor, labels: torch.Tensor) -> float:
    centered_values = values - values.mean()
    centered_labels = labels - labels.mean()
    denominator = torch.sqrt(
        centered_values.square().sum() * centered_labels.square().sum()
    )
    return float(
        (centered_values * centered_labels).sum()
        / denominator.clamp_min(torch.finfo(values.dtype).tiny)
    )


def _physical_budget_metrics(
    *,
    terms,
    value_codes: torch.Tensor,
    output_decoder: torch.Tensor,
    exact_scores: torch.Tensor,
    pages: int,
    page_size: int,
    groups: int,
) -> dict[str, torch.Tensor | float]:
    query_heads = int(exact_scores.shape[0])
    heads_per_group = query_heads // groups
    probabilities = torch.softmax(exact_scores, dim=-1)
    selected_m = top_page_masks(terms.proxy_logits, pages=pages)
    selected_2m = top_page_masks(terms.proxy_logits, pages=2 * pages)
    teacher_m = top_page_masks(terms.teacher_logits, pages=pages)
    physical_m = selected_m.reshape(groups, heads_per_group, -1).any(dim=1)
    physical_2m = selected_2m.reshape(groups, heads_per_group, -1).any(dim=1)
    teacher_physical = teacher_m.reshape(groups, heads_per_group, -1).any(dim=1)
    added = physical_2m & ~physical_m
    false_negative = teacher_physical & ~physical_m
    recovered = false_negative & added
    expanded_added = added.repeat_interleave(heads_per_group, dim=0)
    expanded_m = physical_m.repeat_interleave(heads_per_group, dim=0)
    expanded_2m = physical_2m.repeat_interleave(heads_per_group, dim=0)
    added_mass = torch.sum(terms.teacher_mass * expanded_added, dim=-1)
    selected_mass_m = torch.sum(terms.teacher_mass * expanded_m, dim=-1)
    selected_mass_2m = torch.sum(terms.teacher_mass * expanded_2m, dim=-1)
    false_negative_count = false_negative.sum(dim=-1)
    recovery = recovered.sum(dim=-1) / false_negative_count.clamp_min(1)
    teacher_recall_m = (physical_m & teacher_physical).sum(dim=-1) / (
        teacher_physical.sum(dim=-1).clamp_min(1)
    )
    teacher_recall_2m = (physical_2m & teacher_physical).sum(dim=-1) / (
        teacher_physical.sum(dim=-1).clamp_min(1)
    )

    head_to_group = torch.arange(query_heads, device=exact_scores.device) // heads_per_group
    head_codes = value_codes.permute(1, 0, 2).index_select(0, head_to_group)
    dense_latent = torch.einsum("ht,htr->hr", probabilities, head_codes)
    dense_output = torch.einsum("hr,hro->o", dense_latent, output_decoder)
    result: dict[str, torch.Tensor | float] = {
        "physical_pages_m": physical_m.sum(dim=-1),
        "physical_pages_2m": physical_2m.sum(dim=-1),
        "physical_added_pages": added.sum(dim=-1),
        "physical_selected_mass_m": selected_mass_m,
        "physical_selected_mass_2m": selected_mass_2m,
        "physical_added_mass": added_mass,
        "physical_teacher_page_recall_m": teacher_recall_m,
        "physical_teacher_page_recall_2m": teacher_recall_2m,
        "physical_false_negative_recovery": recovery,
        "dense_output_energy": float(dense_output.square().sum()),
    }
    for label, mask in (("m", physical_m), ("2m", physical_2m)):
        token_mask = mask.repeat_interleave(page_size, dim=-1)[
            :, : exact_scores.shape[-1]
        ]
        query_mask = token_mask.repeat_interleave(heads_per_group, dim=0)
        sparse_probability = torch.softmax(
            exact_scores.masked_fill(~query_mask, -torch.inf),
            dim=-1,
        )
        sparse_latent = torch.einsum(
            "ht,htr->hr",
            sparse_probability,
            head_codes,
        )
        sparse_output = torch.einsum("hr,hro->o", sparse_latent, output_decoder)
        result[f"output_squared_error_{label}"] = float(
            (sparse_output - dense_output).square().sum()
        )
    return result


def _record_arm(
    store: dict[str, Any],
    *,
    exact_scores: torch.Tensor,
    proxy_scores: torch.Tensor,
    value_codes: torch.Tensor,
    output_decoder: torch.Tensor,
    pages: int,
    page_size: int,
    cutoff_band: int,
    groups: int,
) -> None:
    terms = page_lse_terms(exact_scores, proxy_scores, page_size=page_size)
    inside, cutoff, inner_boundary, outer_boundary = _rank_masks(
        terms.teacher_logits,
        pages=pages,
        band=cutoff_band,
    )
    proxy_inside = top_page_masks(terms.proxy_logits, pages=pages)
    changed = inside ^ proxy_inside
    tensors = store.setdefault("tensors", {})
    for key, value in (
        ("remainder", terms.remainder),
        ("true_error", terms.true_error),
        ("linearized_error", terms.linearized_error),
        ("teacher_mass", terms.teacher_mass),
        ("membership_changed", changed.float()),
        ("cutoff_mask", cutoff.float()),
    ):
        _append_tensor(tensors, key, value)
    _append_tensor(tensors, "cutoff_remainder", terms.remainder[cutoff])
    _append_tensor(tensors, "cutoff_changed", changed[cutoff].float())

    fisher = store.setdefault("fisher", {"linearized": {}, "true_lse": {}})
    for label, error in (
        ("linearized", terms.linearized_error),
        ("true_lse", terms.true_error),
    ):
        partition = fisher_partition(error, terms.teacher_mass, inside)
        boundary = fisher_cross_loss(
            error,
            terms.teacher_mass,
            inner_boundary,
            outer_boundary,
        )
        for name, value in (
            ("total", partition.total),
            ("inside_inside", partition.inside_inside),
            ("outside_outside", partition.outside_outside),
            ("cross", partition.cross),
            ("cutoff_cross", boundary),
        ):
            _append_tensor(fisher[label], name, value)

    recovery = budget_recovery(
        terms.teacher_logits,
        terms.proxy_logits,
        terms.teacher_mass,
        pages=pages,
    )
    budget = store.setdefault("budget", {})
    for name in (
        "selected_mass_m",
        "selected_mass_2m",
        "added_mass",
        "false_negative_count",
        "false_negative_recovery",
    ):
        _append_tensor(budget, name, recovery[name])
    physical = _physical_budget_metrics(
        terms=terms,
        value_codes=value_codes,
        output_decoder=output_decoder,
        exact_scores=exact_scores,
        pages=pages,
        page_size=page_size,
        groups=groups,
    )
    for name in (
        "physical_pages_m",
        "physical_pages_2m",
        "physical_added_pages",
        "physical_selected_mass_m",
        "physical_selected_mass_2m",
        "physical_added_mass",
        "physical_teacher_page_recall_m",
        "physical_teacher_page_recall_2m",
        "physical_false_negative_recovery",
    ):
        _append_tensor(budget, name, physical[name])
    store["dense_output_energy"] = store.get("dense_output_energy", 0.0) + float(
        physical["dense_output_energy"]
    )
    store["output_squared_error_m"] = store.get("output_squared_error_m", 0.0) + float(
        physical["output_squared_error_m"]
    )
    store["output_squared_error_2m"] = store.get("output_squared_error_2m", 0.0) + float(
        physical["output_squared_error_2m"]
    )


def _summarize_arm(store: dict[str, Any]) -> dict[str, Any]:
    tensors = {key: torch.cat(value) for key, value in store["tensors"].items()}
    remainder = tensors["remainder"]
    true_error = tensors["true_error"]
    mass = tensors["teacher_mass"]
    changed = tensors["membership_changed"]
    cutoff_remainder = tensors["cutoff_remainder"]
    cutoff_changed = tensors["cutoff_changed"]
    changed_bool = changed.bool()
    stable_bool = ~changed_bool
    cutoff_changed_bool = cutoff_changed.bool()
    cutoff_stable_bool = ~cutoff_changed_bool
    linearization = {
        "remainder_l2": float(torch.linalg.vector_norm(remainder)),
        "remainder_rms": float(torch.sqrt(remainder.square().mean())),
        "teacher_mass_weighted_remainder_rms": float(
            torch.sqrt(torch.sum(mass * remainder.square()) / mass.sum())
        ),
        "remainder_to_true_error_l2_ratio": float(
            torch.linalg.vector_norm(remainder)
            / torch.linalg.vector_norm(true_error).clamp_min(
                torch.finfo(true_error.dtype).tiny
            )
        ),
        "cutoff_remainder_rms": float(
            torch.sqrt(cutoff_remainder.square().mean())
        ),
        "membership_change_rate": float(changed.mean()),
        "abs_remainder_membership_correlation": _pearson(
            remainder.abs(), changed
        ),
        "cutoff_abs_remainder_membership_correlation": _pearson(
            cutoff_remainder.abs(), cutoff_changed
        ),
        "changed_page_abs_remainder_mean": float(
            remainder[changed_bool].abs().mean()
        ),
        "stable_page_abs_remainder_mean": float(
            remainder[stable_bool].abs().mean()
        ),
        "cutoff_changed_page_abs_remainder_mean": float(
            cutoff_remainder[cutoff_changed_bool].abs().mean()
        ),
        "cutoff_stable_page_abs_remainder_mean": float(
            cutoff_remainder[cutoff_stable_bool].abs().mean()
        ),
    }
    fisher_summary: dict[str, Any] = {}
    for label, values in store["fisher"].items():
        sums = {key: float(torch.cat(rows).sum()) for key, rows in values.items()}
        total = sums["total"]
        fisher_summary[label] = {
            **sums,
            "inside_inside_fraction": sums["inside_inside"] / total,
            "outside_outside_fraction": sums["outside_outside"] / total,
            "cross_fraction": sums["cross"] / total,
            "cutoff_cross_fraction": sums["cutoff_cross"] / total,
        }
    budget = {
        key: float(torch.cat(rows).mean())
        for key, rows in store["budget"].items()
    }
    output_m = store["output_squared_error_m"] / store["dense_output_energy"]
    output_2m = store["output_squared_error_2m"] / store["dense_output_energy"]
    budget.update(
        {
            "exact_refined_output_relative_mse_m": output_m,
            "exact_refined_output_relative_mse_2m": output_2m,
            "output_error_relative_reduction_m_to_2m": (
                output_m - output_2m
            )
            / output_m,
        }
    )
    return {
        "page_lse_linearization": linearization,
        "page_fisher_partition": fisher_summary,
        "budget_expansion": budget,
    }


def _mean_tree(records: list[dict[str, Any]]) -> dict[str, Any]:
    first = records[0]
    result: dict[str, Any] = {}
    for key, value in first.items():
        if isinstance(value, dict):
            result[key] = _mean_tree([record[key] for record in records])
        elif isinstance(value, (int, float)):
            result[key] = sum(float(record[key]) for record in records) / len(records)
    return result


@torch.inference_mode()
def main() -> None:
    args = _parse_args()
    torch.set_num_threads(args.torch_num_threads)
    torch.backends.cuda.matmul.allow_tf32 = True
    started = time.monotonic()
    device = torch.device(args.work_device)
    fit_dtype = torch.float64 if args.fit_dtype == "float64" else torch.float32
    model_root = Path(args.model).expanduser().resolve()
    fresh_root = Path(args.fresh_direct_dir).expanduser().resolve()
    calibration_root = Path(args.page_fisher_calibration_root).expanduser().resolve()
    c1_root = Path(args.c1_v80).expanduser().resolve()
    conditional_root = Path(args.conditional_checkpoint).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    layers = _parse_ints(args.layers)
    fresh_manifest = json.loads((fresh_root / "manifest.json").read_text(encoding="utf-8"))
    c1_manifest = json.loads((c1_root / "results.json").read_text(encoding="utf-8"))
    kq_manifest = json.loads(
        (Path(args.kq_init).expanduser().resolve() / "result.json").read_text(
            encoding="utf-8"
        )
    )
    kq_root = Path(args.kq_init).expanduser().resolve()
    kq = load_file(
        str(kq_root / kq_manifest["artifacts"]["factors"]["file"]),
        device="cpu",
    )
    key_bank = kq["kq_svd_key_projector"]
    query_bank = kq["kq_svd_query_projector"]
    cos, sin = _rotary_embeddings(model_root, sequence=32768, device=device)
    layer_records = []

    for ordinal, layer in enumerate(layers, start=1):
        print(
            f"[page boundary] layer={layer} ({ordinal}/{len(layers)}) refit K32",
            flush=True,
        )
        k_encoder, k_query, fit_diagnostics = _fit_k_only_router(
            calibration_root=calibration_root,
            layer=layer,
            key_bank=key_bank,
            query_bank=query_bank,
            device=device,
            dtype=fit_dtype,
            sweeps=args.router_sweeps,
            relative_damping=args.relative_damping,
            iterative_tolerance=args.iterative_tolerance,
            iterative_max_iterations=args.iterative_max_iterations,
        )
        k_encoder = k_encoder.to(device=device, dtype=torch.float32)
        k_query = k_query.to(device=device, dtype=torch.float32)
        conditional = load_file(
            str(conditional_root / f"layer_{layer:03d}.safetensors"),
            device="cpu",
        )
        conditional_tensors = {
            name: conditional[name].to(device=device, dtype=torch.float32)
            for name in (
                "base_left_b32",
                "base_right_b32",
                "base_bias_b32",
                "residual_encoder_b32_r8",
                "residual_query_b32_r8",
            )
        }
        c1_artifact = c1_manifest["artifacts"][str(layer)]
        c1 = load_file(str(c1_root / c1_artifact["file"]), device="cpu")
        value_encoder = c1["value_coordinate_encoders"].to(
            device=device,
            dtype=torch.float32,
        )
        output_decoder = c1["head_output_decoders"].to(
            device=device,
            dtype=torch.float32,
        )
        queries_raw, rows_raw = _load_direct(fresh_root, fresh_manifest, layer)
        stores: dict[str, dict[str, Any]] = {"k_only_r32": {}, "base32_r8": {}}
        documents, _, head_dim = map(int, queries_raw.shape)
        groups = int(rows_raw.shape[2])
        scaling = head_dim**-0.5
        for document in range(documents):
            print(
                f"[page boundary] layer={layer} document={document + 1}/{documents}",
                flush=True,
            )
            rows = rows_raw[document].to(device=device, dtype=torch.float32)
            queries = queries_raw[document].to(device=device, dtype=torch.float32)
            dense_value = rows[..., :head_dim]
            exact_key = rows[..., head_dim:]
            value_codes = _value_codes(dense_value, value_encoder)
            exact_scores = torch.empty(
                queries.shape[0],
                exact_key.shape[0],
                device=device,
                dtype=torch.float32,
            )
            heads_per_group = int(queries.shape[0]) // groups
            for group in range(groups):
                first = group * heads_per_group
                stop = first + heads_per_group
                exact_scores[first:stop] = scaling * (
                    queries[first:stop] @ exact_key[:, group].mT
                )
            proxies = {
                "k_only_r32": _score_k_only(
                    queries,
                    exact_key,
                    encoder=k_encoder,
                    query_factor=k_query,
                ),
                "base32_r8": _score_conditional(
                    queries,
                    exact_key,
                    value_codes,
                    base_left=conditional_tensors["base_left_b32"],
                    base_right=conditional_tensors["base_right_b32"],
                    base_bias=conditional_tensors["base_bias_b32"],
                    residual_encoder=conditional_tensors["residual_encoder_b32_r8"],
                    residual_query=conditional_tensors["residual_query_b32_r8"],
                    cos=cos,
                    sin=sin,
                ),
            }
            for arm, proxy in proxies.items():
                _record_arm(
                    stores[arm],
                    exact_scores=exact_scores,
                    proxy_scores=proxy,
                    value_codes=value_codes,
                    output_decoder=output_decoder,
                    pages=args.pages,
                    page_size=args.page_size,
                    cutoff_band=args.cutoff_band_pages,
                    groups=groups,
                )
        arms = {name: _summarize_arm(store) for name, store in stores.items()}
        layer_records.append(
            {
                "layer": layer,
                "k_only_refit": fit_diagnostics,
                "arms": arms,
            }
        )
        _atomic_json(
            output_path,
            {
                "format": FORMAT,
                "status": "running",
                "command": shlex.join(sys.argv),
                "layers": layer_records,
                "aggregate": {
                    arm: _mean_tree([record["arms"][arm] for record in layer_records])
                    for arm in stores
                },
                "elapsed_seconds": time.monotonic() - started,
            },
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    result = {
        "format": FORMAT,
        "status": "complete",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(sys.argv),
        "protocol": {
            "layers": list(layers),
            "documents": 4,
            "sequence_length": 32768,
            "query_policy": "last_token_full_prefix",
            "page_size": args.page_size,
            "pages_m": args.pages,
            "pages_2m": 2 * args.pages,
            "cutoff_band": (
                f"teacher ranks M-{args.cutoff_band_pages} through "
                f"M+{args.cutoff_band_pages}-1"
            ),
            "k_only_r32": "q8 Page-Fisher refit with the original KQ-SVD initialization",
            "base32_r8": "fixed C1-V80 affine pre-RoPE base32 plus post-RoPE residual8",
            "fisher_partitions": ["linearized", "true_lse"],
            "physical_accounting": "GQA union after per-query-head page selection",
        },
        "layers": layer_records,
        "aggregate": {
            arm: _mean_tree([record["arms"][arm] for record in layer_records])
            for arm in layer_records[0]["arms"]
        },
        "elapsed_seconds": time.monotonic() - started,
        "environment": {
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "python": sys.version,
            "torch": torch.__version__,
            "fit_dtype": args.fit_dtype,
            "cuda_devices": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ],
        },
    }
    _atomic_json(output_path, result)
    print(f"[page boundary] wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
