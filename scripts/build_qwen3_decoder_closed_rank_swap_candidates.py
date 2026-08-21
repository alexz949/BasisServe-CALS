#!/usr/bin/env python3
"""Build and routed-screen decoder-closed within-layer Qwen3 rank swaps."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import sys
import time
from typing import Any, Mapping, Sequence

import torch
from safetensors.torch import load_file

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basisserve.core.decoder_closed_rank_candidates import (
    DecoderClosedRankSwap,
    close_ragged_decoder_with_fixed_encoders,
    enumerate_decoder_closed_rank_swaps,
    marginal_topk_proposal_ids,
    recover_ragged_anchor_from_folded,
    tensor_sha256,
)
from basisserve.core.gqa_routed_ov_joint import (
    combine_quadratics,
    covariance_with_trace_damping,
    evaluate_quadratic,
    fold_ragged_routed_ov_factors,
    function_prior_covariance,
    head_products,
    quadratic_from_target,
)
from basisserve.core.joint_aa_gqa_o import resolve_head_to_kv_group
from scripts.build_qwen3_gqa_routed_ov_solver_ablation import (
    _SafetensorWeightReader,
    _dtype,
    _layout_from_config,
    _load_source_profile,
    _load_statistics,
    _module_name,
    _parse_layers,
    _source_payload,
    _value_metrics,
    _write_json,
)
from scripts.resolve_qwen3_adaptive_schedule_with_routed_ov import _read_schedule


FORMAT = "basisserve.dc_gkl_swap.candidate_shard.v1"
CANDIDATE_FORMAT = "basisserve.dc_gkl_swap.layer_candidate.v1"
METHOD = "decoder_closed_conditional_global_kl_rank_swap_refinement"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--source-rank-bank", required=True)
    parser.add_argument("--source-schedule", required=True)
    parser.add_argument("--decoder-closed-base", required=True)
    parser.add_argument("--routed-stats", required=True)
    parser.add_argument("--a3-metrics", required=True)
    parser.add_argument("--global-kl-profile", required=True)
    parser.add_argument("--baseline-nll-results", default=None)
    parser.add_argument(
        "--baseline-nll-endpoint",
        default="cg16_decoder",
        help=(
            "endpoint name inside --baseline-nll-results; later sequential "
            "iterations may omit the external NLL artifact entirely"
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--layers", default="all")
    parser.add_argument(
        "--proposal-mode",
        choices=("exhaustive", "marginal_topk"),
        default="exhaustive",
    )
    parser.add_argument("--receiver-top-k", type=int, default=4)
    parser.add_argument("--donor-top-k", type=int, default=4)
    parser.add_argument("--top-per-layer", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=0.75)
    parser.add_argument("--coupling-mode", choices=("full_layer",), default="full_layer")
    parser.add_argument("--covariance-damping", type=float, default=1e-7)
    parser.add_argument("--decoder-jitter", type=float, default=0.0)
    parser.add_argument("--anchor-recovery-tolerance", type=float, default=5e-3)
    parser.add_argument("--base-reproduction-tolerance", type=float, default=5e-3)
    parser.add_argument("--work-dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument(
        "--factor-dtype",
        choices=("float32", "float16", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument("--factor-device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_error(left: torch.Tensor, right: torch.Tensor) -> float:
    a = left.detach().double().cpu()
    b = right.detach().double().cpu()
    denominator = torch.linalg.vector_norm(b).clamp_min(torch.finfo(torch.float64).tiny)
    return float(torch.linalg.vector_norm(a - b) / denominator)


def _resolve_decoder_closed_base(
    root: Path,
    *,
    source_root: Path,
    schedule_path: Path,
    schedule_sha256: str,
    schedule_rank_sum: int,
) -> Path:
    candidates = []
    direct = root / "config.json"
    configs = [direct] if direct.is_file() else sorted(root.glob("**/decoder_only/config.json"))
    for config_path in configs:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        if payload.get("format") != "basisserve.a3_gqa_vo.rank_bank.v1":
            continue
        if not payload.get("full_rank_factor_overrides"):
            continue
        endpoint = str(payload.get("endpoint", ""))
        if endpoint == "decoder_only":
            if (
                Path(payload.get("anchor_rank_bank", "")).expanduser().resolve()
                != source_root
            ):
                continue
            if payload.get("schedule_sha256") != schedule_sha256:
                continue
        else:
            if (
                Path(payload.get("source_rank_bank", "")).expanduser().resolve()
                != source_root
            ):
                continue
            if int(payload.get("rank_sum", -1)) != schedule_rank_sum:
                continue
            declared_schedule = payload.get("schedule")
            if declared_schedule is not None and (
                Path(declared_schedule).expanduser().resolve() != schedule_path
            ):
                continue
            declared_sha = payload.get("schedule_sha256")
            if declared_sha is not None and declared_sha != schedule_sha256:
                continue
            if declared_schedule is None:
                colocated_schedule = config_path.parent / "schedule.json"
                if (
                    not colocated_schedule.is_file()
                    or colocated_schedule.resolve() != schedule_path
                ):
                    continue
        candidates.append(config_path.parent)
    if len(candidates) != 1:
        raise ValueError(
            f"expected exactly one matching decoder-closed base under {root}, "
            f"found {len(candidates)}"
        )
    return candidates[0]


def _profile_and_config(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    profile = json.loads((root / config["profile"]).read_text(encoding="utf-8"))
    return config, profile


def _load_rank_payloads(
    *,
    root: Path,
    profile: Mapping[str, Any],
    layer_index: int,
    ranks: Sequence[int],
    head_dim: int,
) -> dict[int, dict[str, Any]]:
    payloads = {}
    for rank in sorted({int(rank) for rank in ranks if int(rank) < head_dim}):
        payloads[rank] = _source_payload(
            root,
            profile,
            layer_index=layer_index,
            rank=rank,
        )
    return payloads


def _cost_table(profile_path: Path) -> dict[tuple[int, int], dict[int, float]]:
    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    result: dict[tuple[int, int], dict[int, float]] = {}
    for entry in payload["entries"]:
        result[(int(entry["layer_index"]), int(entry["group_index"]))] = {
            int(rank): float(metric["mean_kl"])
            for rank, metric in entry["ranks"].items()
        }
    return result


def _baseline_nll_provenance(
    path: Path,
    *,
    base_root: Path,
    schedule_path: Path,
    endpoint_name: str,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != "basisserve.gqa_routed_ov.matched_endpoint_nll.v1":
        raise ValueError(f"unsupported matched baseline NLL artifact: {path}")
    matches = [
        item for item in payload["results"] if item.get("name") == endpoint_name
    ]
    if len(matches) != 1:
        raise ValueError(
            f"matched NLL artifact has no unique {endpoint_name!r} endpoint"
        )
    endpoint = matches[0]
    if Path(endpoint["rank_bank"]).expanduser().resolve() != base_root:
        raise ValueError(
            f"matched {endpoint_name!r} NLL uses a different rank bank"
        )
    if Path(endpoint["schedule"]).expanduser().resolve() != schedule_path:
        raise ValueError(
            f"matched {endpoint_name!r} NLL uses a different schedule"
        )
    return {
        "artifact": str(path),
        "artifact_sha256": _sha256(path),
        "endpoint": endpoint_name,
        "selection_windows": int(payload["selection_windows"]),
        "sequence_length": int(payload["sequence_length"]),
        "mean_selection_nll": float(endpoint["mean_selection_nll"]),
    }


def _base_group_factors(
    *,
    base_root: Path,
    base_profile: Mapping[str, Any],
    layer_index: int,
    ranks: Sequence[int],
    num_groups: int,
    heads_per_group: int,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...], dict[int, dict[str, Any]]]:
    payloads = _load_rank_payloads(
        root=base_root,
        profile=base_profile,
        layer_index=layer_index,
        ranks=ranks,
        head_dim=max(ranks),
    )
    # A schedule-specific bank can contain a rank-128 override, so load every
    # selected rank from its profile rather than assuming native passthrough.
    for rank in sorted(set(map(int, ranks))):
        if rank in payloads:
            continue
        entry = base_profile["layers"][str(layer_index)]["ranks"].get(str(rank))
        if entry is None:
            raise KeyError(f"decoder base layer {layer_index} lacks rank {rank}")
        payloads[rank] = torch.load(
            base_root / entry["factor_path"],
            map_location="cpu",
            weights_only=True,
        )
    v_groups = []
    o_groups = []
    for group, rank in enumerate(map(int, ranks)):
        payload = payloads[rank]
        v_groups.append(
            payload["v_proj_compressed_weight"][group * rank : (group + 1) * rank]
        )
        first_head = group * heads_per_group
        o_groups.append(
            payload["o_decoder_weight"][
                :,
                first_head * rank : (first_head + heads_per_group) * rank,
            ]
        )
    if len(v_groups) != num_groups:
        raise RuntimeError("decoder-base group extraction failed")
    return tuple(v_groups), tuple(o_groups), payloads


def _recover(
    *,
    dense_v: torch.Tensor,
    dense_o: torch.Tensor,
    payloads: Mapping[int, Mapping[str, Any]],
    ranks: Sequence[int],
    layout: Any,
    work_dtype: torch.dtype,
    device: torch.device,
):
    return recover_ragged_anchor_from_folded(
        dense_v=dense_v,
        dense_o=dense_o,
        folded_payloads=payloads,
        group_ranks=ranks,
        num_heads=layout.num_attention_heads,
        num_groups=layout.num_key_value_heads,
        head_dim=layout.head_dim,
        hidden_size=layout.hidden_size,
        work_dtype=work_dtype,
        device=device,
    )


def _save_candidate(
    path: Path,
    *,
    swap: DecoderClosedRankSwap,
    module_name: str,
    folded: Any,
    o_bias: torch.Tensor | None,
    factor_dtype: torch.dtype,
    diagnostics: Mapping[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "format": CANDIDATE_FORMAT,
            "method": METHOD,
            "candidate_id": swap.candidate_id,
            "layer_index": swap.layer_index,
            "module_name": module_name,
            "ranks_before": list(swap.ranks_before),
            "ranks_after": list(swap.ranks_after),
            "receiver_group": swap.receiver_group,
            "donor_group": swap.donor_group,
            "v_group_weights": [
                value.detach().cpu().to(factor_dtype)
                for value in folded.v_group_weights
            ],
            "o_group_weights": [
                value.detach().cpu().to(factor_dtype)
                for value in folded.o_group_weights
            ],
            "v_group_biases": [None] * len(folded.v_group_weights),
            "o_decoder_bias": (
                None if o_bias is None else o_bias.detach().cpu().to(factor_dtype)
            ),
            "diagnostics": dict(diagnostics),
        },
        temporary,
    )
    os.replace(temporary, path)


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha must be in [0, 1]")
    if args.top_per_layer <= 0:
        raise ValueError("--top-per-layer must be positive")
    model_root = Path(args.model).expanduser().resolve()
    source_root = Path(args.source_rank_bank).expanduser().resolve()
    schedule_path = Path(args.source_schedule).expanduser().resolve()
    base_input = Path(args.decoder_closed_base).expanduser().resolve()
    stats_root = Path(args.routed_stats).expanduser().resolve()
    metrics_path = Path(args.a3_metrics).expanduser().resolve()
    profile_path = Path(args.global_kl_profile).expanduser().resolve()
    baseline_nll_path = (
        None
        if args.baseline_nll_results is None
        else Path(args.baseline_nll_results).expanduser().resolve()
    )
    output = Path(args.output_dir).expanduser().resolve()
    for required in (
        model_root / "config.json",
        source_root / "config.json",
        schedule_path,
        stats_root / "config.json",
        metrics_path,
        profile_path,
    ):
        if not required.exists():
            raise FileNotFoundError(required)
    if baseline_nll_path is not None and not baseline_nll_path.is_file():
        raise FileNotFoundError(baseline_nll_path)
    if (output / "BUILD_COMPLETE").is_file() and args.resume:
        print(f"[Resume] complete candidate shard: {output}", flush=True)
        return
    if output.exists():
        raise FileExistsError(f"refusing to overwrite candidate shard: {output}")

    model_config = json.loads((model_root / "config.json").read_text(encoding="utf-8"))
    layout = _layout_from_config(model_config, int(model_config.get("head_dim", 128)))
    num_layers = int(model_config["num_hidden_layers"])
    requested = _parse_layers(args.layers)
    selected_layers = [
        layer for layer in range(num_layers) if requested is None or layer in requested
    ]
    if not selected_layers:
        raise ValueError("--layers selected no model layers")

    source_config = json.loads((source_root / "config.json").read_text(encoding="utf-8"))
    candidate_ranks = tuple(sorted(map(int, source_config["candidate_ranks"])))
    nonfull = min(rank for rank in candidate_ranks if rank < layout.head_dim)
    _, source_profile = _load_source_profile(
        source_root,
        model_name=str(model_root),
        layout=layout,
        rank=nonfull,
    )
    schedule = _read_schedule(
        schedule_path,
        num_layers=num_layers,
        num_groups=layout.num_key_value_heads,
        candidate_ranks=set(candidate_ranks),
    )
    schedule_sha = _sha256(schedule_path)
    base_root = _resolve_decoder_closed_base(
        base_input,
        source_root=source_root,
        schedule_path=schedule_path,
        schedule_sha256=schedule_sha,
        schedule_rank_sum=sum(rank for layer in schedule for rank in layer),
    )
    base_config, base_profile = _profile_and_config(base_root)
    if not base_config.get("full_rank_factor_overrides"):
        raise ValueError("decoder-closed base must include full-rank factor overrides")
    baseline_nll = (
        None
        if baseline_nll_path is None
        else _baseline_nll_provenance(
            baseline_nll_path,
            base_root=base_root,
            schedule_path=schedule_path,
            endpoint_name=args.baseline_nll_endpoint,
        )
    )

    all_swaps = enumerate_decoder_closed_rank_swaps(
        schedule,
        allowed_ranks=candidate_ranks,
        rank_step=16,
    )
    swaps_by_layer: dict[int, list[DecoderClosedRankSwap]] = {
        layer: [] for layer in range(num_layers)
    }
    for swap in all_swaps:
        swaps_by_layer[swap.layer_index].append(swap)
    costs = _cost_table(profile_path)
    marginal_ids = marginal_topk_proposal_ids(
        schedule,
        costs,
        allowed_ranks=candidate_ranks,
        receiver_top_k=args.receiver_top_k,
        donor_top_k=args.donor_top_k,
    )
    if args.proposal_mode == "marginal_topk":
        swaps_by_layer = {
            layer: [swap for swap in swaps if swap.candidate_id in marginal_ids]
            for layer, swaps in swaps_by_layer.items()
        }

    work_dtype = _dtype(args.work_dtype)
    factor_dtype = _dtype(args.factor_dtype)
    device = torch.device(args.factor_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA factor device requested but unavailable")
    mapping = resolve_head_to_kv_group(layout).to(device)
    reader = _SafetensorWeightReader(model_root)
    metric_tensors = load_file(str(metrics_path), device="cpu")
    output.mkdir(parents=True)
    config = {
        "format": FORMAT,
        "status": "incomplete",
        "method": METHOD,
        "model": str(model_root),
        "source_rank_bank": str(source_root),
        "source_schedule": str(schedule_path),
        "source_schedule_sha256": schedule_sha,
        "decoder_closed_base": str(base_root),
        "decoder_closed_base_config_sha256": _sha256(base_root / "config.json"),
        "routed_stats": str(stats_root),
        "a3_metrics": str(metrics_path),
        "global_kl_profile": str(profile_path),
        "baseline_nll": baseline_nll,
        "candidate_ranks": list(candidate_ranks),
        "layers": selected_layers,
        "proposal_mode": args.proposal_mode,
        "exhaustive_candidate_count_global": len(all_swaps),
        "marginal_candidate_count_global": len(marginal_ids),
        "alpha": args.alpha,
        "coupling_mode": args.coupling_mode,
        "top_per_layer": args.top_per_layer,
        "command": shlex.join([sys.executable, *sys.argv]),
        "started_unix_time": time.time(),
    }
    _write_json(output / "config.json", config)

    layer_records = []
    retained_records = []
    heads_per_group = layout.query_heads_per_kv_group
    for layer_index in selected_layers:
        layer_started = time.monotonic()
        module_name = _module_name(layer_index)
        prefix = f"{module_name}."
        dense_v = reader.tensor(prefix + "v_proj.weight")
        dense_o_weight = reader.tensor(prefix + "o_proj.weight")
        dense_v_bias = reader.tensor(prefix + "v_proj.bias", required=False)
        o_bias = reader.tensor(prefix + "o_proj.bias", required=False)
        if dense_v is None or dense_o_weight is None:
            raise KeyError(f"missing dense V/O weights for layer {layer_index}")
        if dense_v_bias is not None:
            raise ValueError("decoder closure does not support dense V bias")
        dense_o = (
            dense_o_weight.to(device=device, dtype=work_dtype)
            .transpose(0, 1)
            .reshape(layout.num_attention_heads, layout.head_dim, layout.hidden_size)
        )
        dense_v_device = dense_v.to(device=device)
        base_ranks = schedule[layer_index]
        fit_covariance, fit_rows, fit_energy = _load_statistics(
            stats_root,
            split="fit",
            layer_index=layer_index,
            layout=layout,
        )
        validation_covariance, validation_rows, validation_energy = _load_statistics(
            stats_root,
            split="validation",
            layer_index=layer_index,
            layout=layout,
        )
        fit_covariance, fit_damping = covariance_with_trace_damping(
            fit_covariance.to(device=device, dtype=work_dtype),
            relative_damping=args.covariance_damping,
        )
        validation_covariance, validation_damping = covariance_with_trace_damping(
            validation_covariance.to(device=device, dtype=work_dtype),
            relative_damping=args.covariance_damping,
        )
        value_metrics = _value_metrics(
            metric_tensors,
            layer_index=layer_index,
            layout=layout,
        ).to(device=device, dtype=work_dtype)
        function_covariance = function_prior_covariance(
            value_metrics,
            head_to_kv_group=mapping,
        )
        routed_fit = quadratic_from_target(
            covariance=fit_covariance,
            target=dense_o,
            name="dc_gkl_fit_routed_full_layer",
            trace_normalize=True,
            precomputed_constant=(fit_energy if fit_damping == 0.0 else None),
        )
        routed_validation = quadratic_from_target(
            covariance=validation_covariance,
            target=dense_o,
            name="dc_gkl_validation_routed_full_layer",
            trace_normalize=False,
            precomputed_constant=(
                validation_energy if validation_damping == 0.0 else None
            ),
        )
        routed_validation_normalized = quadratic_from_target(
            covariance=validation_covariance,
            target=dense_o,
            name="dc_gkl_validation_routed_full_layer_trace_normalized",
            trace_normalize=True,
        )

        def close_rank_vector(
            ranks: Sequence[int],
        ) -> tuple[Any, Any, dict[str, Any]]:
            payloads = _load_rank_payloads(
                root=source_root,
                profile=source_profile,
                layer_index=layer_index,
                ranks=ranks,
                head_dim=layout.head_dim,
            )
            recovered = _recover(
                dense_v=dense_v_device,
                dense_o=dense_o,
                payloads=payloads,
                ranks=ranks,
                layout=layout,
                work_dtype=work_dtype,
                device=device,
            )
            if recovered.maximum_head_product_error > args.anchor_recovery_tolerance:
                raise RuntimeError(
                    f"layer {layer_index} candidate recovery error "
                    f"{recovered.maximum_head_product_error:.6e} exceeds tolerance"
                )
            anchor_product = head_products(
                recovered.A_unique,
                recovered.D_heads,
                mapping,
            )
            prior = quadratic_from_target(
                covariance=function_covariance,
                target=anchor_product,
                name="dc_gkl_candidate_source_product_prior",
                trace_normalize=True,
            )
            combined_fit = combine_quadratics(
                prior,
                routed_fit,
                right_weight=args.alpha,
                name=f"dc_gkl_alpha_{args.alpha:g}_full_layer",
            )
            combined_validation = combine_quadratics(
                prior,
                routed_validation_normalized,
                right_weight=args.alpha,
                name=f"dc_gkl_validation_alpha_{args.alpha:g}_full_layer",
            )
            closure = close_ragged_decoder_with_fixed_encoders(
                objective=combined_fit,
                initial_A=recovered.A_unique,
                initial_D=recovered.D_heads,
                head_to_kv_group=mapping,
                group_ranks=ranks,
                relative_jitter=args.decoder_jitter,
            )
            folded = fold_ragged_routed_ov_factors(
                dense_v_proj_weight=dense_v_device,
                A_unique=closure.A_unique,
                D_heads=closure.D_heads,
                head_to_kv_group=mapping,
                group_ranks=ranks,
                thin_qr=True,
                output_dtype=factor_dtype,
            )
            product = head_products(closure.A_unique, closure.D_heads, mapping)
            diagnostics = {
                "group_ranks": list(map(int, ranks)),
                "source_group_checksums": list(recovered.source_group_checksums),
                "source_group_encoder_sha256": [
                    tensor_sha256(recovered.A_unique[group, :, : int(rank)])
                    for group, rank in enumerate(ranks)
                ],
                "maximum_value_encoder_recovery_error": recovered.maximum_value_encoder_error,
                "maximum_head_product_recovery_error": recovered.maximum_head_product_error,
                "gauge_product_error": closure.gauge_product_error,
                "encoder_sha256_loaded": closure.encoder_sha256_loaded,
                "encoder_sha256_before_solve": closure.encoder_sha256_before_solve,
                "encoder_sha256_after_solve": closure.encoder_sha256_after_solve,
                "encoder_immutable": (
                    closure.encoder_sha256_before_solve
                    == closure.encoder_sha256_after_solve
                ),
                "decoder": asdict(closure.decoder_diagnostics),
                "function_prior_deviation": evaluate_quadratic(
                    prior, closure.A_unique, closure.D_heads, mapping
                ),
                "routed_fit_loss": evaluate_quadratic(
                    routed_fit, closure.A_unique, closure.D_heads, mapping
                ),
                "mixed_fit_loss": evaluate_quadratic(
                    combined_fit, closure.A_unique, closure.D_heads, mapping
                ),
                "routed_validation_loss": evaluate_quadratic(
                    routed_validation, closure.A_unique, closure.D_heads, mapping
                ),
                "mixed_validation_loss": evaluate_quadratic(
                    combined_validation, closure.A_unique, closure.D_heads, mapping
                ),
                "folding": {
                    "maximum_qr_product_error": folded.maximum_qr_product_error,
                    "maximum_dense_fold_error": folded.maximum_dense_fold_error,
                },
            }
            diagnostics["normalized_routed_validation_error"] = diagnostics[
                "routed_validation_loss"
            ] / max(float(routed_validation.constant), 1e-30)
            return closure, folded, {**diagnostics, "dense_head_product": product}

        base_closure, base_folded, base_diagnostics = close_rank_vector(base_ranks)
        base_v_groups, base_o_groups, base_payloads = _base_group_factors(
            base_root=base_root,
            base_profile=base_profile,
            layer_index=layer_index,
            ranks=base_ranks,
            num_groups=layout.num_key_value_heads,
            heads_per_group=heads_per_group,
        )
        reproduction_v = [
            _relative_error(base_folded.v_group_weights[group], base_v_groups[group])
            for group in range(layout.num_key_value_heads)
        ]
        reproduction_o = [
            _relative_error(base_folded.o_group_weights[group], base_o_groups[group])
            for group in range(layout.num_key_value_heads)
        ]
        max_reproduction = max((*reproduction_v, *reproduction_o), default=0.0)
        if max_reproduction > args.base_reproduction_tolerance:
            raise RuntimeError(
                f"layer {layer_index} decoder-base reproduction error "
                f"{max_reproduction:.6e} exceeds tolerance"
            )
        base_product = base_diagnostics.pop("dense_head_product")
        base_validation = float(base_diagnostics["normalized_routed_validation_error"])
        no_op = {
            "group_ranks": list(base_ranks),
            "maximum_relative_factor_error": max_reproduction,
            "maximum_v_relative_error": max(reproduction_v, default=0.0),
            "maximum_o_relative_error": max(reproduction_o, default=0.0),
            "v_group_bitwise_equal": [
                tensor_sha256(base_folded.v_group_weights[group])
                == tensor_sha256(base_v_groups[group])
                for group in range(layout.num_key_value_heads)
            ],
            "o_group_bitwise_equal": [
                tensor_sha256(base_folded.o_group_weights[group])
                == tensor_sha256(base_o_groups[group])
                for group in range(layout.num_key_value_heads)
            ],
            "base_payload_checksums": {
                str(rank): tensor_sha256(payload["o_decoder_weight"])
                for rank, payload in base_payloads.items()
            },
            "diagnostics": base_diagnostics,
        }

        candidate_records = []
        retained_payloads: dict[str, tuple[Any, dict[str, Any]]] = {}
        layer_swaps = swaps_by_layer[layer_index]
        for candidate_index, swap in enumerate(layer_swaps, start=1):
            closure, folded, diagnostics = close_rank_vector(swap.ranks_after)
            product = diagnostics.pop("dense_head_product")
            diagnostics["relative_product_change_vs_closed_base"] = _relative_error(
                product,
                base_product,
            )
            diagnostics["routed_validation_improvement_vs_closed_base"] = (
                base_validation - diagnostics["normalized_routed_validation_error"]
            )
            diagnostics["layer_rank_sum_before"] = sum(swap.ranks_before)
            diagnostics["layer_rank_sum_after"] = sum(swap.ranks_after)
            diagnostics["decoder_input_width"] = (
                heads_per_group * sum(swap.ranks_after)
            )
            diagnostics["expected_decoder_input_width"] = (
                heads_per_group * sum(swap.ranks_before)
            )
            diagnostics["marginal_topk_proposed"] = swap.candidate_id in marginal_ids
            unchanged_groups = [
                group
                for group, (before_rank, after_rank) in enumerate(
                    zip(swap.ranks_before, swap.ranks_after, strict=True)
                )
                if before_rank == after_rank
            ]
            diagnostics["unchanged_group_encoder_checksums_match_base"] = all(
                diagnostics["source_group_encoder_sha256"][group]
                == base_diagnostics["source_group_encoder_sha256"][group]
                and diagnostics["source_group_checksums"][group]["v_sha256"]
                == base_diagnostics["source_group_checksums"][group]["v_sha256"]
                for group in unchanged_groups
            )
            if not diagnostics["unchanged_group_encoder_checksums_match_base"]:
                raise RuntimeError("an unchanged candidate encoder differs from the base")
            if diagnostics["decoder_input_width"] != diagnostics[
                "expected_decoder_input_width"
            ]:
                raise RuntimeError("candidate decoder width changed after a rank swap")
            record = {
                **swap.to_dict(),
                "metrics": diagnostics,
            }
            candidate_records.append(record)
            if diagnostics["routed_validation_improvement_vs_closed_base"] > 0:
                retained_payloads[swap.candidate_id] = (folded, diagnostics)
                keep = sorted(
                    retained_payloads,
                    key=lambda candidate_id: (
                        retained_payloads[candidate_id][1][
                            "normalized_routed_validation_error"
                        ],
                        candidate_id,
                    ),
                )[: args.top_per_layer]
                for candidate_id in tuple(retained_payloads):
                    if candidate_id not in keep:
                        del retained_payloads[candidate_id]
            print(
                f"[Candidate] layer={layer_index} "
                f"index={candidate_index}/{len(layer_swaps)} "
                f"id={swap.candidate_id} "
                f"val={diagnostics['normalized_routed_validation_error']:.9f} "
                f"delta={diagnostics['routed_validation_improvement_vs_closed_base']:+.9f}",
                flush=True,
            )
            del closure

        ordered = sorted(
            candidate_records,
            key=lambda item: (
                item["metrics"]["normalized_routed_validation_error"],
                item["candidate_id"],
            ),
        )
        retained = [
            item
            for item in ordered
            if item["metrics"]["routed_validation_improvement_vs_closed_base"] > 0
        ][: args.top_per_layer]
        for item in retained:
            candidate_id = item["candidate_id"]
            folded, diagnostics = retained_payloads[candidate_id]
            swap = next(value for value in layer_swaps if value.candidate_id == candidate_id)
            relative = Path("candidates") / f"{candidate_id}.pt"
            _save_candidate(
                output / relative,
                swap=swap,
                module_name=module_name,
                folded=folded,
                o_bias=o_bias,
                factor_dtype=factor_dtype,
                diagnostics=diagnostics,
            )
            item["factor_path"] = str(relative)
            item["factor_sha256"] = _sha256(output / relative)
            retained_records.append(item)
        best = ordered[0] if ordered else None
        layer_record = {
            "layer_index": layer_index,
            "module_name": module_name,
            "base_ranks": list(base_ranks),
            "layer_rank_sum": sum(base_ranks),
            "feasible_exhaustive_candidates": len(
                [swap for swap in all_swaps if swap.layer_index == layer_index]
            ),
            "evaluated_candidates": len(layer_swaps),
            "retained_candidates": [item["candidate_id"] for item in retained],
            "marginal_topk_contains_best": (
                None if best is None else best["candidate_id"] in marginal_ids
            ),
            "best_candidate_id": None if best is None else best["candidate_id"],
            "no_op_reproduction": no_op,
            "fit_rows": fit_rows,
            "validation_rows": validation_rows,
            "fit_absolute_damping": fit_damping,
            "validation_absolute_damping": validation_damping,
            "candidates": candidate_records,
            "elapsed_seconds": time.monotonic() - layer_started,
        }
        layer_records.append(layer_record)
        _write_json(output / "layer_results.json", layer_records)
        _write_json(
            output / "retained_candidates.json",
            {"format": FORMAT, "candidates": retained_records},
        )
        print(
            f"[LayerDone] layer={layer_index} evaluated={len(layer_swaps)} "
            f"retained={len(retained)} elapsed={layer_record['elapsed_seconds']:.1f}s",
            flush=True,
        )
        del base_closure, base_folded, retained_payloads
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    config.update(
        {
            "status": "complete",
            "completed_unix_time": time.time(),
            "evaluated_candidate_count": sum(
                item["evaluated_candidates"] for item in layer_records
            ),
            "retained_candidate_count": len(retained_records),
            "rank_sum": sum(rank for layer in schedule for rank in layer),
        }
    )
    _write_json(output / "config.json", config)
    (output / "BUILD_COMPLETE").touch()
    print(
        f"[Done] {output} evaluated={config['evaluated_candidate_count']} "
        f"retained={config['retained_candidate_count']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
