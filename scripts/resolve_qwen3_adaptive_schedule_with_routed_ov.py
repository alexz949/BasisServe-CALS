#!/usr/bin/env python3
"""Coordinate a fixed ragged Qwen3 GQA V/O schedule with the routed objective.

This builder never changes a rank.  It recovers the actually installed source
anchor for every physical KV group, solves the complete ragged layer decoder,
optionally performs one fixed-budget encoder sweep, and writes layer shards for
later assembly into ordinary BasisServe rank banks.
"""

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
from typing import Any, Sequence

import torch
from safetensors.torch import load_file

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basisserve.core.decoder_closed_rank_candidates import (
    recover_ragged_anchor_from_folded,
)
from basisserve.core.gqa_routed_ov_joint import (
    combine_quadratics,
    covariance_with_trace_damping,
    evaluate_quadratic,
    fit_routed_ov_joint,
    fold_ragged_routed_ov_factors,
    function_prior_covariance,
    head_products,
    quadratic_from_target,
)
from basisserve.core.joint_aa_gqa_o import resolve_head_to_kv_group
from scripts.build_qwen3_gqa_routed_ov_solver_ablation import (
    LAYER_FORMAT,
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


FORMAT = "basisserve.gqa_routed_ov.schedule_specific_shard.v1"
METHOD = "schedule_specific_routed_factor_coordination"
ENDPOINTS = ("alpha0_control", "decoder_only", "routed_cg16_one_sweep")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--stats-dir", required=True)
    parser.add_argument("--a3-metrics", required=True)
    parser.add_argument("--anchor-rank-bank", required=True)
    parser.add_argument("--schedule", required=True)
    parser.add_argument("--anchor-name", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--layers", default="all")
    parser.add_argument(
        "--build-endpoints",
        default=",".join(ENDPOINTS),
    )
    parser.add_argument("--alpha", type=float, default=0.75)
    parser.add_argument("--coupling-mode", choices=("full_layer",), default="full_layer")
    parser.add_argument("--cg-fixed-iterations", type=int, default=16)
    parser.add_argument("--cg-tolerance", type=float, default=1e-8)
    parser.add_argument("--max-sweeps", type=int, default=1)
    parser.add_argument(
        "--freeze-full-rank-encoders",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--covariance-damping", type=float, default=1e-7)
    parser.add_argument("--encoder-damping", type=float, default=1e-8)
    parser.add_argument("--decoder-jitter", type=float, default=0.0)
    parser.add_argument("--maximum-backtracks", type=int, default=10)
    parser.add_argument("--anchor-recovery-tolerance", type=float, default=5e-3)
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


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(item) for item in values)
    return ordered[min(math.ceil(fraction * len(ordered)) - 1, len(ordered) - 1)]


def _read_schedule(
    path: Path,
    *,
    num_layers: int,
    num_groups: int,
    candidate_ranks: set[int],
) -> tuple[tuple[int, ...], ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != "basisserve.a3_gqa_vo.group_rank_schedule.v1":
        pass
    ranks = tuple(
        tuple(int(rank) for rank in layer)
        for layer in payload["selected_ranks"]
    )
    if len(ranks) != num_layers or any(len(layer) != num_groups for layer in ranks):
        pass
    if any(rank not in candidate_ranks for layer in ranks for rank in layer):
        pass
    rank_sum = sum(rank for layer in ranks for rank in layer)
    expected_sum = int(payload["budget"]["rank_sum"])
    if rank_sum != expected_sum:
        pass
    histogram = {
        str(rank): sum(item == rank for layer in ranks for item in layer)
        for rank in sorted(candidate_ranks)
    }
    recorded = {str(key): int(value) for key, value in payload["rank_histogram"].items()}
    if histogram != recorded:
        pass
    return ranks


def _recover_ragged_anchor(
    *,
    dense_v: torch.Tensor,
    dense_o: torch.Tensor,
    source_root: Path,
    source_profile: dict[str, Any],
    layer_index: int,
    group_ranks: Sequence[int],
    num_heads: int,
    num_groups: int,
    head_dim: int,
    hidden_size: int,
    work_dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any], dict[int, dict[str, Any]]]:
    source_payloads = {
        int(rank): _source_payload(
            source_root,
            source_profile,
            layer_index=layer_index,
            rank=int(rank),
        )
        for rank in sorted(set(map(int, group_ranks)))
        if int(rank) < head_dim
    }
    recovered = recover_ragged_anchor_from_folded(
        dense_v=dense_v,
        dense_o=dense_o,
        group_ranks=group_ranks,
        num_heads=num_heads,
        num_groups=num_groups,
        head_dim=head_dim,
        hidden_size=hidden_size,
        work_dtype=work_dtype,
        device=device,
        folded_payloads=source_payloads,
    )
    value_errors = list(recovered.value_encoder_errors)
    product_errors = list(recovered.head_product_errors)
    diagnostics = {
        "value_encoder_errors": value_errors,
        "head_product_errors": product_errors,
        "maximum_value_encoder_error": recovered.maximum_value_encoder_error,
        "maximum_head_product_error": recovered.maximum_head_product_error,
        "median_head_product_error": _percentile(product_errors, 0.5),
        "p95_head_product_error": _percentile(product_errors, 0.95),
    }
    return recovered.A_unique, recovered.D_heads, diagnostics, source_payloads


def _endpoint_specification(
    endpoint: str,
    *,
    prior: Any,
    combined: Any,
) -> tuple[Any, int, bool]:
    if endpoint == "alpha0_control":
        return prior, 1, True
    if endpoint == "decoder_only":
        return combined, 0, False
    if endpoint == "routed_cg16_one_sweep":
        return combined, 1, True
    pass


def _save_endpoint_layer(
    *,
    endpoint_root: Path,
    endpoint: str,
    module_name: str,
    layer_index: int,
    group_ranks: Sequence[int],
    folded: Any,
    source_root: Path,
    source_profile: dict[str, Any],
    dense_v: torch.Tensor,
    dense_o_weight: torch.Tensor,
    o_bias: torch.Tensor | None,
    factor_dtype: torch.dtype,
    diagnostics: dict[str, Any],
    method: str = METHOD,
) -> dict[str, str]:
    num_groups = len(group_ranks)
    heads_per_group = folded.o_group_weights[0].shape[1] // group_ranks[0]
    rank_paths: dict[str, str] = {}
    layer_root = endpoint_root / f"layer_{layer_index:04d}"
    layer_root.mkdir(parents=True, exist_ok=True)
    for rank in sorted(set(int(item) for item in group_ranks)):
        if rank == dense_v.shape[0] // num_groups:
            base_v = dense_v.detach().cpu().to(factor_dtype).clone()
            base_o = dense_o_weight.detach().cpu().to(factor_dtype).clone()
        else:
            source = _source_payload(
                source_root,
                source_profile,
                layer_index=layer_index,
                rank=rank,
            )
            base_v = source["v_proj_compressed_weight"].detach().cpu().to(factor_dtype).clone()
            base_o = source["o_decoder_weight"].detach().cpu().to(factor_dtype).clone()
        for group, selected_rank in enumerate(group_ranks):
            if int(selected_rank) != rank:
                continue
            base_v[group * rank : (group + 1) * rank] = folded.v_group_weights[group]
            first_head = group * heads_per_group
            o_slice = slice(first_head * rank, (first_head + heads_per_group) * rank)
            base_o[:, o_slice] = folded.o_group_weights[group]
        path = layer_root / f"rank_{rank:04d}.pt"
        temporary = path.with_suffix(".pt.tmp")
        torch.save(
            {
                "format": LAYER_FORMAT,
                "method": method,
                "endpoint": endpoint,
                "module_name": module_name,
                "layer_index": layer_index,
                "rank_per_kv_head": rank,
                "v_proj_compressed_weight": base_v,
                "v_proj_compressed_bias": None,
                "o_decoder_weight": base_o,
                "o_decoder_bias": (
                    None if o_bias is None else o_bias.detach().cpu().to(factor_dtype)
                ),
                "diagnostics": diagnostics,
            },
            temporary,
        )
        os.replace(temporary, path)
        rank_paths[str(rank)] = str(path.relative_to(endpoint_root))
    return rank_paths


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    endpoints = tuple(
        item.strip() for item in args.build_endpoints.split(",") if item.strip()
    )
    if not endpoints or any(item not in ENDPOINTS for item in endpoints):
        pass
    if args.max_sweeps != 1 or args.cg_fixed_iterations <= 0:
        pass
    if not 0.0 <= args.alpha <= 1.0:
        pass

    model_root = Path(args.model).expanduser().resolve()
    stats_root = Path(args.stats_dir).expanduser().resolve()
    metrics_path = Path(args.a3_metrics).expanduser().resolve()
    source_root = Path(args.anchor_rank_bank).expanduser().resolve()
    schedule_path = Path(args.schedule).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    for required in (
        model_root / "config.json",
        stats_root / "config.json",
        metrics_path,
        source_root / "config.json",
        schedule_path,
    ):
        if not required.exists():
            pass
    if (output / "BUILD_COMPLETE").exists() and args.resume:
        print(f"[Resume] already complete: {output}", flush=True)
        return
    if output.exists():
        pass

    model_config = json.loads((model_root / "config.json").read_text(encoding="utf-8"))
    layout = _layout_from_config(model_config, int(model_config.get("head_dim", 128)))
    num_layers = int(model_config["num_hidden_layers"])
    requested = _parse_layers(args.layers)
    selected_layers = [
        layer for layer in range(num_layers) if requested is None or layer in requested
    ]
    if not selected_layers or (
        requested is not None and not requested.issubset(set(range(num_layers)))
    ):
        pass

    source_config = json.loads((source_root / "config.json").read_text(encoding="utf-8"))
    candidate_ranks = {int(item) for item in source_config["candidate_ranks"]}
    nonfull_probe = min(rank for rank in candidate_ranks if rank < layout.head_dim)
    _, source_profile = _load_source_profile(
        source_root,
        model_name=str(model_root),
        layout=layout,
        rank=nonfull_probe,
    )
    schedule = _read_schedule(
        schedule_path,
        num_layers=num_layers,
        num_groups=layout.num_key_value_heads,
        candidate_ranks=candidate_ranks,
    )
    schedule_metadata = json.loads(schedule_path.read_text(encoding="utf-8"))
    source_profile_path = Path(schedule_metadata["source_profile"]).expanduser().resolve()
    if source_profile_path.is_file():
        profiled = json.loads(source_profile_path.read_text(encoding="utf-8"))
        profiled_bank = Path(profiled["rank_bank"]).expanduser().resolve()
        if profiled_bank != source_root:
            pass

    stats_config = json.loads((stats_root / "config.json").read_text(encoding="utf-8"))
    if stats_config.get("status") != "complete":
        pass
    if set(selected_layers) - {int(item) for item in stats_config["layers"]}:
        pass

    work_dtype = _dtype(args.work_dtype)
    factor_dtype = _dtype(args.factor_dtype)
    device = torch.device(args.factor_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        pass
    mapping = resolve_head_to_kv_group(layout).to(device)
    reader = _SafetensorWeightReader(model_root)
    metric_tensors = load_file(str(metrics_path), device="cpu")
    output.mkdir(parents=True)
    root_config = {
        "format": FORMAT,
        "status": "incomplete",
        "method": METHOD,
        "model": str(model_root),
        "stats_dir": str(stats_root),
        "a3_metrics": str(metrics_path),
        "anchor_name": args.anchor_name,
        "anchor_rank_bank": str(source_root),
        "schedule": str(schedule_path),
        "schedule_sha256": _sha256(schedule_path),
        "source_config_sha256": _sha256(source_root / "config.json"),
        "source_profile_sha256": _sha256(source_root / source_config["profile"]),
        "layers": selected_layers,
        "endpoints": list(endpoints),
        "alpha": args.alpha,
        "coupling_mode": args.coupling_mode,
        "cg_fixed_iterations": args.cg_fixed_iterations,
        "freeze_full_rank_encoders": args.freeze_full_rank_encoders,
        "command": shlex.join([sys.executable, *sys.argv]),
        "started_unix_time": time.time(),
    }
    _write_json(output / "config.json", root_config)
    layer_results: list[dict[str, Any]] = []

    for layer_index in selected_layers:
        started = time.monotonic()
        module_name = _module_name(layer_index)
        prefix = f"{module_name}."
        dense_v = reader.tensor(prefix + "v_proj.weight")
        dense_o_weight = reader.tensor(prefix + "o_proj.weight")
        dense_v_bias = reader.tensor(prefix + "v_proj.bias", required=False)
        o_bias = reader.tensor(prefix + "o_proj.bias", required=False)
        if dense_v_bias is not None:
            pass
        dense_v_device = dense_v.to(device=device)
        dense_o = dense_o_weight.to(device=device, dtype=work_dtype).transpose(0, 1).reshape(
            layout.num_attention_heads,
            layout.head_dim,
            layout.hidden_size,
        )
        group_ranks = schedule[layer_index]
        initial_A, initial_D, recovery, _ = _recover_ragged_anchor(
            dense_v=dense_v_device,
            dense_o=dense_o,
            source_root=source_root,
            source_profile=source_profile,
            layer_index=layer_index,
            group_ranks=group_ranks,
            num_heads=layout.num_attention_heads,
            num_groups=layout.num_key_value_heads,
            head_dim=layout.head_dim,
            hidden_size=layout.hidden_size,
            work_dtype=work_dtype,
            device=device,
        )
        if recovery["maximum_head_product_error"] > args.anchor_recovery_tolerance:
            pass
        anchor_product = head_products(initial_A, initial_D, mapping)
        fit_covariance, fit_rows, fit_energy = _load_statistics(
            stats_root, split="fit", layer_index=layer_index, layout=layout
        )
        validation_covariance, validation_rows, validation_energy = _load_statistics(
            stats_root, split="validation", layer_index=layer_index, layout=layout
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
            metric_tensors, layer_index=layer_index, layout=layout
        ).to(device=device, dtype=work_dtype)
        function_covariance = function_prior_covariance(
            value_metrics, head_to_kv_group=mapping
        )
        prior = quadratic_from_target(
            covariance=function_covariance,
            target=anchor_product,
            name=f"{args.anchor_name}_schedule_product_prior",
            trace_normalize=True,
        )
        routed = quadratic_from_target(
            covariance=fit_covariance,
            target=dense_o,
            name="fit_routed_full_layer",
            trace_normalize=True,
            precomputed_constant=(fit_energy if fit_damping == 0.0 else None),
        )
        combined = combine_quadratics(
            prior,
            routed,
            right_weight=args.alpha,
            name=f"{args.anchor_name}_alpha_{args.alpha:g}_full_layer",
        )
        validation = quadratic_from_target(
            covariance=validation_covariance,
            target=dense_o,
            name="validation_full_routed",
            trace_normalize=False,
            precomputed_constant=(
                validation_energy if validation_damping == 0.0 else None
            ),
        )
        raw = {
            "fit_total_loss": evaluate_quadratic(combined, initial_A, initial_D, mapping),
            "function_prior_loss": evaluate_quadratic(prior, initial_A, initial_D, mapping),
            "routed_loss": evaluate_quadratic(routed, initial_A, initial_D, mapping),
            "validation_loss": evaluate_quadratic(validation, initial_A, initial_D, mapping),
        }
        raw["normalized_validation_error"] = raw["validation_loss"] / max(
            float(validation.constant), 1e-30
        )
        encoder_groups = tuple(
            group
            for group, rank in enumerate(group_ranks)
            if not args.freeze_full_rank_encoders or rank < layout.head_dim
        )
        endpoint_results: dict[str, Any] = {}
        for endpoint in endpoints:
            objective, sweeps, final_redecoder = _endpoint_specification(
                endpoint, prior=prior, combined=combined
            )
            result = fit_routed_ov_joint(
                objective=objective,
                initial_A=initial_A,
                initial_D=initial_D,
                head_to_kv_group=mapping,
                group_ranks=group_ranks,
                coupling_mode=args.coupling_mode,
                maximum_sweeps=sweeps,
                minimum_sweeps=sweeps,
                relative_objective_tolerance=0.0,
                patience=1,
                decoder_relative_jitter=args.decoder_jitter,
                encoder_relative_damping=args.encoder_damping,
                cg_relative_tolerance=args.cg_tolerance,
                cg_max_iterations=args.cg_fixed_iterations,
                cg_fixed_iterations=True,
                maximum_backtracks=args.maximum_backtracks,
                encoder_group_indices=(encoder_groups if sweeps else ()),
                component_objectives={"function_prior": prior, "routed": routed},
                final_decoder_solve=final_redecoder,
                verify_encoder_step_objective=True,
                work_dtype=work_dtype,
                work_device=device,
            )
            final_A = result.A_unique.to(device=device, dtype=work_dtype)
            final_D = result.D_heads.to(device=device, dtype=work_dtype)
            endpoint_product = head_products(final_A, final_D, mapping)
            product_errors = []
            for head in range(layout.num_attention_heads):
                product_errors.append(
                    float(
                        torch.linalg.vector_norm(endpoint_product[head] - anchor_product[head])
                        / torch.linalg.vector_norm(anchor_product[head]).clamp_min(
                            torch.finfo(work_dtype).tiny
                        )
                    )
                )
            folded = fold_ragged_routed_ov_factors(
                dense_v_proj_weight=dense_v_device,
                A_unique=final_A,
                D_heads=final_D,
                head_to_kv_group=mapping,
                group_ranks=group_ranks,
                thin_qr=True,
                output_dtype=factor_dtype,
            )
            steps = [
                step
                for sweep_result in result.sweeps
                for step in sweep_result.encoder_steps
            ]
            diagnostics = {
                "endpoint": endpoint,
                "group_ranks": list(group_ranks),
                "fit_total_loss": evaluate_quadratic(combined, final_A, final_D, mapping),
                "endpoint_objective_loss": result.final_loss,
                "function_prior_loss": evaluate_quadratic(prior, final_A, final_D, mapping),
                "routed_loss": evaluate_quadratic(routed, final_A, final_D, mapping),
                "validation_loss": evaluate_quadratic(validation, final_A, final_D, mapping),
                "source_product_errors": product_errors,
                "maximum_source_product_error": max(product_errors),
                "median_source_product_error": _percentile(product_errors, 0.5),
                "p95_source_product_error": _percentile(product_errors, 0.95),
                "initial_decoder": asdict(result.initial_decoder),
                "final_decoder": (
                    None if result.final_decoder is None else asdict(result.final_decoder)
                ),
                "sweeps": [asdict(item) for item in result.sweeps],
                "attribution": asdict(result.attribution),
                "cg_groups": [
                    {
                        "group_index": step.group_index,
                        "rank": int(group_ranks[step.group_index]),
                        **asdict(step),
                    }
                    for step in steps
                ],
                "folding": {
                    "maximum_qr_product_error": folded.maximum_qr_product_error,
                    "maximum_dense_fold_error": folded.maximum_dense_fold_error,
                },
            }
            diagnostics["normalized_validation_error"] = diagnostics[
                "validation_loss"
            ] / max(float(validation.constant), 1e-30)
            if endpoint == "alpha0_control" and diagnostics[
                "maximum_source_product_error"
            ] > 1e-7:
                pass
            endpoint_root = output / endpoint
            paths = _save_endpoint_layer(
                endpoint_root=endpoint_root,
                endpoint=endpoint,
                module_name=module_name,
                layer_index=layer_index,
                group_ranks=group_ranks,
                folded=folded,
                source_root=source_root,
                source_profile=source_profile,
                dense_v=dense_v,
                dense_o_weight=dense_o_weight,
                o_bias=o_bias,
                factor_dtype=factor_dtype,
                diagnostics=diagnostics,
            )
            endpoint_results[endpoint] = {**diagnostics, "factor_paths": paths}
            print(
                f"[Endpoint] anchor={args.anchor_name} layer={layer_index} "
                f"name={endpoint} fit={diagnostics['fit_total_loss']:.9f} "
                f"validation={diagnostics['normalized_validation_error']:.9f}",
                flush=True,
            )
        layer_record = {
            "layer_index": layer_index,
            "module_name": module_name,
            "group_ranks": list(group_ranks),
            "layer_rank_sum": sum(group_ranks),
            "v_cache_width": sum(group_ranks),
            "compressed_o_width": layout.query_heads_per_kv_group * sum(group_ranks),
            "anchor_recovery": recovery,
            "raw_anchor": raw,
            "fit_rows": fit_rows,
            "validation_rows": validation_rows,
            "fit_absolute_damping": fit_damping,
            "validation_absolute_damping": validation_damping,
            "endpoints": endpoint_results,
            "elapsed_seconds": time.monotonic() - started,
        }
        layer_results.append(layer_record)
        _write_json(output / "layer_results.json", layer_results)
        print(
            f"[LayerDone] anchor={args.anchor_name} layer={layer_index} "
            f"ranks={list(group_ranks)} elapsed={layer_record['elapsed_seconds']:.1f}s",
            flush=True,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    root_config.update(
        {
            "status": "complete",
            "completed_unix_time": time.time(),
            "rank_sum": sum(rank for row in schedule for rank in row),
            "schedule_rank_histogram": schedule_metadata["rank_histogram"],
        }
    )
    _write_json(output / "config.json", root_config)
    (output / "BUILD_COMPLETE").touch()
    print(f"[Done] {output}", flush=True)


if __name__ == "__main__":
    main()
