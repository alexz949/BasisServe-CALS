#!/usr/bin/env python3
"""Evaluate Store80 sharing against branch drops and equal-storage hard splits."""

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

from safetensors.torch import load_file
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basisserve.checkpoint.gqa_joint_routing_payload_s80_qwen3 import (  # noqa: E402
    load_s80_factor_bank,
)
from basisserve.core.gqa_joint_routing_payload_s80 import (  # noqa: E402
    S80Layout,
    dense_o_weight_to_head_blocks,
)
from basisserve.core.gqa_joint_routing_payload_s80_ablation import (  # noqa: E402
    fit_page_fisher_router,
    refit_page_fisher_query_factors,
)
from basisserve.core.gqa_joint_routing_payload_s80_fisher import (  # noqa: E402
    compact_softmax_fisher_loss,
    prepare_softmax_fisher_routing,
    softmax_fisher_routing_diagnostics,
)
from basisserve.core.gqa_routed_ov_joint import (  # noqa: E402
    RoutedOVQuadratic,
    evaluate_quadratic,
    fit_routed_ov_joint,
    quadratic_from_target,
    solve_free_decoder,
)
from scripts.build_qwen3_gqa_joint_routing_payload_s80 import (  # noqa: E402
    _SafetensorWeightReader,
    _load_c1_layer,
    _load_compact_fisher_layer,
    _load_direct_layer,
    _load_json,
    _load_statistics,
    _parse_layers,
)


FORMAT = "basisserve.qwen3_8b.s80_page_fisher_synergy.v1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--stats-dir", required=True)
    parser.add_argument("--page-fisher-fit-dir", required=True)
    parser.add_argument("--page-fisher-validation-dir", required=True)
    parser.add_argument("--joint-checkpoint", required=True)
    parser.add_argument("--c1-v48", required=True)
    parser.add_argument("--c1-v64", required=True)
    parser.add_argument("--c1-v80", required=True)
    parser.add_argument("--kq-r32-init", required=True)
    parser.add_argument("--direct-validation-dir")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--layers", default="all")
    parser.add_argument("--payload-sweeps", type=int, default=20)
    parser.add_argument("--router-sweeps", type=int, default=10)
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument("--exact-token-budget", type=int, default=2048)
    parser.add_argument("--relative-damping", type=float, default=1e-5)
    parser.add_argument("--iterative-tolerance", type=float, default=1e-5)
    parser.add_argument("--iterative-max-iterations", type=int, default=100)
    parser.add_argument("--work-dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--work-device", default="cuda")
    parser.add_argument("--torch-num-threads", type=int, default=4)
    return parser.parse_args()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _payload_objective(
    payload_statistics,
    *,
    dense_o_weight: torch.Tensor,
    layout: S80Layout,
    device: torch.device,
    dtype: torch.dtype,
) -> RoutedOVQuadratic:
    target = dense_o_weight_to_head_blocks(dense_o_weight, layout).to(
        device=device,
        dtype=dtype,
    )
    covariance = payload_statistics.covariance_blocks.to(device=device, dtype=dtype)
    value_covariance = covariance[..., : layout.value_dim, : layout.value_dim]
    return quadratic_from_target(
        covariance=value_covariance,
        target=target,
        name="s80_hard_split_value_payload",
        trace_normalize=False,
    )


def _full_payload_objective(
    payload_statistics,
    *,
    dense_o_weight: torch.Tensor,
    layout: S80Layout,
    device: torch.device,
    dtype: torch.dtype,
) -> RoutedOVQuadratic:
    dense_blocks = dense_o_weight_to_head_blocks(dense_o_weight, layout).to(
        device=device,
        dtype=dtype,
    )
    zeros = dense_blocks.new_zeros(
        layout.num_attention_heads,
        layout.key_dim,
        layout.hidden_size,
    )
    return quadratic_from_target(
        covariance=payload_statistics.covariance_blocks.to(device=device, dtype=dtype),
        target=torch.cat((dense_blocks, zeros), dim=1),
        name="s80_joint_payload",
        trace_normalize=False,
    )


def _payload_relative_mse(
    objective: RoutedOVQuadratic,
    *,
    encoders: torch.Tensor,
    decoders: torch.Tensor,
    mapping: torch.Tensor,
    energy: float,
) -> float:
    device = objective.covariance.device
    dtype = objective.covariance.dtype
    return evaluate_quadratic(
        objective,
        encoders.to(device=device, dtype=dtype),
        decoders.to(device=device, dtype=dtype),
        mapping.to(device=device, dtype=torch.long),
    ) / float(energy)


def _embed_value_encoder(
    value_encoder: torch.Tensor,
    *,
    layout: S80Layout,
) -> torch.Tensor:
    result = value_encoder.new_zeros(
        layout.num_key_value_heads,
        layout.joint_dim,
        value_encoder.shape[-1],
    )
    result[:, : layout.value_dim] = value_encoder
    return result


def _router_metrics(statistics, encoders: torch.Tensor, queries: torch.Tensor) -> dict[str, float]:
    device = statistics.queries_by_head.device
    dtype = statistics.queries_by_head.dtype
    loss = compact_softmax_fisher_loss(
        statistics,
        routing_payload_encoders=encoders.to(device=device, dtype=dtype),
        routing_query_factors=queries.to(device=device, dtype=dtype),
    )
    return {
        "loss": loss,
        "nmse": loss / statistics.teacher_fisher_energy,
        "teacher_energy": statistics.teacher_fisher_energy,
    }


def _cg_summary(records) -> dict[str, float | int]:
    rows = tuple(records)
    return {
        "systems": len(rows),
        "converged": sum(item.converged for item in rows),
        "maximum_iterations": max((item.iterations for item in rows), default=0),
        "maximum_relative_residual": max(
            (item.relative_residual for item in rows),
            default=0.0,
        ),
    }


def _selection_metrics(
    direct,
    *,
    layout: S80Layout,
    selection_encoder: torch.Tensor,
    query_factors: torch.Tensor,
    payload_encoder: torch.Tensor,
    payload_decoder: torch.Tensor,
    page_size: int,
    exact_token_budget: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, float | int]:
    routing = prepare_softmax_fisher_routing(
        direct,
        head_to_kv_group=layout.head_to_kv_group(),
        value_dim=layout.value_dim,
        key_dim=layout.key_dim,
        device=device,
        dtype=dtype,
    )
    empty = payload_encoder[..., :0]
    return softmax_fisher_routing_diagnostics(
        routing,
        selection_routing_encoders=selection_encoder.to(device=device, dtype=dtype),
        payload_routing_encoders=empty.to(device=device, dtype=dtype),
        payload_only_encoders=payload_encoder.to(device=device, dtype=dtype),
        routing_query_factors=query_factors.to(device=device, dtype=dtype),
        payload_decoders=payload_decoder.to(device=device, dtype=dtype),
        page_size=page_size,
        exact_token_budget=exact_token_budget,
    )


def _joint_selection_metrics(
    direct,
    *,
    layout: S80Layout,
    selection_encoder: torch.Tensor,
    query_factors: torch.Tensor,
    joint_encoder: torch.Tensor,
    payload_decoder: torch.Tensor,
    page_size: int,
    exact_token_budget: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, float | int]:
    routing = prepare_softmax_fisher_routing(
        direct,
        head_to_kv_group=layout.head_to_kv_group(),
        value_dim=layout.value_dim,
        key_dim=layout.key_dim,
        device=device,
        dtype=dtype,
    )
    route = layout.routing_rank
    return softmax_fisher_routing_diagnostics(
        routing,
        selection_routing_encoders=selection_encoder.to(device=device, dtype=dtype),
        payload_routing_encoders=joint_encoder[..., :route].to(
            device=device,
            dtype=dtype,
        ),
        payload_only_encoders=joint_encoder[..., route:].to(
            device=device,
            dtype=dtype,
        ),
        routing_query_factors=query_factors.to(device=device, dtype=dtype),
        payload_decoders=payload_decoder.to(device=device, dtype=dtype),
        page_size=page_size,
        exact_token_budget=exact_token_budget,
    )


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    arms = sorted(records[0]["arms"])
    result: dict[str, Any] = {}
    for arm in arms:
        rows = [record["arms"][arm] for record in records]
        scalar_keys = sorted(
            key
            for key, value in rows[0].items()
            if isinstance(value, (int, float))
            and key
            not in {"rank", "storage_rank", "payload_rank", "routing_rank"}
        )
        result[arm] = {
            key: sum(float(row[key]) for row in rows) / len(rows)
            for key in scalar_keys
        }
        for key in ("rank", "storage_rank", "payload_rank", "routing_rank"):
            if key in rows[0]:
                result[arm][key] = rows[0][key]
        if "selection" in rows[0]:
            selection_keys = sorted(
                key
                for key, value in rows[0]["selection"].items()
                if isinstance(value, (int, float))
                and key not in {"documents", "tokens", "page_size", "exact_token_budget"}
            )
            result[arm]["selection"] = {
                key: sum(float(row["selection"][key]) for row in rows) / len(rows)
                for key in selection_keys
            }
            for key in ("page_size", "exact_token_budget"):
                if key in rows[0]["selection"]:
                    result[arm]["selection"][key] = rows[0]["selection"][key]
    return result


def _recover_joint_encoder(
    folded,
    *,
    dense_v_weight: torch.Tensor,
    layout: S80Layout,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, str, float]:
    """Load the unfolded encoder or recover its V branch by a QR least-squares solve."""

    if folded.joint_encoder is not None:
        return (
            folded.joint_encoder.to(device=device, dtype=dtype),
            "checkpoint",
            0.0,
        )
    dense_v = dense_v_weight.to(device=device, dtype=dtype).reshape(
        layout.num_key_value_heads,
        layout.value_dim,
        layout.hidden_size,
    )
    folded_v = folded.v_joint_proj_weight.to(device=device, dtype=dtype).reshape(
        layout.num_key_value_heads,
        layout.joint_rank,
        layout.hidden_size,
    )
    value_rows = []
    for group in range(layout.num_key_value_heads):
        value_rows.append(
            torch.linalg.lstsq(
                dense_v[group].mT,
                folded_v[group].mT,
                driver="gels",
            ).solution
        )
    value_encoder = torch.stack(value_rows)
    key_encoder = folded.k_joint_encoder.to(device=device, dtype=dtype)
    reconstructed = torch.bmm(value_encoder.mT, dense_v)
    relative_residual = float(
        torch.sqrt(
            (reconstructed - folded_v).square().sum()
            / folded_v.square().sum()
        )
    )
    return (
        torch.cat((value_encoder, key_encoder), dim=1),
        "qr_recovered_from_folded_v",
        relative_residual,
    )


def _write_progress(
    path: Path,
    *,
    status: str,
    args: argparse.Namespace,
    records: list[dict[str, Any]],
    started: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(
        path,
        {
            "format": FORMAT,
            "status": status,
            "command": shlex.join(sys.argv),
            "completed_layers": [record["layer"] for record in records],
            "layers": records,
            "aggregate": {} if not records else _aggregate(records),
            "elapsed_seconds": time.monotonic() - started,
            "environment": {
                "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
                "python": sys.version,
                "torch": torch.__version__,
                "work_dtype": args.work_dtype,
                "work_device": args.work_device,
            },
        },
    )


def main() -> None:
    args = _parse_args()
    torch.set_num_threads(args.torch_num_threads)
    started = time.monotonic()
    device = torch.device(args.work_device)
    dtype = torch.float64 if args.work_dtype == "float64" else torch.float32
    model_root = Path(args.model).expanduser().resolve()
    stats_root = Path(args.stats_dir).expanduser().resolve()
    fit_fisher_root = Path(args.page_fisher_fit_dir).expanduser().resolve()
    validation_fisher_root = Path(args.page_fisher_validation_dir).expanduser().resolve()
    joint_root = Path(args.joint_checkpoint).expanduser().resolve()
    direct_root = (
        None
        if args.direct_validation_dir is None
        else Path(args.direct_validation_dir).expanduser().resolve()
    )
    output_path = Path(args.output_json).expanduser().resolve()
    config = _load_json(model_root / "config.json")
    layout = S80Layout(
        hidden_size=int(config["hidden_size"]),
        num_attention_heads=int(config["num_attention_heads"]),
        num_key_value_heads=int(config["num_key_value_heads"]),
        value_dim=int(config.get("head_dim", int(config["hidden_size"]) // int(config["num_attention_heads"]))),
        key_dim=int(config.get("head_dim", int(config["hidden_size"]) // int(config["num_attention_heads"]))),
        joint_rank=80,
        routing_rank=32,
    )
    layers = _parse_layers(args.layers, int(config["num_hidden_layers"]))
    mapping = layout.head_to_kv_group(device=device)
    stats_manifest = _load_json(stats_root / "manifest.json")
    fit_fisher_manifest = _load_json(fit_fisher_root / "manifest.json")
    validation_fisher_manifest = _load_json(validation_fisher_root / "manifest.json")
    direct_manifest = None if direct_root is None else _load_json(direct_root / "manifest.json")
    joint_manifest, joint_bank = load_s80_factor_bank(joint_root)
    c1_roots = {
        rank: Path(path).expanduser().resolve()
        for rank, path in ((48, args.c1_v48), (64, args.c1_v64), (80, args.c1_v80))
    }
    c1_manifests = {
        rank: _load_json(root / "results.json") for rank, root in c1_roots.items()
    }
    kq_root = Path(args.kq_r32_init).expanduser().resolve()
    kq_manifest = _load_json(kq_root / "result.json")
    kq_tensors = load_file(
        str(kq_root / kq_manifest["artifacts"]["factors"]["file"]),
        device="cpu",
    )
    key_bank = kq_tensors["kq_svd_key_projector"]
    query_bank = kq_tensors["kq_svd_query_projector"]
    weights = _SafetensorWeightReader(model_root)
    layer_records: list[dict[str, Any]] = []

    for ordinal, layer in enumerate(layers, start=1):
        layer_started = time.monotonic()
        print(
            f"[S80 Page-Fisher synergy] layer={layer} "
            f"({ordinal}/{len(layers)}) loading",
            flush=True,
        )
        fit_payload, _ = _load_statistics(
            stats_root,
            stats_manifest,
            split="fit",
            layer_index=layer,
        )
        validation_payload, _ = _load_statistics(
            stats_root,
            stats_manifest,
            split="validation",
            layer_index=layer,
        )
        fit_fisher = _load_compact_fisher_layer(
            fit_fisher_root,
            fit_fisher_manifest,
            layer,
            device=device,
            dtype=dtype,
        )
        validation_fisher = _load_compact_fisher_layer(
            validation_fisher_root,
            validation_fisher_manifest,
            layer,
            device=device,
            dtype=dtype,
        )
        dense_o = weights.tensor(f"model.layers.{layer}.self_attn.o_proj.weight").to(
            device=device,
            dtype=dtype,
        )
        dense_v = weights.tensor(f"model.layers.{layer}.self_attn.v_proj.weight")
        full_fit_payload = _full_payload_objective(
            fit_payload,
            dense_o_weight=dense_o,
            layout=layout,
            device=device,
            dtype=dtype,
        )
        full_validation_payload = _full_payload_objective(
            validation_payload,
            dense_o_weight=dense_o,
            layout=layout,
            device=device,
            dtype=dtype,
        )
        value_fit_payload = _payload_objective(
            fit_payload,
            dense_o_weight=dense_o,
            layout=layout,
            device=device,
            dtype=dtype,
        )
        value_validation_payload = _payload_objective(
            validation_payload,
            dense_o_weight=dense_o,
            layout=layout,
            device=device,
            dtype=dtype,
        )

        folded = joint_bank[layer][0]
        joint_encoder, joint_encoder_source, joint_encoder_recovery_residual = (
            _recover_joint_encoder(
            folded,
            dense_v_weight=dense_v,
            layout=layout,
            device=device,
            dtype=dtype,
            )
        )
        joint_decoder = folded.o_decoder_weight.to(device=device, dtype=dtype).mT.reshape(
            layout.num_attention_heads,
            layout.joint_rank,
            layout.hidden_size,
        )
        joint_query = folded.routing_query_factor.to(device=device, dtype=dtype)
        joint_decoder, _ = solve_free_decoder(
            objective=full_fit_payload,
            A_unique=joint_encoder,
            head_to_kv_group=mapping,
            coupling_mode="full_layer",
        )
        value_only_joint = joint_encoder.clone()
        value_only_joint[:, layout.value_dim :] = 0
        value_only_decoder, _ = solve_free_decoder(
            objective=full_fit_payload,
            A_unique=value_only_joint,
            head_to_kv_group=mapping,
            coupling_mode="full_layer",
        )

        route_branches: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        learned_route = joint_encoder[..., : layout.routing_rank]
        for name, encoder in {
            "joint_route_full": learned_route,
            "joint_route_k_only": torch.cat(
                (torch.zeros_like(learned_route[:, : layout.value_dim]), learned_route[:, layout.value_dim :]),
                dim=1,
            ),
            "joint_route_v_only": torch.cat(
                (learned_route[:, : layout.value_dim], torch.zeros_like(learned_route[:, layout.value_dim :])),
                dim=1,
            ),
        }.items():
            refitted_query, _ = refit_page_fisher_query_factors(
                fit_fisher,
                routing_encoders=encoder,
                relative_damping=args.relative_damping,
                relative_tolerance=args.iterative_tolerance,
                max_iterations=args.iterative_max_iterations,
            )
            route_branches[name] = (encoder, refitted_query)
        print(
            f"[S80 Page-Fisher synergy] layer={layer} route-branch U refits complete",
            flush=True,
        )

        hard_payloads: dict[int, tuple[torch.Tensor, torch.Tensor, Any]] = {}
        for rank in (48, 64, 80):
            initial_a, initial_d = _load_c1_layer(
                c1_roots[rank],
                c1_manifests[rank],
                layer,
            )
            payload_fit = fit_routed_ov_joint(
                objective=value_fit_payload,
                initial_A=initial_a.to(device=device, dtype=dtype),
                initial_D=initial_d.to(device=device, dtype=dtype),
                head_to_kv_group=mapping,
                coupling_mode="full_layer",
                maximum_sweeps=args.payload_sweeps,
                minimum_sweeps=args.payload_sweeps,
                relative_objective_tolerance=0.0,
                patience=args.payload_sweeps,
                encoder_relative_damping=args.relative_damping,
                cg_relative_tolerance=args.iterative_tolerance,
                cg_max_iterations=args.iterative_max_iterations,
                final_decoder_solve=True,
                work_dtype=dtype,
                work_device=device,
            )
            hard_payloads[rank] = (
                payload_fit.A_unique,
                payload_fit.D_heads,
                payload_fit,
            )
            print(
                f"[S80 Page-Fisher synergy] layer={layer} V{rank} payload refit "
                f"loss={payload_fit.final_loss:.9g}",
                flush=True,
            )

        hard_routers = {}
        active_key_rows = torch.arange(
            layout.value_dim,
            layout.joint_dim,
            device=device,
        )
        for rank in (16, 32):
            initial_encoder = torch.zeros(
                layout.num_key_value_heads,
                layout.joint_dim,
                rank,
                device=device,
                dtype=dtype,
            )
            initial_encoder[:, layout.value_dim :] = key_bank[layer, :, :, :rank].to(
                device=device,
                dtype=dtype,
            )
            initial_query = query_bank[layer, :, :, :rank].to(device=device, dtype=dtype)
            if initial_query.shape[0] == layout.num_key_value_heads:
                initial_query = initial_query.repeat_interleave(
                    layout.query_heads_per_kv_group,
                    dim=0,
                )
            hard_routers[rank] = fit_page_fisher_router(
                fit_fisher,
                initial_routing_encoders=initial_encoder,
                initial_query_factors=initial_query,
                active_joint_rows=active_key_rows,
                sweeps=args.router_sweeps,
                relative_damping=args.relative_damping,
                relative_tolerance=args.iterative_tolerance,
                max_iterations=args.iterative_max_iterations,
            )
            router = hard_routers[rank]
            print(
                f"[S80 Page-Fisher synergy] layer={layer} K{rank} Page-Fisher "
                f"router loss={_router_metrics(fit_fisher, router.routing_encoders, router.routing_query_factors)['loss']:.9g}",
                flush=True,
            )

        full_route_encoder, full_route_query = route_branches["joint_route_full"]
        arms: dict[str, dict[str, Any]] = {
            "joint80": {
                "storage_rank": 80,
                "payload_fit_relative_mse": _payload_relative_mse(
                    full_fit_payload,
                    encoders=joint_encoder,
                    decoders=joint_decoder,
                    mapping=mapping,
                    energy=fit_payload.dense_output_energy,
                ),
                "payload_validation_relative_mse": _payload_relative_mse(
                    full_validation_payload,
                    encoders=joint_encoder,
                    decoders=joint_decoder,
                    mapping=mapping,
                    energy=validation_payload.dense_output_energy,
                ),
                "routing_validation_page_fisher_nmse": _router_metrics(
                    validation_fisher,
                    full_route_encoder,
                    full_route_query,
                )["nmse"],
                "routing_fit_page_fisher_nmse": _router_metrics(
                    fit_fisher,
                    full_route_encoder,
                    full_route_query,
                )["nmse"],
                "joint_encoder_source": joint_encoder_source,
                "joint_encoder_recovery_relative_residual": (
                    joint_encoder_recovery_residual
                ),
            },
            "joint80_payload_v_only_refit": {
                "storage_rank": 80,
                "payload_fit_relative_mse": _payload_relative_mse(
                    full_fit_payload,
                    encoders=value_only_joint,
                    decoders=value_only_decoder,
                    mapping=mapping,
                    energy=fit_payload.dense_output_energy,
                ),
                "payload_validation_relative_mse": _payload_relative_mse(
                    full_validation_payload,
                    encoders=value_only_joint,
                    decoders=value_only_decoder,
                    mapping=mapping,
                    energy=validation_payload.dense_output_energy,
                ),
            },
            "joint80_checkpoint_u": {
                "storage_rank": 80,
                "routing_fit_page_fisher_nmse": _router_metrics(
                    fit_fisher,
                    learned_route,
                    joint_query,
                )["nmse"],
                "routing_validation_page_fisher_nmse": _router_metrics(
                    validation_fisher,
                    learned_route,
                    joint_query,
                )["nmse"],
            },
        }
        for name, (encoder, query_factor) in route_branches.items():
            arms[name] = {
                "rank": layout.routing_rank,
                "routing_fit_page_fisher_nmse": _router_metrics(
                    fit_fisher,
                    encoder,
                    query_factor,
                )["nmse"],
                "routing_validation_page_fisher_nmse": _router_metrics(
                    validation_fisher,
                    encoder,
                    query_factor,
                )["nmse"],
            }
        for name, payload_rank, routing_rank in (
            ("hard_v48_k32", 48, 32),
            ("hard_v64_k16", 64, 16),
            ("independent_v80_k32", 80, 32),
        ):
            payload_encoder, payload_decoder, payload_fit = hard_payloads[payload_rank]
            router = hard_routers[routing_rank]
            router_fit_metrics = _router_metrics(
                fit_fisher,
                router.routing_encoders,
                router.routing_query_factors,
            )
            arms[name] = {
                "storage_rank": payload_rank + routing_rank,
                "payload_rank": payload_rank,
                "routing_rank": routing_rank,
                "payload_validation_relative_mse": _payload_relative_mse(
                    value_validation_payload,
                    encoders=payload_encoder,
                    decoders=payload_decoder,
                    mapping=mapping,
                    energy=validation_payload.dense_output_energy,
                ),
                "payload_fit_relative_mse": _payload_relative_mse(
                    value_fit_payload,
                    encoders=payload_encoder,
                    decoders=payload_decoder,
                    mapping=mapping,
                    energy=fit_payload.dense_output_energy,
                ),
                "routing_fit_page_fisher_nmse": router_fit_metrics["nmse"],
                "routing_validation_page_fisher_nmse": _router_metrics(
                    validation_fisher,
                    router.routing_encoders,
                    router.routing_query_factors,
                )["nmse"],
                "payload_final_fit_loss": payload_fit.final_loss,
                "router_final_fit_loss": router_fit_metrics["loss"],
                "router_query_solver": _cg_summary(router.final_query_diagnostics),
            }

        if direct_root is not None and direct_manifest is not None:
            direct = _load_direct_layer(direct_root, direct_manifest, layer)
            for name, (encoder, query_factor) in route_branches.items():
                arms[name]["selection"] = _joint_selection_metrics(
                    direct,
                    layout=layout,
                    selection_encoder=encoder,
                    query_factors=query_factor,
                    joint_encoder=joint_encoder,
                    payload_decoder=joint_decoder,
                    page_size=args.page_size,
                    exact_token_budget=args.exact_token_budget,
                    device=device,
                    dtype=dtype,
                )
            arms["joint80"]["selection"] = arms["joint_route_full"]["selection"]
            arms["joint80_checkpoint_u"]["selection"] = _joint_selection_metrics(
                direct,
                layout=layout,
                selection_encoder=learned_route,
                query_factors=joint_query,
                joint_encoder=joint_encoder,
                payload_decoder=joint_decoder,
                page_size=args.page_size,
                exact_token_budget=args.exact_token_budget,
                device=device,
                dtype=dtype,
            )
            for name, payload_rank, routing_rank in (
                ("hard_v48_k32", 48, 32),
                ("hard_v64_k16", 64, 16),
                ("independent_v80_k32", 80, 32),
            ):
                payload_encoder, payload_decoder, _ = hard_payloads[payload_rank]
                router = hard_routers[routing_rank]
                arms[name]["selection"] = _selection_metrics(
                    direct,
                    layout=layout,
                    selection_encoder=router.routing_encoders,
                    query_factors=router.routing_query_factors,
                    payload_encoder=_embed_value_encoder(payload_encoder, layout=layout),
                    payload_decoder=payload_decoder,
                    page_size=args.page_size,
                    exact_token_budget=args.exact_token_budget,
                    device=device,
                    dtype=dtype,
                )

        layer_records.append(
            {
                "layer": layer,
                "arms": arms,
                "seconds": time.monotonic() - layer_started,
            }
        )
        print(
            f"[S80 Page-Fisher synergy] layer={layer} "
            f"({ordinal}/{len(layers)}) seconds={layer_records[-1]['seconds']:.2f}",
            flush=True,
        )
        _write_progress(
            output_path,
            status="running",
            args=args,
            records=layer_records,
            started=started,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    payload = {
        "format": FORMAT,
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(sys.argv),
        "protocol": {
            "fisher_unit": "Page64 teacher-mass distribution",
            "page_size": args.page_size,
            "exact_token_budget": args.exact_token_budget,
            "hard_splits": ["V48+K32", "V64+K16"],
            "independent_reference": "V80+K32",
            "branch_refits": "U refit after route branch drop; D refit after payload K drop",
            "payload_sweeps": args.payload_sweeps,
            "router_sweeps": args.router_sweeps,
            "relative_damping": args.relative_damping,
            "iterative_tolerance": args.iterative_tolerance,
            "iterative_max_iterations": args.iterative_max_iterations,
        },
        "sources": {
            "model": str(model_root),
            "statistics": str(stats_root),
            "page_fisher_fit": str(fit_fisher_root),
            "page_fisher_validation": str(validation_fisher_root),
            "joint_checkpoint": str(joint_root),
            "direct_validation": None if direct_root is None else str(direct_root),
        },
        "layers": layer_records,
        "aggregate": _aggregate(layer_records),
        "elapsed_seconds": time.monotonic() - started,
        "environment": {
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "python": sys.version,
            "torch": torch.__version__,
            "work_dtype": args.work_dtype,
            "work_device": str(device),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_path, payload)
    print(f"[S80 Page-Fisher synergy] wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
