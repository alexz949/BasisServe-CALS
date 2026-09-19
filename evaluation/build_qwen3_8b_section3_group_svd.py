#!/usr/bin/env python3
"""Build the activation-aware per-KV-group truncated-SVD baseline."""

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

import torch
from torch import Tensor


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation import fit_llama2_mha_c1_joint as joint  # noqa: E402


FORMAT = "basisserve.qwen3_8b.section3.activation_group_svd.v1"
RANKS = (64, 96)


def _check(condition: bool, message: str) -> bool:
    if condition:
        return True
    print(f"[Error] {message}", file=sys.stderr, flush=True)
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rank", type=int, choices=RANKS, required=True)
    parser.add_argument("--layers", default="all")
    parser.add_argument(
        "--factor-dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--torch-num-threads", type=int, default=4)
    parser.add_argument(
        "--direct-svd-audit-groups",
        type=int,
        default=0,
        help="Directly audit this many leading physical groups per selected layer",
    )
    parser.add_argument("--covariance-ridge", type=float, default=1e-5)
    return parser.parse_args()


def _direct_svd_audit(
    target: Tensor,
    covariance: Tensor,
    encoders: Tensor,
    decoders: Tensor,
    mapping: Tensor,
    initialization: list[dict[str, Any]],
    *,
    rank: int,
    groups: int,
) -> list[dict[str, Any]]:
    rows = []
    for group in range(min(groups, joint.NUM_KV_HEADS)):
        heads = torch.nonzero(mapping == group, as_tuple=False).flatten()
        dense = target.index_select(0, heads).permute(1, 0, 2).reshape(
            joint.HEAD_DIM, -1
        )
        pooled = covariance[heads, heads].mean(dim=0)
        pooled = 0.5 * (pooled + pooled.T)
        damped = pooled + float(
            initialization[group]["effective_absolute_ridge"]
        ) * torch.eye(
            joint.HEAD_DIM,
            device=pooled.device,
            dtype=pooled.dtype,
        )
        cholesky = torch.linalg.cholesky(damped)
        whitened = cholesky.T @ dense
        left, singular, right = torch.linalg.svd(whitened, full_matrices=False)
        raw_encoder = torch.linalg.solve_triangular(
            cholesky.T,
            left[:, :rank],
            upper=True,
        )
        expected = raw_encoder @ (singular[:rank].unsqueeze(1) * right[:rank])
        observed = encoders[group] @ torch.cat(
            [decoders[int(head)] for head in heads], dim=1
        )
        denominator = torch.linalg.vector_norm(expected).clamp_min(
            torch.finfo(expected.dtype).tiny
        )
        rows.append(
            {
                "group": group,
                "query_heads": list(map(int, heads.tolist())),
                "relative_weighted_svd_product_difference": float(
                    torch.linalg.vector_norm(observed - expected) / denominator
                ),
            }
        )
    return rows


@torch.no_grad()
def build(args: argparse.Namespace) -> int:
    joint.activate_model_profile("qwen3_8b")
    torch.set_num_threads(args.torch_num_threads)
    device = torch.device(args.device)
    if not _check(device.type == "cuda" and torch.cuda.is_available(), "CUDA is required"):
        return 2
    torch.cuda.set_device(device)
    snapshot_dir = Path(args.snapshot_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not _check(not output_dir.exists(), f"output already exists: {output_dir}"):
        return 2
    manifest = joint._snapshot_manifest(snapshot_dir)
    if not _check(
        manifest.get("format") == joint.COVARIANCE_SNAPSHOT_FORMAT,
        "Section-3 group SVD requires a covariance snapshot with held-out statistics",
    ):
        return 2
    layers = joint._parse_layers(args.layers)
    factor_dtype = joint._factor_dtype(args.factor_dtype)
    mapping = joint._head_to_kv_group(device=device)
    output_dir.mkdir(parents=True)
    records = []
    started = time.perf_counter()
    for layer in layers:
        layer_started = time.perf_counter()
        fit_matrix, heldout_matrix, weight, source = joint._load_covariance_layer(
            snapshot_dir, manifest, layer
        )
        fit_covariance = joint._covariance_matrix_to_blocks(
            fit_matrix, device=device, dtype=torch.float64
        )
        heldout_covariance = joint._covariance_matrix_to_blocks(
            heldout_matrix, device=device, dtype=torch.float64
        )
        target = joint._dense_head_targets(
            weight, device=device, dtype=torch.float64
        )
        initialized = joint.initialize_group_pooled_routed_svd(
            covariance=fit_covariance,
            target=target,
            head_to_kv_group=mapping,
            group_ranks=(args.rank,) * joint.NUM_KV_HEADS,
            covariance_ridge=args.covariance_ridge,
        )
        encoders = initialized.A_unique
        initialization = [asdict(item) for item in initialized.groups]
        decoders = target.new_empty(
            joint.NUM_HEADS,
            args.rank,
            joint.HIDDEN_SIZE,
        )
        for group in range(joint.NUM_KV_HEADS):
            heads = torch.nonzero(mapping == group, as_tuple=False).flatten()
            pooled = fit_covariance[heads, heads].mean(dim=0)
            pooled = 0.5 * (pooled + pooled.T)
            damped = pooled + float(
                initialization[group]["effective_absolute_ridge"]
            ) * torch.eye(
                joint.HEAD_DIM,
                device=device,
                dtype=torch.float64,
            )
            encoder = encoders[group]
            normal = encoder.T @ damped @ encoder
            right = torch.einsum(
                "ri,hio->hro",
                encoder.T @ damped,
                target.index_select(0, heads),
            )
            decoders.index_copy_(0, heads, torch.linalg.solve(normal, right))
        fit_objective = joint.quadratic_from_target(
            covariance=fit_covariance,
            target=target,
            name=f"qwen3_8b_section3_group_svd_fit_layer_{layer:03d}",
            trace_normalize=False,
        )
        heldout_objective = joint.quadratic_from_target(
            covariance=heldout_covariance,
            target=target,
            name=f"qwen3_8b_section3_group_svd_heldout_layer_{layer:03d}",
            trace_normalize=False,
        )
        artifact_A = encoders.to(device="cpu", dtype=factor_dtype).contiguous()
        artifact_D = decoders.to(device="cpu", dtype=factor_dtype).contiguous()
        quantized_A = artifact_A.to(device=device, dtype=torch.float64)
        quantized_D = artifact_D.to(device=device, dtype=torch.float64)
        fit_loss = joint.evaluate_quadratic(
            fit_objective, quantized_A, quantized_D, mapping
        )
        heldout_loss = joint.evaluate_quadratic(
            heldout_objective, quantized_A, quantized_D, mapping
        )
        audit = _direct_svd_audit(
            target,
            fit_covariance,
            encoders,
            decoders,
            mapping,
            initialization,
            rank=args.rank,
            groups=args.direct_svd_audit_groups,
        )
        artifact_path = output_dir / f"layer_{layer:03d}.safetensors"
        tensors = {
            "value_coordinate_encoders": artifact_A,
            "head_output_decoders": artifact_D,
        }
        joint._atomic_safetensors(artifact_path, tensors)
        record = {
            "format": FORMAT,
            "layer": layer,
            "source_snapshot": source,
            "method": "per_kv_group_activation_weighted_svd",
            "rank_per_kv_head": args.rank,
            "physical_kv_groups": joint.NUM_KV_HEADS,
            "query_heads_per_group": joint.NUM_HEADS // joint.NUM_KV_HEADS,
            "fit_factor_dtype_relative_mse": joint._relative_loss(
                fit_loss, fit_objective.constant
            ),
            "heldout_factor_dtype_relative_mse": joint._relative_loss(
                heldout_loss, heldout_objective.constant
            ),
            "maximum_orthogonality_error": float(
                torch.linalg.matrix_norm(
                    encoders.transpose(1, 2) @ encoders
                    - torch.eye(args.rank, device=device, dtype=torch.float64),
                    ord="fro",
                    dim=(-2, -1),
                ).max()
            ),
            "direct_svd_audit": audit,
            "initialization": initialization,
            "artifact": {
                "file": artifact_path.name,
                "sha256": joint._sha256(artifact_path),
                "tensors": {
                    key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                    for key, value in tensors.items()
                },
            },
            "elapsed_seconds": time.perf_counter() - layer_started,
        }
        joint._atomic_json(output_dir / f"layer_{layer:03d}.json", record)
        records.append(record)
        print(
            f"[Group SVD] layer={layer} rank={args.rank} "
            f"heldout={record['heldout_factor_dtype_relative_mse']:.9g}",
            flush=True,
        )
        del (
            fit_matrix,
            heldout_matrix,
            weight,
            fit_covariance,
            heldout_covariance,
            target,
            encoders,
            decoders,
            fit_objective,
            heldout_objective,
            artifact_A,
            artifact_D,
            quantized_A,
            quantized_D,
        )
        torch.cuda.empty_cache()

    calibration = manifest["calibration"]
    fit_config = {
        "model": str(Path(manifest["model"]["path"]).expanduser().resolve()),
        "model_config_sha256": manifest["model"]["config_sha256"],
        "snapshot_dir": str(snapshot_dir),
        "snapshot_manifest_sha256": joint._sha256(snapshot_dir / "manifest.json"),
        "fit_windows": int(calibration["fit_windows"]),
        "validation_windows": int(calibration["heldout_windows"]),
        "positions_per_window": int(calibration["positions_per_window"]),
        "cache_rank_per_head": args.rank,
        "model_label": joint.MODEL_LABEL,
        "model_type": joint.MODEL_TYPE,
        "attention_type": joint.ATTENTION_TYPE,
        "num_query_heads": joint.NUM_HEADS,
        "num_physical_kv_heads": joint.NUM_KV_HEADS,
        "head_dim": joint.HEAD_DIM,
        "hidden_size": joint.HIDDEN_SIZE,
        "num_hidden_layers": joint.NUM_LAYERS,
        "v_retained_ratio": args.rank / joint.HEAD_DIM,
        "factor_dtype": args.factor_dtype,
        "work_dtype": "float64",
        "objective": "independent per-physical-KV-group activation-weighted SVD under pooled diagonal attention-output covariance",
        "covariance_ridge": args.covariance_ridge,
        "cross_head_covariance": False,
        "encoder_sweeps": 0,
        "checkpoint_policy": "closed-form weighted Eckart-Young endpoint",
    }
    payload = {
        "format": joint.FORMAT,
        "section3_format": FORMAT,
        "status": "complete" if layers == tuple(range(joint.NUM_LAYERS)) else "smoke_complete",
        "command": shlex.join(sys.argv),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "layers": list(layers),
        "fit_config": fit_config,
        "artifacts": {str(row["layer"]): row["artifact"] for row in records},
        "records": records,
        "aggregate": {
            "mean_fit_factor_dtype_relative_mse": sum(
                float(row["fit_factor_dtype_relative_mse"]) for row in records
            )
            / len(records),
            "mean_heldout_factor_dtype_relative_mse": sum(
                float(row["heldout_factor_dtype_relative_mse"]) for row in records
            )
            / len(records),
            "maximum_direct_weighted_svd_product_difference": max(
                (
                    float(audit["relative_weighted_svd_product_difference"])
                    for row in records
                    for audit in row["direct_svd_audit"]
                ),
                default=None,
            ),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "environment": {
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_device": torch.cuda.get_device_name(device),
        },
    }
    joint._atomic_json(output_dir / "results.json", payload)
    print(f"[Complete] {output_dir / 'results.json'}", flush=True)
    return 0


def main() -> None:
    status = build(parse_args())
    if status:
        sys.exit(status)


if __name__ == "__main__":
    main()
