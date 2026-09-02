#!/usr/bin/env python3
"""Fit a post-RoPE K-only Route32 router with Page-output pullback."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys
import time
from typing import Any

from safetensors.torch import load_file, save_file
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basisserve.core.gqa_joint_routing_payload_s80_ablation import (  # noqa: E402
    fit_page_fisher_router,
)
from basisserve.core.gqa_joint_routing_payload_s80_fisher import (  # noqa: E402
    S80CompactSoftmaxFisherRouting,
    compact_softmax_fisher_loss,
    prepare_compact_softmax_fisher_routing,
)


FORMAT = "basisserve.qwen3.s80_page_output_pullback_router.v1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-dir", required=True)
    parser.add_argument("--validation-dir", required=True)
    parser.add_argument("--kq-r32-init", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--layers", default="all")
    parser.add_argument("--routing-rank", type=int, default=32)
    parser.add_argument("--output-weight", type=float, default=1.0)
    parser.add_argument("--page-fisher-weight", type=float, default=0.0)
    parser.add_argument(
        "--component-normalization",
        choices=("none", "teacher_energy"),
        default="none",
    )
    parser.add_argument("--sweeps", type=int, default=20)
    parser.add_argument("--relative-damping", type=float, default=1e-5)
    parser.add_argument("--iterative-tolerance", type=float, default=1e-5)
    parser.add_argument("--iterative-max-iterations", type=int, default=70)
    parser.add_argument("--work-device", default="cuda")
    parser.add_argument(
        "--work-dtype",
        choices=("float32", "float64"),
        default="float64",
    )
    parser.add_argument(
        "--factor-dtype",
        choices=("bfloat16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--torch-num-threads", type=int, default=4)
    return parser.parse_args()


def _parse_layers(spec: str, total: int) -> tuple[int, ...]:
    if spec == "all":
        return tuple(range(total))
    selected: set[int] = set()
    for part in spec.split(","):
        item = part.strip()
        if not item:
            continue
        if "-" in item:
            left, right = item.split("-", 1)
            selected.update(range(int(left), int(right) + 1))
        else:
            selected.add(int(item))
    return tuple(sorted(selected))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_safetensors(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    save_file(tensors, str(temporary))
    os.replace(temporary, path)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_metric(
    root: Path,
    manifest: dict[str, Any],
    layer: int,
    *,
    output_weight: float,
    page_fisher_weight: float,
    component_normalization: str,
    device: torch.device,
    dtype: torch.dtype,
) -> S80CompactSoftmaxFisherRouting:
    artifact = manifest["artifacts"][str(layer)]
    tensors = load_file(str(root / artifact["file"]), device="cpu")
    output_energy = float(tensors["teacher_output_energy"])
    page_fisher_energy = float(tensors["teacher_page_fisher_energy"])
    output_scale = float(output_weight)
    page_fisher_scale = float(page_fisher_weight)
    if component_normalization == "teacher_energy":
        output_scale /= output_energy
        page_fisher_scale /= page_fisher_energy
    packed = (
        output_scale * tensors["output_grams_packed_by_head"]
        + page_fisher_scale
        * tensors["page_fisher_grams_packed_by_head"]
    )
    energy = output_scale * output_energy + page_fisher_scale * page_fisher_energy
    return prepare_compact_softmax_fisher_routing(
        queries_by_head=tensors["queries_by_head"],
        fisher_grams_packed_by_head=packed,
        head_to_kv_group=tensors["head_to_kv_group"],
        value_dim=0,
        key_dim=int(tensors["key_dim"]),
        scaling=float(tensors["scaling"]),
        teacher_fisher_energy=energy,
        device=device,
        dtype=dtype,
    )


def _metrics(
    statistics: S80CompactSoftmaxFisherRouting,
    *,
    encoders: torch.Tensor,
    query_factors: torch.Tensor,
) -> dict[str, float | int]:
    loss = compact_softmax_fisher_loss(
        statistics,
        routing_payload_encoders=encoders,
        routing_query_factors=query_factors,
    )
    return {
        "observations_per_head": statistics.documents,
        "loss": loss,
        "teacher_energy": statistics.teacher_fisher_energy,
        "relative_loss": loss / statistics.teacher_fisher_energy,
    }


def main() -> None:
    args = _parse_args()
    torch.set_num_threads(args.torch_num_threads)
    started = time.monotonic()
    fit_root = Path(args.fit_dir).expanduser().resolve()
    validation_root = Path(args.validation_dir).expanduser().resolve()
    init_root = Path(args.kq_r32_init).expanduser().resolve()
    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    fit_manifest_path = fit_root / "manifest.json"
    validation_manifest_path = validation_root / "manifest.json"
    init_manifest_path = init_root / "result.json"
    fit_manifest = _load_json(fit_manifest_path)
    validation_manifest = _load_json(validation_manifest_path)
    init_manifest = _load_json(init_manifest_path)
    init_tensors = load_file(
        str(init_root / init_manifest["artifacts"]["factors"]["file"]),
        device="cpu",
    )
    key_bank = init_tensors["kq_svd_key_projector"]
    query_bank = init_tensors["kq_svd_query_projector"]
    total_layers = int(fit_manifest["geometry"]["layers"])
    layers = _parse_layers(args.layers, total_layers)
    device = torch.device(args.work_device)
    dtype = torch.float64 if args.work_dtype == "float64" else torch.float32
    factor_dtype = (
        torch.bfloat16 if args.factor_dtype == "bfloat16" else torch.float32
    )
    records: list[dict[str, Any]] = []
    artifacts: dict[str, Any] = {}

    for ordinal, layer in enumerate(layers, start=1):
        layer_started = time.monotonic()
        print(
            f"[Page output router] layer={layer} ({ordinal}/{len(layers)}) loading",
            flush=True,
        )
        fit_statistics = _load_metric(
            fit_root,
            fit_manifest,
            layer,
            output_weight=args.output_weight,
            page_fisher_weight=args.page_fisher_weight,
            component_normalization=args.component_normalization,
            device=device,
            dtype=dtype,
        )
        validation_statistics = _load_metric(
            validation_root,
            validation_manifest,
            layer,
            output_weight=args.output_weight,
            page_fisher_weight=args.page_fisher_weight,
            component_normalization=args.component_normalization,
            device=device,
            dtype=dtype,
        )
        rank = int(args.routing_rank)
        initial_encoder = key_bank[layer, :, :, :rank].to(
            device=device,
            dtype=dtype,
        )
        initial_query = query_bank[layer, :, :, :rank].to(
            device=device,
            dtype=dtype,
        )
        if initial_query.shape[0] != fit_statistics.queries_by_head.shape[0]:
            repeats = (
                fit_statistics.queries_by_head.shape[0]
                // initial_query.shape[0]
            )
            initial_query = initial_query.repeat_interleave(repeats, dim=0)
        initial_fit = _metrics(
            fit_statistics,
            encoders=initial_encoder,
            query_factors=initial_query,
        )
        initial_validation = _metrics(
            validation_statistics,
            encoders=initial_encoder,
            query_factors=initial_query,
        )
        fitted = fit_page_fisher_router(
            fit_statistics,
            initial_routing_encoders=initial_encoder,
            initial_query_factors=initial_query,
            active_joint_rows=torch.arange(
                fit_statistics.key_dim,
                device=device,
            ),
            sweeps=args.sweeps,
            relative_damping=args.relative_damping,
            relative_tolerance=args.iterative_tolerance,
            max_iterations=args.iterative_max_iterations,
        )
        fit_metrics = _metrics(
            fit_statistics,
            encoders=fitted.routing_encoders,
            query_factors=fitted.routing_query_factors,
        )
        validation_metrics = _metrics(
            validation_statistics,
            encoders=fitted.routing_encoders,
            query_factors=fitted.routing_query_factors,
        )
        artifact_path = output_root / f"layer_{layer:03d}.safetensors"
        _atomic_safetensors(
            artifact_path,
            {
                "key_routing_encoders": fitted.routing_encoders.to(
                    device="cpu",
                    dtype=factor_dtype,
                ).contiguous(),
                "query_routing_factors": fitted.routing_query_factors.to(
                    device="cpu",
                    dtype=factor_dtype,
                ).contiguous(),
            },
        )
        record = {
            "layer": layer,
            "initial_fit": initial_fit,
            "initial_validation": initial_validation,
            "fit": fit_metrics,
            "validation": validation_metrics,
            "sweeps": [asdict(item) for item in fitted.sweeps],
            "final_query_iterations": [
                item.iterations for item in fitted.final_query_diagnostics
            ],
            "encoder_iterations_by_sweep": [
                [item.iterations for item in sweep]
                for sweep in fitted.encoder_diagnostics
            ],
            "elapsed_seconds": time.monotonic() - layer_started,
        }
        records.append(record)
        artifacts[str(layer)] = {
            "file": artifact_path.name,
            "sha256": _sha256(artifact_path),
            "factor_dtype": args.factor_dtype,
        }
        print(
            f"[Page output router] layer={layer} "
            f"fit={fit_metrics['relative_loss']:.8f} "
            f"validation={validation_metrics['relative_loss']:.8f} "
            f"initial_validation={initial_validation['relative_loss']:.8f} "
            f"elapsed={record['elapsed_seconds']:.1f}s",
            flush=True,
        )

    result = {
        "format": FORMAT,
        "status": "complete",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(sys.argv),
        "elapsed_seconds": time.monotonic() - started,
        "objective": {
            "routing_coordinates": "post_rope_K_only",
            "output_pullback_weight": args.output_weight,
            "page_fisher_weight": args.page_fisher_weight,
            "component_normalization": args.component_normalization,
            "page_output_metric": (
                "Kbar.T J_rho Cbar (D D.T) Cbar.T J_rho Kbar"
            ),
        },
        "solver": {
            "method": "deterministic_BCD_PCG",
            "sweeps": args.sweeps,
            "relative_damping": args.relative_damping,
            "relative_tolerance": args.iterative_tolerance,
            "max_iterations": args.iterative_max_iterations,
            "work_device": args.work_device,
            "work_dtype": args.work_dtype,
        },
        "geometry": {
            "layers": total_layers,
            "layer_coverage": list(layers),
            "routing_rank": args.routing_rank,
            "key_dim": int(fit_manifest["geometry"]["key_dim"]),
            "query_heads": int(fit_manifest["geometry"]["query_heads"]),
            "kv_heads": int(fit_manifest["geometry"]["kv_heads"]),
            "page_size": int(fit_manifest["geometry"]["page_size"]),
        },
        "sources": {
            "fit_manifest": str(fit_manifest_path),
            "fit_manifest_sha256": _sha256(fit_manifest_path),
            "validation_manifest": str(validation_manifest_path),
            "validation_manifest_sha256": _sha256(validation_manifest_path),
            "initialization_manifest": str(init_manifest_path),
            "initialization_manifest_sha256": _sha256(init_manifest_path),
        },
        "records": records,
        "artifacts": artifacts,
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
    _atomic_json(output_root / "result.json", result)
    print(f"[Page output router] wrote {output_root / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
