#!/usr/bin/env python3
"""Fit C1-K proxy factors from captured valid post-RoPE Q/K pairs.

The input pair bank is a safetensors file with, for every layer ``L``:
``layers.L.query``, ``layers.L.key``, ``layers.L.query_head`` and optionally
``layers.L.weight``. Query and Key rows are already-sampled valid causal pairs.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from safetensors.torch import load_file, save_file
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basisserve.core.c1_k_proxy_fit import (  # noqa: E402
    KProxyPairSamples,
    fit_gqa_k_proxy,
)


FORMAT = "basisserve.c1_k_proxy.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _parse_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def _available_layers(bank: dict[str, torch.Tensor]) -> list[int]:
    layers = []
    for name in bank:
        pieces = name.split(".")
        if len(pieces) == 3 and pieces[0] == "layers" and pieces[2] == "query":
            layers.append(int(pieces[1]))
    return sorted(set(layers))


def _load_pair_samples(
    pair_bank_path: Path,
    bank: dict[str, torch.Tensor] | None,
    pair_manifest: dict[str, Any] | None,
    *,
    layer: int,
    device: torch.device,
) -> KProxyPairSamples:
    if bank is not None:
        prefix = f"layers.{layer}"
        return KProxyPairSamples(
            query=bank[f"{prefix}.query"],
            key=bank[f"{prefix}.key"],
            query_head=bank[f"{prefix}.query_head"].long(),
            weight=bank.get(f"{prefix}.weight"),
        )
    assert pair_manifest is not None
    artifact = pair_manifest["artifacts"][str(layer)]
    path = pair_bank_path / artifact["file"]
    if _sha256(path) != artifact["sha256"]:
        raise ValueError(f"captured pair hash mismatch at layer {layer}")
    tensors = load_file(str(path), device=str(device))
    required = {"fit_pair_query", "fit_pair_key", "fit_pair_query_head"}
    if not required.issubset(tensors):
        raise ValueError(f"capture layer {layer} is missing {required - set(tensors)}")
    return KProxyPairSamples(
        query=tensors["fit_pair_query"],
        key=tensors["fit_pair_key"],
        query_head=tensors["fit_pair_query_head"].long(),
        weight=tensors.get("fit_pair_weight"),
    )


def _file_or_manifest_hash(path: Path | None) -> str | None:
    if path is None:
        return None
    if path.is_dir():
        candidates = (path / "manifest.json", path / "results.json", path / "result.json")
        target = next((candidate for candidate in candidates if candidate.exists()), None)
        if target is None:
            raise FileNotFoundError(f"no manifest/results JSON found in {path}")
    else:
        target = path
    return _sha256(target)


def fit(args: argparse.Namespace) -> None:
    if args.num_query_heads <= 0 or args.num_kv_heads <= 0:
        raise ValueError("head counts must be positive")
    if args.num_query_heads % args.num_kv_heads:
        raise ValueError("query heads must divide evenly across KV heads")
    pair_bank_path = args.pair_bank.expanduser().resolve()
    device = torch.device(getattr(args, "device", "cpu"))
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if pair_bank_path.is_dir():
        pair_manifest_path = pair_bank_path / "manifest.json"
        pair_manifest = json.loads(pair_manifest_path.read_text(encoding="utf-8"))
        bank = None
        available = sorted(int(layer) for layer in pair_manifest["artifacts"])
        pair_bank_digest_path = pair_manifest_path
    else:
        pair_manifest = None
        bank = load_file(str(pair_bank_path), device=str(device))
        available = _available_layers(bank)
        pair_bank_digest_path = pair_bank_path
    layers = available if args.layers == "all" else _parse_ints(args.layers)
    if not layers or any(layer not in available for layer in layers):
        raise ValueError(f"requested layers are not in pair bank; available={available}")
    ranks = _parse_ints(args.proxy_ranks)
    first_samples = _load_pair_samples(
        pair_bank_path, bank, pair_manifest, layer=layers[0], device=device
    )
    head_dim = int(first_samples.query.shape[-1])
    del first_samples
    if not ranks or min(ranks) <= 0 or max(ranks) > head_dim:
        raise ValueError(f"proxy ranks must be in [1, {head_dim}]")

    output_dir.mkdir(parents=True)
    artifacts: dict[str, dict[str, dict[str, Any]]] = {}
    records = []
    for rank in ranks:
        artifacts[str(rank)] = {}
        for layer in layers:
            samples = _load_pair_samples(
                pair_bank_path,
                bank,
                pair_manifest,
                layer=layer,
                device=device,
            )
            result = fit_gqa_k_proxy(
                samples,
                num_query_heads=args.num_query_heads,
                num_kv_heads=args.num_kv_heads,
                proxy_rank=rank,
                initialization=args.initialization,
                als_sweeps=args.als_sweeps,
                ridge=args.ridge,
                cg_iterations=args.cg_iterations,
                cg_tolerance=args.cg_tolerance,
                accumulation_dtype=(
                    torch.float64 if args.accumulation_dtype == "float64" else torch.float32
                ),
            )
            filename = f"layer_{layer:03d}_rank_{rank:03d}.safetensors"
            path = output_dir / filename
            save_file(
                {
                    "key_encoder": result.factors.key_encoder.float().cpu().contiguous(),
                    "query_encoders": result.factors.query_encoders.float().cpu().contiguous(),
                },
                str(path),
                metadata={
                    "format": FORMAT,
                    "layer": str(layer),
                    "proxy_rank": str(rank),
                },
            )
            artifact = {
                "file": filename,
                "sha256": _sha256(path),
                "pairs": len(samples.query),
                "initial_objective": result.objective_history[0],
                "final_objective": result.objective_history[-1],
            }
            artifacts[str(rank)][str(layer)] = artifact
            records.append({"rank": rank, "layer": layer, **artifact})
            print(
                f"[C1-KRefine fit] rank={rank} layer={layer} "
                f"objective={artifact['initial_objective']:.6e}"
                f"->{artifact['final_objective']:.6e}",
                flush=True,
            )

    model_path = args.model_path.expanduser().resolve() if args.model_path else None
    c1_export = args.c1_export.expanduser().resolve() if args.c1_export else None
    model_config = model_path / "config.json" if model_path and model_path.is_dir() else model_path
    manifest = {
        "format": FORMAT,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_identifier": str(model_path) if model_path else "unspecified",
        "model_config_sha256": _sha256(model_config) if model_config else None,
        "c1_export_sha256": _file_or_manifest_hash(c1_export),
        "pair_bank": {
            "file": str(pair_bank_path),
            "manifest_or_file_sha256": _sha256(pair_bank_digest_path),
        },
        "layer_coverage": layers,
        "proxy_ranks": ranks,
        "head_dim": head_dim,
        "num_kv_heads": args.num_kv_heads,
        "num_query_heads": args.num_query_heads,
        "query_head_to_kv_head": [
            head // (args.num_query_heads // args.num_kv_heads)
            for head in range(args.num_query_heads)
        ],
        "calibration": {
            "source": "pre_sampled_valid_causal_pairs",
            "weight_mode": (
                pair_manifest.get("calibration", {})
                .get("pair_sampling", {})
                .get("weight_mode", "uniform")
                if pair_manifest is not None
                else (
                    "uniform"
                    if not any(name.endswith(".weight") for name in bank)
                    else "stored"
                )
            ),
        },
        "pair_sampling": (
            pair_manifest.get("calibration", {}).get("pair_sampling", {})
            if pair_manifest is not None
            else {"provided_by_pair_bank": True}
        ),
        "fit": {
            "initialization": args.initialization,
            "ridge": args.ridge,
            "als_sweeps": args.als_sweeps,
            "cg_iterations": args.cg_iterations,
            "cg_tolerance": args.cg_tolerance,
            "accumulation_dtype": args.accumulation_dtype,
            "device": str(device),
        },
        "artifacts": artifacts,
        "records": records,
        "git_commit": _git_commit(),
        "torch_version": torch.__version__,
    }
    try:
        import transformers

        manifest["transformers_version"] = transformers.__version__
    except ImportError:
        manifest["transformers_version"] = None
    _atomic_json(output_dir / "manifest.json", manifest)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-bank", type=Path, required=True)
    parser.add_argument("--proxy-ranks", required=True)
    parser.add_argument("--layers", default="all")
    parser.add_argument("--num-query-heads", type=int, required=True)
    parser.add_argument("--num-kv-heads", type=int, required=True)
    parser.add_argument("--initialization", choices=("pca_shared", "als"), default="als")
    parser.add_argument("--als-sweeps", type=int, default=3)
    parser.add_argument("--ridge", type=float, default=1.0e-5)
    parser.add_argument("--cg-iterations", type=int, default=32)
    parser.add_argument("--cg-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--accumulation-dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--c1-export", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


if __name__ == "__main__":
    fit(_parser().parse_args())
