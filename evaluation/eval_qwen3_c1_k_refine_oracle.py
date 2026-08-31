#!/usr/bin/env python3
"""Replay captured C1 attention tensors through the C1-KRefine oracle."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any

from safetensors.torch import load_file
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basisserve.core.c1_k_refine import (  # noqa: E402
    GQAKProxyFactors,
    KProxyConfig,
    c1_k_refine_attention_reference,
    c1_k_refine_attention_streaming,
    k_refine_quality_statistics,
    project_proxy_key,
)


FORMAT = "basisserve.c1_k_refine_oracle.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _parse_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def _parse_strings(value: str) -> list[str]:
    return [item for item in value.split(",") if item]


def _capture_layers(tensors: dict[str, torch.Tensor]) -> list[int]:
    result = []
    for name in tensors:
        pieces = name.split(".")
        if len(pieces) == 3 and pieces[0] == "layers" and pieces[2] == "query":
            result.append(int(pieces[1]))
    return sorted(set(result))


def _load_capture_layer(
    capture_path: Path,
    capture: dict[str, torch.Tensor] | None,
    capture_manifest: dict[str, Any] | None,
    *,
    layer: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    if capture is not None:
        prefix = f"layers.{layer}"
        return (
            capture[f"{prefix}.query"],
            capture[f"{prefix}.exact_key"],
            capture[f"{prefix}.c1_value"],
            capture.get(f"{prefix}.attention_mask"),
            capture.get(f"{prefix}.decoder"),
        )
    assert capture_manifest is not None
    artifact = capture_manifest["artifacts"][str(layer)]
    path = capture_path / artifact["file"]
    if _sha256(path) != artifact["sha256"]:
        raise ValueError(f"capture hash mismatch at layer {layer}")
    tensors = load_file(str(path), device=str(device))
    required = {"query", "exact_key", "c1_value"}
    if not required.issubset(tensors):
        raise ValueError(f"capture layer {layer} is missing {required - set(tensors)}")
    return (
        tensors["query"],
        tensors["exact_key"],
        tensors["c1_value"],
        tensors.get("attention_mask"),
        tensors.get("decoder"),
    )


def _load_c1_decoder(
    c1_dir: Path | None,
    c1_result: dict[str, Any] | None,
    *,
    layer: int,
    device: torch.device,
) -> torch.Tensor | None:
    if c1_dir is None or c1_result is None:
        return None
    artifact = c1_result["artifacts"][str(layer)]
    path = c1_dir / artifact["file"]
    if "sha256" in artifact and _sha256(path) != artifact["sha256"]:
        raise ValueError(f"C1 factor hash mismatch at layer {layer}")
    tensors = load_file(str(path), device=str(device))
    decoder = tensors.get("head_output_decoders")
    if decoder is None:
        raise ValueError(f"C1 factors have no head_output_decoders at layer {layer}")
    return decoder


def _load_factors(
    export_dir: Path,
    manifest: dict[str, Any],
    *,
    rank: int,
    layer: int,
    device: torch.device,
) -> GQAKProxyFactors:
    artifact = manifest["artifacts"][str(rank)][str(layer)]
    path = export_dir / artifact["file"]
    if _sha256(path) != artifact["sha256"]:
        raise ValueError(f"factor hash mismatch for layer={layer}, rank={rank}")
    tensors = load_file(str(path), device=str(device))
    if set(tensors) != {"key_encoder", "query_encoders"}:
        raise ValueError(f"unexpected factor tensors in {path}")
    return GQAKProxyFactors(tensors["key_encoder"], tensors["query_encoders"])


def _timed(callable_, *, device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    value = callable_()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return value, time.perf_counter() - started


def _aggregate(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    metric_names = sorted(
        {
            name
            for record in records
            for name, value in record.items()
            if isinstance(value, (int, float))
            and name not in {"layer", "proxy_rank", "page_size", "exact_token_budget", "recent_exact_window"}
            and math.isfinite(float(value))
        }
    )
    aggregated = {}
    for name in metric_names:
        values = [(float(record[name]), int(record["layer"])) for record in records if name in record and math.isfinite(float(record[name]))]
        if not values:
            continue
        ordered = sorted(value for value, _ in values)
        def percentile(fraction: float) -> float:
            return ordered[min(math.ceil(fraction * len(ordered)) - 1, len(ordered) - 1)]

        worst_value, worst_layer = max(values)
        aggregated[name] = {
            "mean": statistics.fmean(ordered),
            "median": statistics.median(ordered),
            "p90": percentile(0.90),
            "p95": percentile(0.95),
            "worst": worst_value,
            "worst_layer": worst_layer,
            "worst_source": "all_physical_kv_heads",
        }
    return aggregated


def _markdown(payload: dict[str, Any]) -> str:
    records = sorted(
        payload["records"],
        key=lambda row: (
            row.get("c1_latent_relative_l2", math.inf),
            row["resident_proxy_key_bytes"] + row["exact_key_bytes_consulted"],
        ),
    )
    lines = [
        "# C1-KRefine layer oracle",
        "",
        "Logical byte/FLOP counts below are not measured serving speedups.",
        "",
        "| layer | policy | rK | page | exact budget | recent | mode | logical K bytes | C1 latent rel-L2 | attention KL | stream max abs |",
        "|---:|:---|---:|---:|---:|---:|:---|---:|---:|---:|---:|",
    ]
    for row in records:
        logical_k = row["resident_proxy_key_bytes"] + row["exact_key_bytes_consulted"]
        lines.append(
            f"| {row['layer']} | {row['policy']} | {row['proxy_rank']} | "
            f"{row['page_size']} | {row['exact_token_budget']} | "
            f"{row['recent_exact_window']} | {row['page_score_mode']} | "
            f"{logical_k:.0f} | {row['c1_latent_relative_l2']:.6e} | "
            f"{row['attention_kl_exact_to_mixed']:.6e} | "
            f"{row['streaming_materialized_max_abs']:.3e} |"
        )
    return "\n".join(lines) + "\n"


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> None:
    capture_path = args.capture.expanduser().resolve()
    export_dir = args.k_proxy_export.expanduser().resolve()
    manifest = json.loads((export_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") != "basisserve.c1_k_proxy.v1":
        raise ValueError("unsupported K-proxy export format")
    c1_dir = None
    c1_result = None
    if args.c1_export:
        c1_path = args.c1_export.expanduser().resolve()
        if c1_path.is_dir():
            candidates = (
                c1_path / "manifest.json",
                c1_path / "results.json",
                c1_path / "result.json",
            )
            c1_manifest = next(
                (candidate for candidate in candidates if candidate.exists()), None
            )
            if c1_manifest is None:
                raise FileNotFoundError(f"no C1 manifest/results JSON found in {c1_path}")
        else:
            c1_manifest = c1_path
        expected = manifest.get("c1_export_sha256")
        if expected is not None and _sha256(c1_manifest) != expected:
            raise ValueError("C1 export hash does not match K-proxy fit manifest")
        c1_dir = c1_path if c1_path.is_dir() else c1_path.parent
        c1_result = json.loads(c1_manifest.read_text(encoding="utf-8"))
    device = torch.device(args.device)
    if capture_path.is_dir():
        capture_manifest_path = capture_path / "manifest.json"
        capture_manifest = json.loads(capture_manifest_path.read_text(encoding="utf-8"))
        capture = None
        available_layers = sorted(int(layer) for layer in capture_manifest["artifacts"])
        capture_digest_path = capture_manifest_path
        capture_model_hash = capture_manifest.get("model", {}).get("config_sha256")
        factor_model_hash = manifest.get("model_config_sha256")
        if (
            capture_model_hash is not None
            and factor_model_hash is not None
            and capture_model_hash != factor_model_hash
        ):
            raise ValueError("capture and K-proxy factors belong to different models")
        capture_c1_hash = capture_manifest.get("c1_export", {}).get(
            "results_sha256"
        )
        if (
            capture_c1_hash is not None
            and c1_result is not None
            and capture_c1_hash != _sha256(c1_manifest)
        ):
            raise ValueError("capture and replay use different C1 exports")
    else:
        capture_manifest = None
        capture = load_file(str(capture_path), device=str(device))
        available_layers = _capture_layers(capture)
        capture_digest_path = capture_path
    layers = available_layers if args.layers == "all" else _parse_ints(args.layers)
    if not layers or any(layer not in available_layers for layer in layers):
        raise ValueError(f"requested layers are not captured; available={available_layers}")
    ranks = _parse_ints(args.proxy_ranks)
    page_sizes = _parse_ints(args.page_sizes)
    budgets = _parse_ints(args.exact_token_budgets)
    recent_windows = _parse_ints(args.recent_exact_windows)
    modes = _parse_strings(args.page_score_modes)
    policies = _parse_strings(args.policies)
    proxy_dtype = getattr(torch, args.proxy_dtype)
    records: list[dict[str, Any]] = []

    for layer in layers:
        query, exact_key, c1_value, mask, decoder = _load_capture_layer(
            capture_path,
            capture,
            capture_manifest,
            layer=layer,
            device=device,
        )
        example_start = int(getattr(args, "example_start", 0))
        examples = int(getattr(args, "examples", 0))
        example_stop = len(query) if examples == 0 else example_start + examples
        if example_start < 0 or example_stop > len(query) or example_start >= example_stop:
            raise ValueError("requested replay examples are outside the capture")
        query = query[example_start:example_stop]
        exact_key = exact_key[example_start:example_stop]
        c1_value = c1_value[example_start:example_stop]
        if mask is not None:
            mask = mask[example_start:example_stop]
        if decoder is None:
            decoder = _load_c1_decoder(
                c1_dir, c1_result, layer=layer, device=device
            )
        head_dim = int(query.shape[-1])
        kv_heads = int(exact_key.shape[1])
        for rank in ranks:
            factors = _load_factors(
                export_dir, manifest, rank=rank, layer=layer, device=device
            )
            proxy_key = project_proxy_key(exact_key, factors).to(proxy_dtype)
            for page_size in page_sizes:
                exact_config = KProxyConfig(
                    proxy_rank=rank,
                    page_size=page_size,
                    exact_token_budget=int(exact_key.shape[2]),
                    score_policy="full_exact",
                    proxy_dtype=args.proxy_dtype,
                )
                full_exact = c1_k_refine_attention_reference(
                    query,
                    exact_key,
                    proxy_key,
                    c1_value,
                    factors,
                    exact_config,
                    mask,
                    layer_idx=layer,
                )
                for budget in budgets:
                    for recent in recent_windows:
                        for mode in modes:
                            for policy in policies:
                                if policy == "sparse_exact" and budget == 0 and recent == 0:
                                    print(
                                        "[C1-KRefine oracle] skip sparse_exact with empty support",
                                        flush=True,
                                    )
                                    continue
                                config = KProxyConfig(
                                    proxy_rank=rank,
                                    page_size=page_size,
                                    exact_token_budget=budget,
                                    recent_exact_window=recent,
                                    page_score_mode=mode,
                                    score_policy=policy,
                                    proxy_dtype=args.proxy_dtype,
                                )
                                materialized, materialized_seconds = _timed(
                                    lambda: c1_k_refine_attention_reference(
                                        query,
                                        exact_key,
                                        proxy_key,
                                        c1_value,
                                        factors,
                                        config,
                                        mask,
                                        layer_idx=layer,
                                    ),
                                    device=device,
                                )
                                streaming, streaming_seconds = _timed(
                                    lambda: c1_k_refine_attention_streaming(
                                        query,
                                        exact_key,
                                        proxy_key,
                                        c1_value,
                                        factors,
                                        config,
                                        mask,
                                        layer_idx=layer,
                                    ),
                                    device=device,
                                )
                                quality = k_refine_quality_statistics(
                                    full_exact,
                                    materialized,
                                    config=config,
                                    head_dim=head_dim,
                                    num_kv_heads=kv_heads,
                                    attention_mask=mask,
                                    decoder=decoder,
                                    attention_top_k=args.attention_top_k,
                                )
                                record = {
                                    "layer": layer,
                                    "source": "all_physical_kv_heads",
                                    "policy": policy,
                                    "proxy_rank": rank,
                                    "page_size": page_size,
                                    "exact_token_budget": budget,
                                    "recent_exact_window": recent,
                                    "page_score_mode": mode,
                                    **materialized.statistics,
                                    **quality,
                                    "materialized_reference_seconds": materialized_seconds,
                                    "streaming_reference_seconds": streaming_seconds,
                                    "streaming_materialized_max_abs": float(
                                        (streaming.output - materialized.output).abs().max()
                                    ),
                                }
                                records.append(record)
                                print(
                                    f"[C1-KRefine oracle] layer={layer} rK={rank} "
                                    f"page={page_size} budget={budget} recent={recent} "
                                    f"mode={mode} policy={policy} "
                                    f"latent={quality['c1_latent_relative_l2']:.3e}",
                                    flush=True,
                                )

    payload = {
        "format": FORMAT,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "capture": {
            "file": str(capture_path),
            "manifest_or_file_sha256": _sha256(capture_digest_path),
        },
        "k_proxy_export": {
            "directory": str(export_dir),
            "manifest_sha256": _sha256(export_dir / "manifest.json"),
        },
        "configuration": vars(args) | {"capture": str(args.capture), "k_proxy_export": str(args.k_proxy_export), "output_json": str(args.output_json), "output_markdown": str(args.output_markdown), "c1_export": None if args.c1_export is None else str(args.c1_export)},
        "records": records,
        "aggregate": _aggregate(records),
    }
    _atomic_text(args.output_json.expanduser().resolve(), json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    _atomic_text(args.output_markdown.expanduser().resolve(), _markdown(payload))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--c1-export", type=Path)
    parser.add_argument("--k-proxy-export", type=Path, required=True)
    parser.add_argument("--layers", default="all")
    parser.add_argument("--example-start", type=int, default=0)
    parser.add_argument("--examples", type=int, default=0)
    parser.add_argument("--proxy-ranks", required=True)
    parser.add_argument("--page-sizes", default="32,64,128")
    parser.add_argument("--exact-token-budgets", default="0,64,128,256,512,1024")
    parser.add_argument("--recent-exact-windows", default="0,128,256")
    parser.add_argument("--page-score-modes", default="max,logsumexp")
    parser.add_argument("--policies", default="proxy_only,proxy_exact_refine,sparse_exact")
    parser.add_argument("--proxy-dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--attention-top-k", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    return parser


if __name__ == "__main__":
    evaluate(_parser().parse_args())
