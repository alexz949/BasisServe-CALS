#!/usr/bin/env python3
"""Evaluate a same-layer C1-Value-derived Key page index on captured Qwen3 data."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shlex
import statistics
import sys
import time
from typing import Any

from safetensors.torch import save_file
import torch
from transformers import AutoConfig
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basisserve.core.c1_k_reverse_shadow import (  # noqa: E402
    ReverseShadowConfig,
    build_post_rope_k_landmarks,
    c1_k_reverse_shadow_attention,
    reverse_shadow_quality_statistics,
)
from basisserve.core.c1_v_k_index import (  # noqa: E402
    apply_rotary,
    fit_c1_v_to_pre_rope_k,
    invert_rotary,
    page_center_rotary_embeddings,
    project_c1_v_to_pre_rope_k,
    relative_squared_error,
)
from evaluation.eval_qwen3_c1_k_refine_oracle import (  # noqa: E402
    _load_c1_decoder,
    _load_capture_layer,
    _sha256,
)


FORMAT = "basisserve.qwen3_8b.c1_v_k_page_oracle.v1"


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_safetensors(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    save_file(tensors, str(temporary), metadata={"format": FORMAT})
    os.replace(temporary, path)


def _parse_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def _mean_cosine(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    cosine = torch.nn.functional.cosine_similarity(
        reference.detach().float(), candidate.detach().float(), dim=-1
    )
    return float(cosine.mean())


def _rotary_embeddings(
    *,
    model_path: Path,
    example: torch.Tensor,
    sequence: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    config = AutoConfig.from_pretrained(str(model_path), local_files_only=True)
    rotary = Qwen3RotaryEmbedding(config, device=example.device)
    position_ids = torch.arange(
        sequence, device=example.device, dtype=torch.long
    ).unsqueeze(0).expand(int(example.shape[0]), -1)
    return rotary(example, position_ids)


def _policy_specifications(
    *,
    exact_key: torch.Tensor,
    predicted_pre_key: torch.Tensor,
    c1_value: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    page_size: int,
    landmark_dtype: str,
    attention_mask: torch.Tensor | None,
) -> dict[str, tuple[str, Any, float, str]]:
    exact_landmarks = build_post_rope_k_landmarks(
        exact_key,
        page_size=page_size,
        attention_mask=attention_mask,
        landmark_dtype=landmark_dtype,
    )
    predicted_post_key = apply_rotary(predicted_pre_key, cos, sin)
    token_rope_landmarks = build_post_rope_k_landmarks(
        predicted_post_key,
        page_size=page_size,
        attention_mask=attention_mask,
        landmark_dtype=landmark_dtype,
    )
    center_cos, center_sin = page_center_rotary_embeddings(
        cos, sin, page_size=page_size
    )
    centered_post_key = apply_rotary(predicted_pre_key, center_cos, center_sin)
    center_landmarks = build_post_rope_k_landmarks(
        centered_post_key,
        page_size=page_size,
        attention_mask=attention_mask,
        landmark_dtype=landmark_dtype,
    )
    batch, kv_heads, sequence, value_rank = map(int, c1_value.shape)
    pages = math.ceil(sequence / page_size)
    dtype_bytes = torch.empty((), dtype=center_landmarks.values.dtype).element_size()
    v_centroid_bytes = float(
        batch * kv_heads * pages * (value_rank * dtype_bytes + 1)
    )
    return {
        "teacher_mass": (
            "teacher_mass",
            exact_landmarks,
            float(exact_key.numel() * exact_key.element_size()),
            "full exact K oracle",
        ),
        "exact_k_mean": (
            "mean_landmark",
            exact_landmarks,
            float(exact_landmarks.values.numel() * exact_landmarks.values.element_size()),
            "stored post-RoPE K page mean",
        ),
        "exact_k_quest": (
            "quest_minmax",
            exact_landmarks,
            float(
                exact_landmarks.page_mins.numel()
                * exact_landmarks.page_mins.element_size()
                + exact_landmarks.page_maxes.numel()
                * exact_landmarks.page_maxes.element_size()
            ),
            "stored post-RoPE K page Min/Max",
        ),
        "v_to_k_token_rope_mean": (
            "mean_landmark",
            token_rope_landmarks,
            float(
                token_rope_landmarks.values.numel()
                * token_rope_landmarks.values.element_size()
            ),
            "diagnostic upper bound: per-token V-to-K then exact RoPE",
        ),
        "v_to_k_page_center": (
            "mean_landmark",
            center_landmarks,
            v_centroid_bytes,
            "deployable index: C1-V page centroid plus center-position RoPE",
        ),
    }


def _summary(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[
            (
                str(record["policy"]),
                int(record["page_size"]),
                int(record["exact_token_budget"]),
            )
        ].append(record)
    result = []
    for (policy, page_size, budget), rows in sorted(grouped.items()):
        result.append(
            {
                "policy": policy,
                "page_size": page_size,
                "exact_token_budget": budget,
                "layers": len(rows),
                "mean_selected_mass": statistics.fmean(
                    float(row["exact_attention_mass_selected"]) for row in rows
                ),
                "mean_mass_page_recall": statistics.fmean(
                    float(row["exact_top_mass_page_recall"]) for row in rows
                ),
                "mean_c1_latent_relative_l2": statistics.fmean(
                    float(row["c1_latent_relative_l2"]) for row in rows
                ),
                "mean_decoded_output_relative_l2": statistics.fmean(
                    float(row["c1_decoded_output_relative_l2"]) for row in rows
                ),
            }
        )
    return result


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Qwen3-8B same-layer C1-V to K-page oracle",
        "",
        "The V-to-K map is fit on C4 window 0 and evaluated only on the "
        "disjoint C4 window 1. All policies use the actual heldout query and "
        "fetch exact selected K pages; this measures selector quality, not "
        "CPU-offload latency.",
        "",
        "## V-to-K reconstruction",
        "",
        "| layer | fit pre-RoPE rel-MSE | heldout pre-RoPE rel-MSE | "
        "heldout pre-RoPE cosine | heldout post-RoPE rel-MSE |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in payload["reconstruction"]:
        lines.append(
            f"| {row['layer']} | {row['fit_pre_rope_relative_mse']:.6e} | "
            f"{row['heldout_pre_rope_relative_mse']:.6e} | "
            f"{row['heldout_pre_rope_mean_cosine']:.6f} | "
            f"{row['heldout_post_rope_relative_mse']:.6e} |"
        )
    lines.extend(
        [
            "",
            "## Selector quality (mean over requested layers)",
            "",
            "| policy | page | exact-token budget | selected mass | "
            "mass-page recall | C1 latent rel-L2 | decoded rel-L2 |",
            "|:---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["summary"]:
        lines.append(
            f"| {row['policy']} | {row['page_size']} | "
            f"{row['exact_token_budget']} | {row['mean_selected_mass']:.6f} | "
            f"{row['mean_mass_page_recall']:.6f} | "
            f"{row['mean_c1_latent_relative_l2']:.6e} | "
            f"{row['mean_decoded_output_relative_l2']:.6e} |"
        )
    return "\n".join(lines) + "\n"


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    capture_dir = args.capture.expanduser().resolve()
    capture_manifest_path = capture_dir / "manifest.json"
    capture_manifest = json.loads(capture_manifest_path.read_text(encoding="utf-8"))
    model_path = Path(capture_manifest["model"]["path"])
    c1_dir = args.c1_export.expanduser().resolve()
    c1_result_path = c1_dir / "results.json"
    c1_result = json.loads(c1_result_path.read_text(encoding="utf-8"))
    if capture_manifest["c1_export"]["results_sha256"] != _sha256(c1_result_path):
        raise ValueError("capture and replay use different C1 exports")
    device = torch.device(args.device)
    layers = _parse_ints(args.layers)
    page_sizes = _parse_ints(args.page_sizes)
    budgets = _parse_ints(args.exact_token_budgets)
    if not layers or not page_sizes or not budgets:
        raise ValueError("layers, page sizes, and budgets must be nonempty")
    if args.fit_examples <= 0 or args.heldout_examples <= 0:
        raise ValueError("fit and heldout example counts must be positive")
    if args.fit_start + args.fit_examples > args.heldout_start:
        raise ValueError("fit and heldout examples must be disjoint and ordered")

    records: list[dict[str, Any]] = []
    reconstruction: list[dict[str, Any]] = []
    factor_tensors: dict[str, torch.Tensor] = {}
    for layer in layers:
        query, exact_key, c1_value, mask, decoder = _load_capture_layer(
            capture_dir,
            None,
            capture_manifest,
            layer=layer,
            device=device,
        )
        stop = args.heldout_start + args.heldout_examples
        if stop > len(query):
            raise ValueError(f"layer {layer} capture does not contain heldout examples")
        cos, sin = _rotary_embeddings(
            model_path=model_path,
            example=exact_key,
            sequence=int(exact_key.shape[2]),
        )
        pre_rope_key = invert_rotary(exact_key, cos, sin)
        fit_slice = slice(args.fit_start, args.fit_start + args.fit_examples)
        heldout_slice = slice(args.heldout_start, stop)
        factors = fit_c1_v_to_pre_rope_k(
            c1_value[fit_slice], pre_rope_key[fit_slice]
        )
        factor_tensors[f"layers.{layer}.v_to_pre_rope_k"] = factors.weight.cpu()
        fit_prediction = project_c1_v_to_pre_rope_k(c1_value[fit_slice], factors)
        heldout_prediction = project_c1_v_to_pre_rope_k(
            c1_value[heldout_slice], factors
        )
        heldout_post_prediction = apply_rotary(
            heldout_prediction,
            cos[heldout_slice],
            sin[heldout_slice],
        )
        reconstruction.append(
            {
                "layer": layer,
                "fit_pre_rope_relative_mse": relative_squared_error(
                    pre_rope_key[fit_slice], fit_prediction
                ),
                "heldout_pre_rope_relative_mse": relative_squared_error(
                    pre_rope_key[heldout_slice], heldout_prediction
                ),
                "heldout_pre_rope_mean_cosine": _mean_cosine(
                    pre_rope_key[heldout_slice], heldout_prediction
                ),
                "heldout_post_rope_relative_mse": relative_squared_error(
                    exact_key[heldout_slice], heldout_post_prediction
                ),
            }
        )

        heldout_query = query[heldout_slice]
        heldout_key = exact_key[heldout_slice]
        heldout_value = c1_value[heldout_slice]
        heldout_mask = None if mask is None else mask[heldout_slice]
        if decoder is None:
            decoder = _load_c1_decoder(
                c1_dir, c1_result, layer=layer, device=device
            )
        for page_size in page_sizes:
            policies = _policy_specifications(
                exact_key=heldout_key,
                predicted_pre_key=heldout_prediction,
                c1_value=heldout_value,
                cos=cos[heldout_slice],
                sin=sin[heldout_slice],
                page_size=page_size,
                landmark_dtype=args.landmark_dtype,
                attention_mask=heldout_mask,
            )
            reference_landmarks = policies["exact_k_mean"][1]
            full_config = ReverseShadowConfig(
                page_size=page_size,
                exact_token_budget=int(heldout_key.shape[2]),
                selector="teacher_exact",
                landmark_dtype=args.landmark_dtype,
            )
            full_exact = c1_k_reverse_shadow_attention(
                heldout_query,
                reference_landmarks,
                heldout_value,
                full_config,
                heldout_key,
                heldout_mask,
                layer_idx=layer,
            )
            for policy, (
                selector,
                landmarks,
                logical_index_bytes,
                index_description,
            ) in policies.items():
                for budget in budgets:
                    config = ReverseShadowConfig(
                        page_size=page_size,
                        exact_token_budget=budget,
                        selector=selector,
                        landmark_dtype=args.landmark_dtype,
                    )
                    candidate = c1_k_reverse_shadow_attention(
                        heldout_query,
                        landmarks,
                        heldout_value,
                        config,
                        heldout_key,
                        heldout_mask,
                        layer_idx=layer,
                    )
                    quality = reverse_shadow_quality_statistics(
                        full_exact,
                        candidate,
                        query=heldout_query,
                        exact_key=heldout_key,
                        config=config,
                        attention_mask=heldout_mask,
                        decoder=decoder,
                        attention_top_k=args.attention_top_k,
                    )
                    record = {
                        "layer": layer,
                        "policy": policy,
                        "selector": selector,
                        "index_description": index_description,
                        "page_size": page_size,
                        "exact_token_budget": budget,
                        "logical_resident_index_bytes": logical_index_bytes,
                        **candidate.statistics,
                        **quality,
                    }
                    records.append(record)
                    print(
                        f"[C1 V->K] layer={layer} policy={policy} "
                        f"page={page_size} budget={budget} "
                        f"mass={quality['exact_attention_mass_selected']:.4f} "
                        f"decoded={quality['c1_decoded_output_relative_l2']:.3e}",
                        flush=True,
                    )
        del query, exact_key, c1_value, pre_rope_key, factors
        torch.cuda.empty_cache()

    output_factors = args.output_factors.expanduser().resolve()
    _atomic_safetensors(output_factors, factor_tensors)
    payload = {
        "format": FORMAT,
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(sys.argv),
        "method": {
            "fit": "per-physical-head FP64 SVD least squares without ridge",
            "target": "pre-RoPE exact K",
            "selection_query": "actual heldout post-RoPE Q",
            "attention": "exact selected K pages and resident C1-V64",
        },
        "capture": {
            "directory": str(capture_dir),
            "manifest_sha256": _sha256(capture_manifest_path),
            "fit_examples": [args.fit_start, args.fit_start + args.fit_examples],
            "heldout_examples": [args.heldout_start, stop],
        },
        "c1_export": {
            "directory": str(c1_dir),
            "results_sha256": _sha256(c1_result_path),
        },
        "factors": {
            "file": str(output_factors),
            "sha256": _sha256(output_factors),
        },
        "configuration": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "reconstruction": reconstruction,
        "records": records,
        "summary": _summary(records),
        "runtime": {
            "seconds": time.perf_counter() - started,
            "torch_version": torch.__version__,
        },
    }
    output_json = args.output_json.expanduser().resolve()
    output_markdown = args.output_markdown.expanduser().resolve()
    _atomic_text(output_json, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _atomic_text(output_markdown, _markdown(payload))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--c1-export", type=Path, required=True)
    parser.add_argument("--layers", default="0,17,35")
    parser.add_argument("--fit-start", type=int, default=0)
    parser.add_argument("--fit-examples", type=int, default=1)
    parser.add_argument("--heldout-start", type=int, default=1)
    parser.add_argument("--heldout-examples", type=int, default=1)
    parser.add_argument("--page-sizes", default="16,32,64")
    parser.add_argument("--exact-token-budgets", default="64,128,256,512,1024")
    parser.add_argument(
        "--landmark-dtype",
        choices=("float32", "float16", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument("--attention-top-k", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-factors", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    return parser


if __name__ == "__main__":
    evaluate(_parser().parse_args())
