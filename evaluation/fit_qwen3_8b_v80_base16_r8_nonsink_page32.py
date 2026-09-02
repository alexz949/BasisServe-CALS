#!/usr/bin/env python3
"""Fit pinned-sink non-sink Page32 residual-R8 routers for Qwen3-8B."""

from __future__ import annotations

import argparse
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


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basisserve.core.c1_v_conditional_k_router import (  # noqa: E402
    AffineReducedRankMap,
)
from basisserve.core.gqa_joint_routing_payload_s80_fisher import (  # noqa: E402
    softmax_fisher_transform,
)
from evaluation.eval_qwen3_8b_v80_base16_page32_budget_sweep import (  # noqa: E402
    _output_from_scores,
)
from evaluation.eval_qwen3_8b_v80_base16_page32_sink_diagnostics import (  # noqa: E402
    _fixed_budget_page_mask,
    _group_page_scores,
    _token_mask,
)
from evaluation.eval_qwen3_8b_v80_conditional_residual_router import (  # noqa: E402
    _apply_base,
    _build_residual_statistics,
    _discover_capture,
    _fit_residual_grid,
    _load_direct,
    _parse_ints,
    _post_rope_rows,
    _rotary_embeddings,
    _stack_base_map,
    _value_codes,
)


FORMAT = "basisserve.qwen3_8b.v80_base16_r8_nonsink_page32.v1"
BASE_RANK = 16
RESIDUAL_RANKS = (0, 8)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--calibration-root", required=True)
    parser.add_argument("--fresh-direct-dir", required=True)
    parser.add_argument("--c1-v80", required=True)
    parser.add_argument("--base16-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--layers", default="0,17,35")
    parser.add_argument("--page-size", type=int, default=32)
    parser.add_argument("--pinned-prefix-pages", type=int, default=1)
    parser.add_argument("--physical-token-budget", type=int, default=4096)
    parser.add_argument("--router-sweeps", type=int, default=40)
    parser.add_argument("--relative-damping", type=float, default=1e-5)
    parser.add_argument("--iterative-tolerance", type=float, default=1e-5)
    parser.add_argument("--iterative-max-iterations", type=int, default=100)
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


def _write_factors(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    save_file(tensors, str(temporary), metadata={"format": FORMAT})
    os.replace(temporary, path)


def _load_base_maps(root: Path, layer: int) -> tuple[AffineReducedRankMap, ...]:
    artifact = sorted(root.glob(f"shard_*/layer_{layer:03d}.safetensors"))[0]
    tensors = load_file(str(artifact), device="cpu")
    left = tensors["base_left_b16"]
    right = tensors["base_right_b16"]
    bias = tensors["base_bias_b16"]
    return tuple(
        AffineReducedRankMap(
            left=left[group],
            right=right[group],
            bias=bias[group],
        )
        for group in range(int(left.shape[0]))
    )


def _new_metrics() -> dict[str, Any]:
    return {
        "conditional_raw_error": 0.0,
        "conditional_raw_energy": 0.0,
        "conditional_fisher_error": 0.0,
        "conditional_fisher_energy": 0.0,
        "dense_output_energy": 0.0,
        "exact_refined_output_error": 0.0,
        "attention_mass": [],
        "non_sink_attention_mass": [],
        "non_sink_page_recall": [],
        "selected_fraction": [],
    }


def _update_metrics(
    accumulator: dict[str, Any],
    *,
    proxy_scores: torch.Tensor,
    exact_scores: torch.Tensor,
    exact_probabilities: torch.Tensor,
    head_codes: torch.Tensor,
    output_decoder: torch.Tensor,
    dense_output: torch.Tensor,
    num_kv_heads: int,
    page_size: int,
    page_budget: int,
    pinned_prefix_pages: int,
) -> None:
    query_heads, tokens = map(int, exact_scores.shape)
    heads_per_group = query_heads // num_kv_heads
    proxy_group_scores = _group_page_scores(
        proxy_scores,
        num_kv_heads=num_kv_heads,
        page_size=page_size,
        excluded_prefix_pages=pinned_prefix_pages,
    )
    exact_group_scores = _group_page_scores(
        exact_scores,
        num_kv_heads=num_kv_heads,
        page_size=page_size,
        excluded_prefix_pages=pinned_prefix_pages,
    )
    selected_pages = _fixed_budget_page_mask(
        proxy_group_scores,
        page_budget=page_budget,
        forced_prefix_pages=pinned_prefix_pages,
    )
    oracle_pages = _fixed_budget_page_mask(
        exact_group_scores,
        page_budget=page_budget,
        forced_prefix_pages=pinned_prefix_pages,
    )
    selected_tokens = _token_mask(
        selected_pages,
        page_size=page_size,
        tokens=tokens,
    )
    query_mask = selected_tokens.repeat_interleave(heads_per_group, dim=0)
    selected_mass = (exact_probabilities * query_mask).sum(dim=-1)
    first_token = pinned_prefix_pages * page_size
    non_sink_probabilities = exact_probabilities[:, first_token:]
    non_sink_selected = query_mask[:, first_token:]
    non_sink_mass = (
        (non_sink_probabilities * non_sink_selected).sum(dim=-1)
        / non_sink_probabilities.sum(dim=-1).clamp_min(
            torch.finfo(torch.float32).tiny
        )
    )
    routed_selected = selected_pages[:, pinned_prefix_pages:]
    routed_oracle = oracle_pages[:, pinned_prefix_pages:]
    non_sink_page_recall = (
        (routed_selected & routed_oracle).sum(dim=-1)
        / routed_oracle.sum(dim=-1).clamp_min(1)
    )
    exact_sparse_output = _output_from_scores(
        exact_scores,
        token_mask=selected_tokens,
        head_codes=head_codes,
        output_decoder=output_decoder,
    )

    conditional_exact_scores = exact_scores[:, first_token:]
    conditional_proxy_scores = proxy_scores[:, first_token:]
    conditional_probabilities = torch.softmax(conditional_exact_scores, dim=-1)
    score_delta = conditional_proxy_scores - conditional_exact_scores
    accumulator["conditional_raw_error"] += float(score_delta.square().sum())
    accumulator["conditional_raw_energy"] += float(
        conditional_exact_scores.square().sum()
    )
    accumulator["conditional_fisher_error"] += float(
        softmax_fisher_transform(
            score_delta,
            conditional_probabilities,
        ).square().sum()
    )
    accumulator["conditional_fisher_energy"] += float(
        softmax_fisher_transform(
            conditional_exact_scores,
            conditional_probabilities,
        ).square().sum()
    )
    accumulator["dense_output_energy"] += float(dense_output.square().sum())
    accumulator["exact_refined_output_error"] += float(
        (exact_sparse_output - dense_output).square().sum()
    )
    accumulator["attention_mass"].extend(selected_mass.tolist())
    accumulator["non_sink_attention_mass"].extend(non_sink_mass.tolist())
    accumulator["non_sink_page_recall"].extend(non_sink_page_recall.tolist())
    accumulator["selected_fraction"].extend(
        selected_tokens.float().mean(dim=-1).tolist()
    )


def _finish_metrics(
    accumulator: dict[str, Any],
    *,
    tokens: int,
) -> dict[str, float]:
    selected_fraction = sum(accumulator["selected_fraction"]) / len(
        accumulator["selected_fraction"]
    )
    return {
        "conditional_non_sink_raw_score_nmse": accumulator[
            "conditional_raw_error"
        ]
        / accumulator["conditional_raw_energy"],
        "conditional_non_sink_softmax_fisher_nmse": accumulator[
            "conditional_fisher_error"
        ]
        / accumulator["conditional_fisher_energy"],
        "non_sink_physical_page_recall_mean": sum(
            accumulator["non_sink_page_recall"]
        )
        / len(accumulator["non_sink_page_recall"]),
        "non_sink_physical_page_recall_minimum": min(
            accumulator["non_sink_page_recall"]
        ),
        "attention_mass_recall_mean": sum(accumulator["attention_mass"])
        / len(accumulator["attention_mass"]),
        "attention_mass_recall_minimum": min(accumulator["attention_mass"]),
        "non_sink_attention_mass_recall_mean": sum(
            accumulator["non_sink_attention_mass"]
        )
        / len(accumulator["non_sink_attention_mass"]),
        "non_sink_attention_mass_recall_minimum": min(
            accumulator["non_sink_attention_mass"]
        ),
        "selected_token_fraction_mean": selected_fraction,
        "physical_token_budget_mean": selected_fraction * tokens,
        "exact_refined_output_relative_mse": accumulator[
            "exact_refined_output_error"
        ]
        / accumulator["dense_output_energy"],
    }


def _evaluate_fresh(
    queries_raw: torch.Tensor,
    rows_raw: torch.Tensor,
    *,
    value_encoder: torch.Tensor,
    output_decoder: torch.Tensor,
    base_maps: tuple[AffineReducedRankMap, ...],
    factor_bank: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]],
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
    scaling = head_dim**-0.5
    encoder = value_encoder.to(device=device, dtype=torch.float32)
    decoder = output_decoder.to(device=device, dtype=torch.float32)
    base_factors = _stack_base_map(base_maps, device=device)
    residual_encoder, query_factor = (
        tensor.to(device=device, dtype=torch.float32)
        for tensor in factor_bank[(BASE_RANK, 8)]
    )
    accumulators = {"b16_r0": _new_metrics(), "b16_r8": _new_metrics()}

    for document in range(documents):
        print(
            f"    fresh pinned-sink evaluation document={document + 1}/{documents}",
            flush=True,
        )
        current = rows_raw[document].to(device=device, dtype=torch.float32)
        queries = queries_raw[document].to(device=device, dtype=torch.float32)
        dense_value = current[..., :head_dim]
        exact_key = current[..., head_dim:]
        codes = _value_codes(dense_value, encoder)
        base_pre = _apply_base(codes, base_factors)
        base_post = _post_rope_rows(base_pre, cos, sin)
        residual = exact_key - base_post
        exact_scores = torch.empty(
            query_heads,
            tokens,
            device=device,
            dtype=torch.float32,
        )
        base_scores = torch.empty_like(exact_scores)
        residual_scores = torch.empty_like(exact_scores)
        for group in range(groups):
            first = group * heads_per_group
            stop = first + heads_per_group
            exact_scores[first:stop] = scaling * (
                queries[first:stop] @ exact_key[:, group].mT
            )
            base_scores[first:stop] = scaling * (
                queries[first:stop] @ base_post[:, group].mT
            )
            query_code = torch.einsum(
                "hd,hdr->hr",
                queries[first:stop],
                query_factor[first:stop],
            )
            token_code = residual[:, group] @ residual_encoder[group]
            residual_scores[first:stop] = base_scores[first:stop] + scaling * (
                query_code @ token_code.mT
            )
        exact_probabilities = torch.softmax(exact_scores, dim=-1)
        head_codes = codes.permute(1, 0, 2).index_select(0, head_to_group)
        dense_latent = torch.einsum(
            "ht,htr->hr",
            exact_probabilities,
            head_codes,
        )
        dense_output = torch.einsum("hr,hro->o", dense_latent, decoder)
        for name, proxy_scores in (
            ("b16_r0", base_scores),
            ("b16_r8", residual_scores),
        ):
            _update_metrics(
                accumulators[name],
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
        del (
            current,
            queries,
            dense_value,
            exact_key,
            codes,
            base_pre,
            base_post,
            residual,
            exact_scores,
            base_scores,
            residual_scores,
            exact_probabilities,
            head_codes,
            dense_latent,
            dense_output,
        )

    return {
        name: _finish_metrics(accumulator, tokens=tokens)
        for name, accumulator in accumulators.items()
    }


def _aggregate(layer_records: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for arm in ("b16_r0", "b16_r8"):
        fresh_rows = [record["fresh"][arm] for record in layer_records]
        fit_rows = [record["router_fit"][arm] for record in layer_records]
        result[arm] = {
            "base_rank": BASE_RANK,
            "residual_rank": int(arm.rsplit("r", 1)[1]),
            "fit_page_fisher_nmse": sum(
                float(row["fit_page_fisher_nmse"]) for row in fit_rows
            )
            / len(fit_rows),
            "validation_page_fisher_nmse": sum(
                float(row["validation_page_fisher_nmse"])
                for row in fit_rows
            )
            / len(fit_rows),
        }
        for key in fresh_rows[0]:
            values = [float(row[key]) for row in fresh_rows]
            result[arm][key] = (
                min(values)
                if key.endswith("minimum")
                else sum(values) / len(values)
            )
    return result


@torch.inference_mode()
def main() -> None:
    args = _parse_args()
    torch.set_num_threads(args.torch_num_threads)
    torch.backends.cuda.matmul.allow_tf32 = True
    started = time.monotonic()
    model_root = Path(args.model).expanduser().resolve()
    calibration_root = Path(args.calibration_root).expanduser().resolve()
    fresh_root = Path(args.fresh_direct_dir).expanduser().resolve()
    c1_root = Path(args.c1_v80).expanduser().resolve()
    base_root = Path(args.base16_root).expanduser().resolve()
    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    layers = _parse_ints(args.layers)
    device = torch.device(args.work_device)
    sequence = 32768
    page_budget = args.physical_token_budget // args.page_size
    cos, sin = _rotary_embeddings(
        model_root,
        sequence=sequence,
        device=device,
    )
    c1_manifest = json.loads(
        (c1_root / "results.json").read_text(encoding="utf-8")
    )
    fresh_manifest = json.loads(
        (fresh_root / "manifest.json").read_text(encoding="utf-8")
    )
    layer_records = []
    result_path = output_root / "result.json"

    for ordinal, layer in enumerate(layers, start=1):
        layer_started = time.monotonic()
        print(
            f"[Base16 non-sink R8] layer={layer} ({ordinal}/{len(layers)})",
            flush=True,
        )
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
        fit_queries, fit_rows = _load_direct(fit_root, fit_manifest, layer)
        validation_queries, validation_rows = _load_direct(
            validation_root,
            validation_manifest,
            layer,
        )
        fresh_queries, fresh_rows = _load_direct(fresh_root, fresh_manifest, layer)
        c1_artifact = c1_manifest["artifacts"][str(layer)]
        c1_tensors = load_file(str(c1_root / c1_artifact["file"]), device="cpu")
        value_encoder = c1_tensors["value_coordinate_encoders"]
        output_decoder = c1_tensors["head_output_decoders"]
        base_maps = _load_base_maps(base_root, layer)
        base_grid = {BASE_RANK: base_maps}
        fit_statistics, fit_reconstruction = _build_residual_statistics(
            fit_queries,
            fit_rows,
            value_encoder=value_encoder,
            base_maps=base_grid,
            cos=cos,
            sin=sin,
            page_size=args.page_size,
            excluded_prefix_pages=args.pinned_prefix_pages,
            device=device,
        )
        validation_statistics, validation_reconstruction = (
            _build_residual_statistics(
                validation_queries,
                validation_rows,
                value_encoder=value_encoder,
                base_maps=base_grid,
                cos=cos,
                sin=sin,
                page_size=args.page_size,
                excluded_prefix_pages=args.pinned_prefix_pages,
                device=device,
            )
        )
        factor_bank, router_fit = _fit_residual_grid(
            fit_statistics,
            validation_statistics,
            residual_ranks=RESIDUAL_RANKS,
            sweeps=args.router_sweeps,
            relative_damping=args.relative_damping,
            iterative_tolerance=args.iterative_tolerance,
            iterative_max_iterations=args.iterative_max_iterations,
            device=device,
        )
        fresh = _evaluate_fresh(
            fresh_queries,
            fresh_rows,
            value_encoder=value_encoder,
            output_decoder=output_decoder,
            base_maps=base_maps,
            factor_bank=factor_bank,
            cos=cos,
            sin=sin,
            page_size=args.page_size,
            page_budget=page_budget,
            pinned_prefix_pages=args.pinned_prefix_pages,
            device=device,
        )
        left, right, bias = (
            torch.stack([getattr(item, name) for item in base_maps]).float()
            for name in ("left", "right", "bias")
        )
        residual_encoder, residual_query = factor_bank[(BASE_RANK, 8)]
        artifact = f"layer_{layer:03d}.safetensors"
        _write_factors(
            output_root / artifact,
            {
                "base_left_b16": left,
                "base_right_b16": right,
                "base_bias_b16": bias,
                "residual_encoder_b16_r8": residual_encoder,
                "residual_query_b16_r8": residual_query,
            },
        )
        layer_records.append(
            {
                "layer": layer,
                "artifact": artifact,
                "fit_base_reconstruction": fit_reconstruction,
                "validation_base_reconstruction": validation_reconstruction,
                "router_fit": router_fit,
                "fresh": fresh,
                "seconds": time.monotonic() - layer_started,
            }
        )
        _write_json(
            result_path,
            {
                "format": FORMAT,
                "status": "running",
                "command": shlex.join(sys.argv),
                "layers": layer_records,
                "aggregate": _aggregate(layer_records),
                "elapsed_seconds": time.monotonic() - started,
            },
        )
        del fit_statistics, validation_statistics, factor_bank, c1_tensors
        if device.type == "cuda":
            torch.cuda.empty_cache()

    result = {
        "format": FORMAT,
        "status": "complete",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(sys.argv),
        "protocol": {
            "model": str(model_root),
            "fixed_payload": "C1-V80 ALS5",
            "fixed_predictive_base": (
                "existing rank-16 affine C1-V80 to pre-RoPE K map"
            ),
            "residual": (
                "rank-8 score correction over exact post-RoPE K minus "
                "exact-RoPE Base16"
            ),
            "residual_objective": (
                "exact-teacher Page32 Fisher conditioned on non-sink pages"
            ),
            "fit_capture": (
                "64 independent C4 documents x 32768 tokens; last-token query"
            ),
            "validation_capture": (
                "16 independent C4 documents x 32768 tokens; last-token query"
            ),
            "final_capture": (
                "fresh 4 independent C4 documents x 32768 tokens; "
                "last-token query"
            ),
            "page_size": args.page_size,
            "pinned_prefix_pages": args.pinned_prefix_pages,
            "physical_token_budget_per_kv_group": args.physical_token_budget,
            "routed_pages_per_kv_group": (
                page_budget - args.pinned_prefix_pages
            ),
            "selection": (
                "pin Page0, condition per-head page mass over remaining "
                "pages, group-max, then select the remaining fixed budget"
            ),
            "router_sweeps": args.router_sweeps,
            "relative_damping": args.relative_damping,
            "iterative_tolerance": args.iterative_tolerance,
            "iterative_max_iterations": args.iterative_max_iterations,
        },
        "layers": layer_records,
        "aggregate": _aggregate(layer_records),
        "elapsed_seconds": time.monotonic() - started,
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
    _write_json(result_path, result)
    print(f"[Base16 non-sink R8] wrote {result_path}", flush=True)


if __name__ == "__main__":
    main()
