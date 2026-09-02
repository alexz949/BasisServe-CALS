#!/usr/bin/env python3
"""Evaluate single-stage Base16 Page32 routing over physical GQA budgets."""

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

from safetensors.torch import load_file, save_file
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basisserve.core.exact_qk_v_offload import (  # noqa: E402
    gqa_group_max_page_mass_mask,
)
from basisserve.core.gqa_joint_routing_payload_s80_fisher import (  # noqa: E402
    softmax_fisher_transform,
)
from evaluation.eval_qwen3_8b_v80_conditional_residual_router import (  # noqa: E402
    _apply_base,
    _discover_capture,
    _fit_base_maps,
    _load_direct,
    _parse_ints,
    _post_rope_rows,
    _rotary_embeddings,
    _stack_base_map,
    _value_codes,
)


FORMAT = "basisserve.qwen3_8b.v80_base16_page32_budget_sweep.v1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--calibration-root", required=True)
    parser.add_argument("--fresh-direct-dir", required=True)
    parser.add_argument("--c1-v80", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--layers", default="0,17,35")
    parser.add_argument("--base-rank", type=int, default=16)
    parser.add_argument("--page-size", type=int, default=32)
    parser.add_argument("--page-counts", default="32,48,64,96,128")
    parser.add_argument("--work-device", default="cuda")
    parser.add_argument("--torch-num-threads", type=int, default=4)
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


def _new_budget_accumulator() -> dict[str, Any]:
    return {
        "page_recall": [],
        "mass_recall": [],
        "oracle_mass_recall": [],
        "oracle_mass_coverage": [],
        "selected_fraction": [],
        "dense_output_energy": 0.0,
        "oracle_output_error": 0.0,
        "refined_output_error": 0.0,
        "base_output_error": 0.0,
        "selected_score_output_error": 0.0,
    }


def _output_from_scores(
    scores: torch.Tensor,
    *,
    token_mask: torch.Tensor,
    head_codes: torch.Tensor,
    output_decoder: torch.Tensor,
) -> torch.Tensor:
    heads_per_group = int(scores.shape[0]) // int(token_mask.shape[0])
    query_mask = token_mask.repeat_interleave(heads_per_group, dim=0)
    probabilities = torch.softmax(
        scores.masked_fill(~query_mask, -torch.inf),
        dim=-1,
    )
    latent = torch.einsum("ht,htr->hr", probabilities, head_codes)
    return torch.einsum("hr,hro->o", latent, output_decoder)


def _update_budget(
    accumulator: dict[str, Any],
    *,
    page_count: int,
    page_size: int,
    proxy_scores: torch.Tensor,
    exact_scores: torch.Tensor,
    exact_probabilities: torch.Tensor,
    head_codes: torch.Tensor,
    output_decoder: torch.Tensor,
    dense_output: torch.Tensor,
    num_kv_heads: int,
) -> None:
    query_heads, tokens = map(int, exact_scores.shape)
    heads_per_group = query_heads // num_kv_heads
    selected_tokens, selected_pages = gqa_group_max_page_mass_mask(
        proxy_scores,
        num_kv_heads=num_kv_heads,
        page_size=page_size,
        pages_per_kv_head=page_count,
    )
    oracle_tokens, oracle_pages = gqa_group_max_page_mass_mask(
        exact_scores,
        num_kv_heads=num_kv_heads,
        page_size=page_size,
        pages_per_kv_head=page_count,
    )
    page_recall = (
        (selected_pages & oracle_pages).sum(dim=-1)
        / oracle_pages.sum(dim=-1).clamp_min(1)
    )
    selected_query_mask = selected_tokens.repeat_interleave(
        heads_per_group,
        dim=0,
    )
    oracle_query_mask = oracle_tokens.repeat_interleave(
        heads_per_group,
        dim=0,
    )
    overlap_pages = selected_pages & oracle_pages
    overlap_tokens = overlap_pages.repeat_interleave(page_size, dim=-1)[
        :, :tokens
    ]
    overlap_query_mask = overlap_tokens.repeat_interleave(
        heads_per_group,
        dim=0,
    )
    selected_mass = (exact_probabilities * selected_query_mask).sum(dim=-1)
    oracle_mass = (exact_probabilities * oracle_query_mask).sum(dim=-1)
    overlap_mass = (exact_probabilities * overlap_query_mask).sum(dim=-1)
    oracle_mass_coverage = overlap_mass / oracle_mass.clamp_min(
        torch.finfo(oracle_mass.dtype).tiny
    )

    oracle_output = _output_from_scores(
        exact_scores,
        token_mask=oracle_tokens,
        head_codes=head_codes,
        output_decoder=output_decoder,
    )
    refined_output = _output_from_scores(
        exact_scores,
        token_mask=selected_tokens,
        head_codes=head_codes,
        output_decoder=output_decoder,
    )
    base_output = _output_from_scores(
        proxy_scores,
        token_mask=selected_tokens,
        head_codes=head_codes,
        output_decoder=output_decoder,
    )

    accumulator["page_recall"].extend(page_recall.tolist())
    accumulator["mass_recall"].extend(selected_mass.tolist())
    accumulator["oracle_mass_recall"].extend(oracle_mass.tolist())
    accumulator["oracle_mass_coverage"].extend(
        oracle_mass_coverage.tolist()
    )
    accumulator["selected_fraction"].extend(
        selected_tokens.float().mean(dim=-1).tolist()
    )
    accumulator["dense_output_energy"] += float(dense_output.square().sum())
    accumulator["oracle_output_error"] += float(
        (oracle_output - dense_output).square().sum()
    )
    accumulator["refined_output_error"] += float(
        (refined_output - dense_output).square().sum()
    )
    accumulator["base_output_error"] += float(
        (base_output - dense_output).square().sum()
    )
    accumulator["selected_score_output_error"] += float(
        (base_output - refined_output).square().sum()
    )


def _finish_budget(
    accumulator: dict[str, Any],
    *,
    tokens: int,
) -> dict[str, float]:
    energy = accumulator["dense_output_energy"]
    selected_fraction = sum(accumulator["selected_fraction"]) / len(
        accumulator["selected_fraction"]
    )
    return {
        "physical_page_recall_mean": sum(accumulator["page_recall"])
        / len(accumulator["page_recall"]),
        "physical_page_recall_minimum": min(accumulator["page_recall"]),
        "attention_mass_recall_mean": sum(accumulator["mass_recall"])
        / len(accumulator["mass_recall"]),
        "attention_mass_recall_minimum": min(accumulator["mass_recall"]),
        "oracle_attention_mass_recall_mean": sum(
            accumulator["oracle_mass_recall"]
        )
        / len(accumulator["oracle_mass_recall"]),
        "oracle_attention_mass_recall_minimum": min(
            accumulator["oracle_mass_recall"]
        ),
        "oracle_mass_weighted_page_coverage_mean": sum(
            accumulator["oracle_mass_coverage"]
        )
        / len(accumulator["oracle_mass_coverage"]),
        "oracle_mass_weighted_page_coverage_minimum": min(
            accumulator["oracle_mass_coverage"]
        ),
        "selected_token_fraction_mean": selected_fraction,
        "physical_token_budget_mean": selected_fraction * tokens,
        "oracle_sparse_output_relative_mse": accumulator[
            "oracle_output_error"
        ]
        / energy,
        "exact_refined_output_relative_mse": accumulator[
            "refined_output_error"
        ]
        / energy,
        "deployable_base_output_relative_mse": accumulator[
            "base_output_error"
        ]
        / energy,
        "selected_score_output_error_over_dense_energy": accumulator[
            "selected_score_output_error"
        ]
        / energy,
    }


def _evaluate_layer(
    queries_raw: torch.Tensor,
    rows_raw: torch.Tensor,
    *,
    value_encoder: torch.Tensor,
    output_decoder: torch.Tensor,
    base_factors: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    cos: torch.Tensor,
    sin: torch.Tensor,
    page_size: int,
    page_counts: tuple[int, ...],
    device: torch.device,
) -> dict[str, Any]:
    documents, query_heads, head_dim = map(int, queries_raw.shape)
    _, tokens, groups, _ = map(int, rows_raw.shape)
    heads_per_group = query_heads // groups
    head_to_group = torch.arange(query_heads, device=device) // heads_per_group
    scaling = head_dim**-0.5
    encoder = value_encoder.to(device=device, dtype=torch.float32)
    decoder = output_decoder.to(device=device, dtype=torch.float32)
    budgets = {count: _new_budget_accumulator() for count in page_counts}
    score_error = 0.0
    score_energy = 0.0
    fisher_error = 0.0
    fisher_energy = 0.0
    key_error = 0.0
    key_energy = 0.0

    for document in range(documents):
        print(
            f"    fresh budget evaluation document={document + 1}/{documents}",
            flush=True,
        )
        current = rows_raw[document].to(device=device, dtype=torch.float32)
        queries = queries_raw[document].to(device=device, dtype=torch.float32)
        dense_value = current[..., :head_dim]
        exact_key = current[..., head_dim:]
        codes = _value_codes(dense_value, encoder)
        base_pre = _apply_base(codes, base_factors)
        base_post = _post_rope_rows(base_pre, cos, sin)
        exact_scores = torch.empty(
            query_heads,
            tokens,
            device=device,
            dtype=torch.float32,
        )
        proxy_scores = torch.empty_like(exact_scores)
        for group in range(groups):
            first = group * heads_per_group
            stop = first + heads_per_group
            exact_scores[first:stop] = scaling * (
                queries[first:stop] @ exact_key[:, group].mT
            )
            proxy_scores[first:stop] = scaling * (
                queries[first:stop] @ base_post[:, group].mT
            )
        exact_probabilities = torch.softmax(exact_scores, dim=-1)
        score_delta = proxy_scores - exact_scores
        score_error += float(score_delta.square().sum())
        score_energy += float(exact_scores.square().sum())
        fisher_error += float(
            softmax_fisher_transform(
                score_delta,
                exact_probabilities,
            ).square().sum()
        )
        fisher_energy += float(
            softmax_fisher_transform(
                exact_scores,
                exact_probabilities,
            ).square().sum()
        )
        key_error += float((base_post - exact_key).square().sum())
        key_energy += float(exact_key.square().sum())

        head_codes = codes.permute(1, 0, 2).index_select(0, head_to_group)
        dense_latent = torch.einsum(
            "ht,htr->hr",
            exact_probabilities,
            head_codes,
        )
        dense_output = torch.einsum("hr,hro->o", dense_latent, decoder)
        for page_count in page_counts:
            _update_budget(
                budgets[page_count],
                page_count=page_count,
                page_size=page_size,
                proxy_scores=proxy_scores,
                exact_scores=exact_scores,
                exact_probabilities=exact_probabilities,
                head_codes=head_codes,
                output_decoder=decoder,
                dense_output=dense_output,
                num_kv_heads=groups,
            )
        del (
            current,
            queries,
            dense_value,
            exact_key,
            codes,
            base_pre,
            base_post,
            exact_scores,
            proxy_scores,
            exact_probabilities,
            head_codes,
            dense_latent,
            dense_output,
        )

    return {
        "proxy": {
            "post_rope_key_relative_mse": key_error / key_energy,
            "raw_score_nmse": score_error / score_energy,
            "softmax_fisher_nmse": fisher_error / fisher_energy,
        },
        "budgets": {
            str(page_count * page_size): {
                "pages_per_physical_kv_group": page_count,
                **_finish_budget(budgets[page_count], tokens=tokens),
            }
            for page_count in page_counts
        },
    }


def _aggregate(layer_records: list[dict[str, Any]]) -> dict[str, Any]:
    proxy_keys = tuple(layer_records[0]["evaluation"]["proxy"])
    budget_keys = tuple(layer_records[0]["evaluation"]["budgets"])
    aggregate = {
        "proxy": {
            key: sum(
                float(record["evaluation"]["proxy"][key])
                for record in layer_records
            )
            / len(layer_records)
            for key in proxy_keys
        },
        "budgets": {},
    }
    for budget in budget_keys:
        rows = [
            record["evaluation"]["budgets"][budget]
            for record in layer_records
        ]
        aggregate["budgets"][budget] = {}
        for key, value in rows[0].items():
            if key == "pages_per_physical_kv_group":
                aggregate["budgets"][budget][key] = int(value)
                continue
            values = [float(row[key]) for row in rows]
            aggregate["budgets"][budget][key] = (
                min(values)
                if key.endswith("minimum")
                else sum(values) / len(values)
            )
    return aggregate


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
    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.work_device)
    layers = _parse_ints(args.layers)
    page_counts = _parse_ints(args.page_counts)
    sequence = 32768
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
    layer_records: list[dict[str, Any]] = []
    result_path = output_root / "result.json"

    for ordinal, layer in enumerate(layers, start=1):
        layer_started = time.monotonic()
        print(
            f"[Base16 Page32 sweep] layer={layer} ({ordinal}/{len(layers)})",
            flush=True,
        )
        fit_root, fit_manifest = _discover_capture(
            calibration_root,
            split="fit",
            layer=layer,
        )
        fit_queries, fit_rows = _load_direct(
            fit_root,
            fit_manifest,
            layer,
        )
        fresh_queries, fresh_rows = _load_direct(
            fresh_root,
            fresh_manifest,
            layer,
        )
        c1_artifact = c1_manifest["artifacts"][str(layer)]
        c1_tensors = load_file(
            str(c1_root / c1_artifact["file"]),
            device="cpu",
        )
        value_encoder = c1_tensors["value_coordinate_encoders"]
        output_decoder = c1_tensors["head_output_decoders"]
        base_maps = _fit_base_maps(
            fit_rows,
            value_encoder=value_encoder,
            base_ranks=(args.base_rank,),
            cos=cos,
            sin=sin,
            device=device,
        )[args.base_rank]
        evaluation = _evaluate_layer(
            fresh_queries,
            fresh_rows,
            value_encoder=value_encoder,
            output_decoder=output_decoder,
            base_factors=_stack_base_map(base_maps, device=device),
            cos=cos,
            sin=sin,
            page_size=args.page_size,
            page_counts=page_counts,
            device=device,
        )
        artifact = f"layer_{layer:03d}.safetensors"
        _write_factors(
            output_root / artifact,
            {
                f"base_left_b{args.base_rank}": torch.stack(
                    [item.left for item in base_maps]
                ).float(),
                f"base_right_b{args.base_rank}": torch.stack(
                    [item.right for item in base_maps]
                ).float(),
                f"base_bias_b{args.base_rank}": torch.stack(
                    [item.bias for item in base_maps]
                ).float(),
            },
        )
        layer_records.append(
            {
                "layer": layer,
                "artifact": artifact,
                "evaluation": evaluation,
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
        del fit_queries, fit_rows, fresh_queries, fresh_rows, c1_tensors
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
            "base": (
                f"rank-{args.base_rank} affine reduced-rank regression from "
                "C1-V80 to pre-RoPE K, followed by exact token RoPE"
            ),
            "residual_rank": 0,
            "selection": "single-stage Base-only routing without exact rerank",
            "fit_capture": (
                "64 independent C4 documents x 32768 tokens; all payload "
                "positions"
            ),
            "final_capture": (
                "fresh 4 independent C4 documents x 32768 tokens; "
                "last-token query"
            ),
            "page_size": args.page_size,
            "pages_per_physical_kv_group": list(page_counts),
            "physical_token_budgets": [
                count * args.page_size for count in page_counts
            ],
            "physical_accounting": (
                "per-query-head normalized page mass, max across each GQA "
                "group, then one fixed physical Top-page set"
            ),
            "diagnostic_exact_k": (
                "exact K is evaluated only on the Base-selected set and does "
                "not change page selection"
            ),
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
    print(f"[Base16 Page32 sweep] wrote {result_path}", flush=True)


if __name__ == "__main__":
    main()
