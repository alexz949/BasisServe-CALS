#!/usr/bin/env python3
"""Diagnose attention-sink failures in 32K Base16 Page32 routing."""

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

from safetensors.torch import load_file
import torch
from torch.nn import functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.eval_qwen3_8b_v80_base16_page32_budget_sweep import (  # noqa: E402
    _output_from_scores,
)
from evaluation.eval_qwen3_8b_v80_conditional_residual_router import (  # noqa: E402
    _apply_base,
    _load_direct,
    _parse_ints,
    _post_rope_rows,
    _rotary_embeddings,
    _value_codes,
)


FORMAT = "basisserve.qwen3_8b.v80_base16_page32_sink_diagnostics.v1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--fresh-direct-dir", required=True)
    parser.add_argument("--c1-v80", required=True)
    parser.add_argument("--base16-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--layers", default="0,17,35")
    parser.add_argument("--page-size", type=int, default=32)
    parser.add_argument("--physical-page-budget", type=int, default=128)
    parser.add_argument("--prefix-pages", default="0,1,2,4,8")
    parser.add_argument("--histogram-token-width", type=int, default=1024)
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


def _group_page_scores(
    scores: torch.Tensor,
    *,
    num_kv_heads: int,
    page_size: int,
    excluded_prefix_pages: int = 0,
) -> torch.Tensor:
    query_heads, visible = map(int, scores.shape)
    heads_per_group = query_heads // num_kv_heads
    pages = math.ceil(visible / page_size)
    padded = F.pad(scores, (0, pages * page_size - visible), value=-torch.inf)
    page_logits = torch.logsumexp(
        padded.reshape(query_heads, pages, page_size),
        dim=-1,
    )
    page_logits[:, :excluded_prefix_pages] = -torch.inf
    page_mass = torch.softmax(page_logits.float(), dim=-1)
    return page_mass.reshape(
        num_kv_heads,
        heads_per_group,
        pages,
    ).amax(dim=1)


def _fixed_budget_page_mask(
    group_scores: torch.Tensor,
    *,
    page_budget: int,
    forced_prefix_pages: int,
) -> torch.Tensor:
    groups, pages = map(int, group_scores.shape)
    forced = min(forced_prefix_pages, page_budget, pages)
    selected = torch.zeros_like(group_scores, dtype=torch.bool)
    selected[:, :forced] = True
    remaining = min(page_budget - forced, pages - forced)
    if remaining:
        candidates = group_scores.clone()
        candidates[:, :forced] = -torch.inf
        indices = candidates.topk(remaining, dim=-1).indices
        selected.scatter_(1, indices, True)
    return selected


def _token_mask(
    page_mask: torch.Tensor,
    *,
    page_size: int,
    tokens: int,
) -> torch.Tensor:
    return page_mask.repeat_interleave(page_size, dim=-1)[:, :tokens]


def _base_factors(root: Path, layer: int) -> tuple[torch.Tensor, ...]:
    artifact = sorted(root.glob(f"shard_*/layer_{layer:03d}.safetensors"))[0]
    tensors = load_file(str(artifact), device="cpu")
    return (
        tensors["base_left_b16"],
        tensors["base_right_b16"],
        tensors["base_bias_b16"],
    )


def _new_layer_totals(
    *,
    prefix_pages: tuple[int, ...],
    histogram_bins: int,
) -> dict[str, Any]:
    return {
        "query_count": 0,
        "physical_group_count": 0,
        "missed_mass_by_position": [0.0] * histogram_bins,
        "all_mass_by_position": [0.0] * histogram_bins,
        "prefix": {
            str(prefix): {
                "exact_mass_sum": 0.0,
                "base_rank_sum": 0.0,
                "base_rank_count": 0,
                "base_budget_hits": 0,
                "base_budget_hit_count": 0,
                "non_sink_selected_mass_sum": 0.0,
                "non_sink_query_count": 0,
            }
            for prefix in prefix_pages
            if prefix
        },
        "forced": {
            str(prefix): {
                "attention_mass_sum": 0.0,
                "page_recall_sum": 0.0,
                "oracle_mass_coverage_sum": 0.0,
                "dense_output_energy": 0.0,
                "exact_refined_output_error": 0.0,
                "deployable_base_output_error": 0.0,
            }
            for prefix in prefix_pages
        },
    }


def _evaluate_layer(
    queries_raw: torch.Tensor,
    rows_raw: torch.Tensor,
    *,
    value_encoder: torch.Tensor,
    output_decoder: torch.Tensor,
    base_factors: tuple[torch.Tensor, ...],
    cos: torch.Tensor,
    sin: torch.Tensor,
    page_size: int,
    page_budget: int,
    prefix_pages: tuple[int, ...],
    histogram_token_width: int,
    device: torch.device,
) -> dict[str, Any]:
    documents, query_heads, head_dim = map(int, queries_raw.shape)
    _, tokens, groups, _ = map(int, rows_raw.shape)
    heads_per_group = query_heads // groups
    head_to_group = torch.arange(query_heads, device=device) // heads_per_group
    histogram_bins = math.ceil(tokens / histogram_token_width)
    totals = _new_layer_totals(
        prefix_pages=prefix_pages,
        histogram_bins=histogram_bins,
    )
    scaling = head_dim**-0.5
    encoder = value_encoder.to(device=device, dtype=torch.float32)
    decoder = output_decoder.to(device=device, dtype=torch.float32)
    left, right, bias = (
        tensor.to(device=device, dtype=torch.float32)
        for tensor in base_factors
    )

    for document in range(documents):
        print(
            f"    sink diagnostic document={document + 1}/{documents}",
            flush=True,
        )
        current = rows_raw[document].to(device=device, dtype=torch.float32)
        queries = queries_raw[document].to(device=device, dtype=torch.float32)
        dense_value = current[..., :head_dim]
        exact_key = current[..., head_dim:]
        codes = _value_codes(dense_value, encoder)
        base_pre = _apply_base(codes, (left, right, bias))
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
        exact_group_scores = _group_page_scores(
            exact_scores,
            num_kv_heads=groups,
            page_size=page_size,
        )
        proxy_group_scores = _group_page_scores(
            proxy_scores,
            num_kv_heads=groups,
            page_size=page_size,
        )
        oracle_pages = _fixed_budget_page_mask(
            exact_group_scores,
            page_budget=page_budget,
            forced_prefix_pages=0,
        )
        base_pages = _fixed_budget_page_mask(
            proxy_group_scores,
            page_budget=page_budget,
            forced_prefix_pages=0,
        )
        oracle_tokens = _token_mask(
            oracle_pages,
            page_size=page_size,
            tokens=tokens,
        )
        base_tokens = _token_mask(
            base_pages,
            page_size=page_size,
            tokens=tokens,
        )
        missed_tokens = _token_mask(
            oracle_pages & ~base_pages,
            page_size=page_size,
            tokens=tokens,
        ).repeat_interleave(heads_per_group, dim=0)
        for index in range(histogram_bins):
            first = index * histogram_token_width
            stop = min(tokens, first + histogram_token_width)
            totals["missed_mass_by_position"][index] += float(
                (
                    exact_probabilities[:, first:stop]
                    * missed_tokens[:, first:stop]
                ).sum()
            )
            totals["all_mass_by_position"][index] += float(
                exact_probabilities[:, first:stop].sum()
            )

        page_order = proxy_group_scores.argsort(dim=-1, descending=True)
        page_ranks = torch.empty_like(page_order)
        ranks = torch.arange(
            1,
            int(page_order.shape[-1]) + 1,
            device=device,
        ).expand_as(page_order)
        page_ranks.scatter_(1, page_order, ranks)
        for prefix in prefix_pages:
            if not prefix:
                continue
            prefix_tokens = min(tokens, prefix * page_size)
            record = totals["prefix"][str(prefix)]
            record["exact_mass_sum"] += float(
                exact_probabilities[:, :prefix_tokens].sum()
            )
            record["base_rank_sum"] += float(page_ranks[:, :prefix].sum())
            record["base_rank_count"] += groups * prefix
            record["base_budget_hits"] += int(base_pages[:, :prefix].sum())
            record["base_budget_hit_count"] += groups * prefix
            non_sink_probability = exact_probabilities[:, prefix_tokens:]
            non_sink_mask = base_tokens.repeat_interleave(
                heads_per_group,
                dim=0,
            )[:, prefix_tokens:]
            retained = (non_sink_probability * non_sink_mask).sum(dim=-1)
            available = non_sink_probability.sum(dim=-1)
            record["non_sink_selected_mass_sum"] += float(
                (retained / available.clamp_min(torch.finfo(torch.float32).tiny)).sum()
            )
            record["non_sink_query_count"] += query_heads

        head_codes = codes.permute(1, 0, 2).index_select(0, head_to_group)
        dense_latent = torch.einsum(
            "ht,htr->hr",
            exact_probabilities,
            head_codes,
        )
        dense_output = torch.einsum("hr,hro->o", dense_latent, decoder)
        dense_energy = float(dense_output.square().sum())
        oracle_query_mask = oracle_tokens.repeat_interleave(
            heads_per_group,
            dim=0,
        )
        oracle_mass = (
            exact_probabilities * oracle_query_mask
        ).sum(dim=-1)

        for prefix in prefix_pages:
            selected_pages = _fixed_budget_page_mask(
                proxy_group_scores,
                page_budget=page_budget,
                forced_prefix_pages=prefix,
            )
            selected_tokens = _token_mask(
                selected_pages,
                page_size=page_size,
                tokens=tokens,
            )
            selected_query_mask = selected_tokens.repeat_interleave(
                heads_per_group,
                dim=0,
            )
            selected_mass = (
                exact_probabilities * selected_query_mask
            ).sum(dim=-1)
            overlap_tokens = _token_mask(
                selected_pages & oracle_pages,
                page_size=page_size,
                tokens=tokens,
            ).repeat_interleave(heads_per_group, dim=0)
            overlap_mass = (
                exact_probabilities * overlap_tokens
            ).sum(dim=-1)
            refined_output = _output_from_scores(
                exact_scores,
                token_mask=selected_tokens,
                head_codes=head_codes,
                output_decoder=decoder,
            )
            base_output = _output_from_scores(
                proxy_scores,
                token_mask=selected_tokens,
                head_codes=head_codes,
                output_decoder=decoder,
            )
            record = totals["forced"][str(prefix)]
            record["attention_mass_sum"] += float(selected_mass.sum())
            record["page_recall_sum"] += float(
                (
                    (selected_pages & oracle_pages).sum(dim=-1)
                    / oracle_pages.sum(dim=-1).clamp_min(1)
                ).sum()
            )
            record["oracle_mass_coverage_sum"] += float(
                (overlap_mass / oracle_mass.clamp_min(torch.finfo(torch.float32).tiny)).sum()
            )
            record["dense_output_energy"] += dense_energy
            record["exact_refined_output_error"] += float(
                (refined_output - dense_output).square().sum()
            )
            record["deployable_base_output_error"] += float(
                (base_output - dense_output).square().sum()
            )

        totals["query_count"] += query_heads
        totals["physical_group_count"] += groups

    return totals


def _merge_totals(records: list[dict[str, Any]]) -> dict[str, Any]:
    merged = json.loads(json.dumps(records[0]))
    for record in records[1:]:
        merged["query_count"] += record["query_count"]
        merged["physical_group_count"] += record["physical_group_count"]
        for key in ("missed_mass_by_position", "all_mass_by_position"):
            merged[key] = [
                left + right for left, right in zip(merged[key], record[key])
            ]
        for group in ("prefix", "forced"):
            for prefix, values in record[group].items():
                for key, value in values.items():
                    merged[group][prefix][key] += value
    return merged


def _finish(
    totals: dict[str, Any],
    *,
    histogram_token_width: int,
) -> dict[str, Any]:
    query_count = totals["query_count"]
    group_count = totals["physical_group_count"]
    missed_total = sum(totals["missed_mass_by_position"])
    histogram = []
    for index, (missed, all_mass) in enumerate(
        zip(
            totals["missed_mass_by_position"],
            totals["all_mass_by_position"],
        )
    ):
        histogram.append(
            {
                "token_start": index * histogram_token_width,
                "token_stop": (index + 1) * histogram_token_width,
                "all_attention_mass_mean": all_mass / query_count,
                "missed_oracle_mass_mean": missed / query_count,
                "share_of_all_missed_oracle_mass": missed / max(missed_total, 1e-300),
            }
        )
    prefix_metrics = {}
    for prefix, record in totals["prefix"].items():
        prefix_metrics[prefix] = {
            "prefix_tokens": int(prefix) * 32,
            "exact_attention_mass_mean": record["exact_mass_sum"] / query_count,
            "base_page_rank_mean": record["base_rank_sum"]
            / record["base_rank_count"],
            "base_b4096_hit_rate": record["base_budget_hits"]
            / record["base_budget_hit_count"],
            "non_sink_attention_mass_recall": record[
                "non_sink_selected_mass_sum"
            ]
            / record["non_sink_query_count"],
        }
    forced_metrics = {}
    for prefix, record in totals["forced"].items():
        energy = record["dense_output_energy"]
        forced_metrics[prefix] = {
            "forced_prefix_pages": int(prefix),
            "forced_prefix_tokens": int(prefix) * 32,
            "attention_mass_recall_mean": record["attention_mass_sum"]
            / query_count,
            "physical_page_recall_mean": record["page_recall_sum"]
            / group_count,
            "oracle_mass_weighted_page_coverage_mean": record[
                "oracle_mass_coverage_sum"
            ]
            / query_count,
            "exact_refined_output_relative_mse": record[
                "exact_refined_output_error"
            ]
            / energy,
            "deployable_base_output_relative_mse": record[
                "deployable_base_output_error"
            ]
            / energy,
        }
    return {
        "missed_oracle_mass_position_histogram": histogram,
        "prefix_diagnostics": prefix_metrics,
        "forced_prefix_b4096": forced_metrics,
    }


@torch.inference_mode()
def main() -> None:
    args = _parse_args()
    torch.set_num_threads(args.torch_num_threads)
    torch.backends.cuda.matmul.allow_tf32 = True
    started = time.monotonic()
    model_root = Path(args.model).expanduser().resolve()
    fresh_root = Path(args.fresh_direct_dir).expanduser().resolve()
    c1_root = Path(args.c1_v80).expanduser().resolve()
    base_root = Path(args.base16_root).expanduser().resolve()
    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    layers = _parse_ints(args.layers)
    prefix_pages = _parse_ints(args.prefix_pages)
    device = torch.device(args.work_device)
    sequence = 32768
    cos, sin = _rotary_embeddings(
        model_root,
        sequence=sequence,
        device=device,
    )
    fresh_manifest = json.loads(
        (fresh_root / "manifest.json").read_text(encoding="utf-8")
    )
    c1_manifest = json.loads(
        (c1_root / "results.json").read_text(encoding="utf-8")
    )
    layer_records = []
    result_path = output_root / "result.json"

    for ordinal, layer in enumerate(layers, start=1):
        layer_started = time.monotonic()
        print(
            f"[Base16 sink diagnostics] layer={layer} ({ordinal}/{len(layers)})",
            flush=True,
        )
        queries, rows = _load_direct(fresh_root, fresh_manifest, layer)
        c1_artifact = c1_manifest["artifacts"][str(layer)]
        c1_tensors = load_file(str(c1_root / c1_artifact["file"]), device="cpu")
        totals = _evaluate_layer(
            queries,
            rows,
            value_encoder=c1_tensors["value_coordinate_encoders"],
            output_decoder=c1_tensors["head_output_decoders"],
            base_factors=_base_factors(base_root, layer),
            cos=cos,
            sin=sin,
            page_size=args.page_size,
            page_budget=args.physical_page_budget,
            prefix_pages=prefix_pages,
            histogram_token_width=args.histogram_token_width,
            device=device,
        )
        layer_records.append(
            {
                "layer": layer,
                "metrics": _finish(
                    totals,
                    histogram_token_width=args.histogram_token_width,
                ),
                "totals": totals,
                "seconds": time.monotonic() - layer_started,
            }
        )
        merged = _merge_totals([record["totals"] for record in layer_records])
        _write_json(
            result_path,
            {
                "format": FORMAT,
                "status": "running",
                "command": shlex.join(sys.argv),
                "layers": layer_records,
                "aggregate": _finish(
                    merged,
                    histogram_token_width=args.histogram_token_width,
                ),
                "elapsed_seconds": time.monotonic() - started,
            },
        )
        del queries, rows, c1_tensors
        if device.type == "cuda":
            torch.cuda.empty_cache()

    merged = _merge_totals([record["totals"] for record in layer_records])
    result = {
        "format": FORMAT,
        "status": "complete",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(sys.argv),
        "protocol": {
            "model": str(model_root),
            "sequence_length": sequence,
            "capture": "fresh 4 independent C4 documents; last-token query",
            "page_size": args.page_size,
            "physical_page_budget_per_kv_group": args.physical_page_budget,
            "physical_token_budget_per_kv_group": (
                args.page_size * args.physical_page_budget
            ),
            "selection": "GQA group-max Base16 page mass",
            "forced_prefix_pages": list(prefix_pages),
            "budget_accounting": (
                "forced prefix pages replace Base16-selected pages; total "
                "physical B4096 remains fixed"
            ),
            "non_sink_metric": (
                "teacher attention is renormalized after excluding each "
                "prefix; the original Base16 B4096 selection is retained"
            ),
        },
        "layers": layer_records,
        "aggregate": _finish(
            merged,
            histogram_token_width=args.histogram_token_width,
        ),
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
    print(f"[Base16 sink diagnostics] wrote {result_path}", flush=True)


if __name__ == "__main__":
    main()
