#!/usr/bin/env python3
"""Measure post-RoPE Key similarity and layer-ahead page prefetch quality."""

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

import torch


FORMAT = "basisserve.qwen3_8b.interlayer_k_prefetch.v1"


def _parse_csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item)


def _parse_csv_floats(value: str) -> tuple[float, ...]:
    return tuple(float(item) for item in value.split(",") if item)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--page-size", type=int, default=32)
    parser.add_argument("--token-budget", type=int, default=4096)
    parser.add_argument("--pinned-prefix-pages", type=int, default=1)
    parser.add_argument("--layer-lags", default="1,2,4")
    parser.add_argument("--overfetch-factors", default="1,1.25,1.5,2")
    parser.add_argument("--cka-samples", type=int, default=8192)
    parser.add_argument("--work-device", default="cuda")
    parser.add_argument("--torch-num-threads", type=int, default=4)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    return parser.parse_args()


def _mmap_bfloat16(root: Path, record: dict[str, Any]) -> torch.Tensor:
    shape = tuple(int(size) for size in record["shape"])
    return torch.from_file(
        str(root / record["file"]),
        shared=False,
        size=math.prod(shape),
        dtype=torch.bfloat16,
    ).reshape(shape)


def _exact_page_state(
    queries: torch.Tensor,
    joint_rows: torch.Tensor,
    *,
    page_size: int,
    selected_pages: int,
    pinned_prefix_pages: int,
    maximum_prefetch_pages: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    documents, tokens, kv_heads, joint_dim = map(int, joint_rows.shape)
    query_heads = int(queries.shape[1])
    head_dim = joint_dim // 2
    query_heads_per_group = query_heads // kv_heads
    pages = math.ceil(tokens / page_size)
    selected_masks = torch.zeros(
        documents, kv_heads, pages, dtype=torch.bool
    )
    page_probabilities = torch.empty(
        documents,
        kv_heads,
        query_heads_per_group,
        pages,
        dtype=torch.float32,
    )
    ranked_pages = torch.empty(
        documents,
        kv_heads,
        maximum_prefetch_pages - pinned_prefix_pages,
        dtype=torch.int16,
    )
    scaling = head_dim**-0.5
    routed_pages = selected_pages - pinned_prefix_pages
    ranked_count = maximum_prefetch_pages - pinned_prefix_pages

    for document in range(documents):
        query = queries[document].reshape(
            kv_heads, query_heads_per_group, head_dim
        ).to(device=device, dtype=torch.bfloat16)
        key = joint_rows[document, :, :, head_dim:].permute(1, 0, 2).to(
            device=device,
            dtype=torch.bfloat16,
        )
        scores = torch.matmul(query, key.transpose(-1, -2)) * scaling
        padding = pages * page_size - tokens
        if padding:
            scores = torch.nn.functional.pad(scores, (0, padding), value=-torch.inf)
        page_logits = torch.logsumexp(
            scores.float().reshape(
                kv_heads,
                query_heads_per_group,
                pages,
                page_size,
            ),
            dim=-1,
        )
        page_probabilities[document].copy_(
            torch.softmax(page_logits, dim=-1).cpu()
        )
        routed_mass = torch.softmax(
            page_logits[..., pinned_prefix_pages:], dim=-1
        )
        group_scores = routed_mass.amax(dim=1)
        order = torch.topk(
            group_scores,
            k=ranked_count,
            dim=-1,
            sorted=True,
        ).indices + pinned_prefix_pages
        ranked_pages[document].copy_(order.to(device="cpu", dtype=torch.int16))
        selected_masks[document, :, :pinned_prefix_pages] = True
        selected_masks[document].scatter_(
            dim=-1,
            index=order[:, :routed_pages].cpu(),
            value=True,
        )
        del query, key, scores, page_logits, routed_mass, group_scores, order

    return selected_masks, page_probabilities, ranked_pages


def _prefetch_mask(
    ranked_pages: torch.Tensor,
    *,
    pages: int,
    total_pages: int,
    pinned_prefix_pages: int,
) -> torch.Tensor:
    documents, kv_heads, _ = map(int, ranked_pages.shape)
    mask = torch.zeros(documents, kv_heads, pages, dtype=torch.bool)
    mask[..., :pinned_prefix_pages] = True
    routed = total_pages - pinned_prefix_pages
    mask.scatter_(
        dim=-1,
        index=ranked_pages[..., :routed].long(),
        value=True,
    )
    return mask


def _set_metrics(
    prefetched: torch.Tensor,
    actual: torch.Tensor,
    target_page_probability: torch.Tensor,
    *,
    pinned_prefix_pages: int,
) -> dict[str, float]:
    intersection = (prefetched & actual).sum(dim=-1).float()
    actual_count = actual.sum(dim=-1).float()
    prefetched_count = prefetched.sum(dim=-1).float()
    union = (prefetched | actual).sum(dim=-1).float()
    per_head_mass = (
        target_page_probability * prefetched.unsqueeze(2).float()
    ).sum(dim=-1)
    non_sink_probability = target_page_probability[..., pinned_prefix_pages:]
    non_sink_probability = non_sink_probability / non_sink_probability.sum(
        dim=-1, keepdim=True
    ).clamp_min(torch.finfo(torch.float32).tiny)
    non_sink_mass = (
        non_sink_probability
        * prefetched[..., pinned_prefix_pages:].unsqueeze(2).float()
    ).sum(dim=-1)
    return {
        "page_recall": float((intersection / actual_count).mean()),
        "page_precision": float((intersection / prefetched_count).mean()),
        "page_jaccard": float((intersection / union).mean()),
        "late_pages_per_kv_head": float((actual_count - intersection).mean()),
        "wasted_pages_per_kv_head": float(
            (prefetched_count - intersection).mean()
        ),
        "fully_covered_kv_head_fraction": float(
            (intersection == actual_count).float().mean()
        ),
        "target_attention_mass_mean": float(per_head_mass.mean()),
        "target_attention_mass_worst_head_mean": float(
            per_head_mass.amin(dim=2).mean()
        ),
        "target_non_sink_mass_mean": float(non_sink_mass.mean()),
        "target_non_sink_mass_worst_head_mean": float(
            non_sink_mass.amin(dim=2).mean()
        ),
    }


def _mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    return {
        key: sum(float(row[key]) for row in rows) / len(rows)
        for key in rows[0]
    }


def _cka_matrices(
    sampled_keys: list[torch.Tensor],
    *,
    device: torch.device,
) -> tuple[list[list[float]], list[list[list[float]]]]:
    layers = len(sampled_keys)
    kv_heads = int(sampled_keys[0].shape[1])
    per_head = [
        [[0.0 for _ in range(layers)] for _ in range(layers)]
        for _ in range(kv_heads)
    ]
    for head in range(kv_heads):
        features = torch.stack(
            [keys[:, head] for keys in sampled_keys], dim=0
        ).to(device=device, dtype=torch.float32)
        features.sub_(features.mean(dim=1, keepdim=True))
        self_norms = []
        for layer in range(layers):
            gram = features[layer].mT @ features[layer]
            self_norms.append(torch.linalg.vector_norm(gram))
        self_norms_tensor = torch.stack(self_norms)
        for left in range(layers):
            per_head[head][left][left] = 1.0
            for right in range(left + 1, layers):
                cross = features[left].mT @ features[right]
                value = float(
                    cross.square().sum()
                    / (self_norms_tensor[left] * self_norms_tensor[right]).clamp_min(
                        torch.finfo(torch.float32).tiny
                    )
                )
                per_head[head][left][right] = value
                per_head[head][right][left] = value
        del features, self_norms_tensor
    mean = [
        [
            sum(per_head[head][left][right] for head in range(kv_heads))
            / kv_heads
            for right in range(layers)
        ]
        for left in range(layers)
    ]
    return mean, per_head


def _markdown(payload: dict[str, Any]) -> str:
    protocol = payload["protocol"]
    lines = [
        "# Qwen3-8B inter-layer Key and page-prefetch analysis",
        "",
        f"Capture: `{protocol['documents']}×{protocol['sequence_length']}` C4, "
        f"post-RoPE K, last-token exact Q.",
        f"Selection: Page{protocol['page_size']}, strict physical "
        f"B{protocol['token_budget']} per KV group, "
        f"{protocol['pinned_prefix_pages']} pinned prefix page.",
        "",
        "## Aggregate layer-ahead prefetch",
        "",
        "| lag | overfetch | recall | precision | late pages/KV | "
        "wasted pages/KV | full coverage | target mass | non-sink mass | "
        "non-sink worst-head |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["aggregate_prefetch"]:
        lines.append(
            f"| {row['layer_lag']} | {row['overfetch_factor']:.2f}× | "
            f"{row['page_recall']:.6f} | {row['page_precision']:.6f} | "
            f"{row['late_pages_per_kv_head']:.3f} | "
            f"{row['wasted_pages_per_kv_head']:.3f} | "
            f"{row['fully_covered_kv_head_fraction']:.6f} | "
            f"{row['target_attention_mass_mean']:.6f} | "
            f"{row['target_non_sink_mass_mean']:.6f} | "
            f"{row['target_non_sink_mass_worst_head_mean']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Adjacent layers at 1× budget",
            "",
            "| source → target | post-RoPE K CKA | page recall | Jaccard | "
            "target mass | non-sink mass | non-sink worst-head |",
            "|:---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["adjacent_layers"]:
        lines.append(
            f"| {row['source_layer']} → {row['target_layer']} | "
            f"{row['post_rope_k_cka']:.6f} | {row['page_recall']:.6f} | "
            f"{row['page_jaccard']:.6f} | "
            f"{row['target_attention_mass_mean']:.6f} | "
            f"{row['target_non_sink_mass_mean']:.6f} | "
            f"{row['target_non_sink_mass_worst_head_mean']:.6f} |"
        )
    return "\n".join(lines) + "\n"


@torch.inference_mode()
def main() -> None:
    args = _parse_args()
    started = time.monotonic()
    torch.set_num_threads(args.torch_num_threads)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    capture_root = args.capture.expanduser().resolve()
    manifest_path = capture_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = manifest["artifacts"]
    layers = tuple(sorted(int(layer) for layer in artifacts))
    first_record = artifacts[str(layers[0])]
    row_shape = tuple(
        int(size) for size in first_record["routing_joint_rows"]["shape"]
    )
    documents, tokens, kv_heads, joint_dim = row_shape
    head_dim = joint_dim // 2
    pages = math.ceil(tokens / args.page_size)
    selected_pages = args.token_budget // args.page_size
    lags = _parse_csv_ints(args.layer_lags)
    overfetch_factors = _parse_csv_floats(args.overfetch_factors)
    maximum_prefetch_pages = min(
        pages,
        max(math.ceil(selected_pages * factor) for factor in overfetch_factors),
    )
    assert layers == tuple(range(len(layers)))
    assert tokens == int(manifest["calibration"]["sequence_length"])
    assert args.token_budget % args.page_size == 0
    assert 0 < args.pinned_prefix_pages < selected_pages <= pages
    assert min(lags) > 0 and min(overfetch_factors) >= 1.0
    assert maximum_prefetch_pages > args.pinned_prefix_pages
    device = torch.device(args.work_device)
    assert device.type != "cuda" or torch.cuda.is_available()

    sample_count = min(args.cka_samples, documents * tokens)
    sample_flat = torch.linspace(
        0,
        documents * tokens - 1,
        steps=sample_count,
        dtype=torch.float64,
    ).round().long()
    sample_documents = torch.div(sample_flat, tokens, rounding_mode="floor")
    sample_tokens = sample_flat.remainder(tokens)

    selections = []
    page_probabilities = []
    rankings = []
    sampled_keys = []
    for ordinal, layer in enumerate(layers, start=1):
        print(f"[Interlayer K] layer={layer} ({ordinal}/{len(layers)})", flush=True)
        record = artifacts[str(layer)]
        queries = _mmap_bfloat16(capture_root, record["routing_queries"])
        rows = _mmap_bfloat16(capture_root, record["routing_joint_rows"])
        selected, probabilities, ranked = _exact_page_state(
            queries,
            rows,
            page_size=args.page_size,
            selected_pages=selected_pages,
            pinned_prefix_pages=args.pinned_prefix_pages,
            maximum_prefetch_pages=maximum_prefetch_pages,
            device=device,
        )
        selections.append(selected)
        page_probabilities.append(probabilities)
        rankings.append(ranked)
        sampled_keys.append(
            rows[
                sample_documents,
                sample_tokens,
                :,
                head_dim:,
            ].clone()
        )
        del queries, rows, selected, probabilities, ranked

    print("[Interlayer K] computing CKA matrices", flush=True)
    cka_mean, cka_by_kv_head = _cka_matrices(sampled_keys, device=device)
    del sampled_keys

    page_overlap_matrix = []
    attention_mass_matrix = []
    non_sink_mass_matrix = []
    for left in layers:
        overlap_row = []
        mass_row = []
        non_sink_mass_row = []
        for right in layers:
            metrics = _set_metrics(
                selections[left],
                selections[right],
                page_probabilities[right],
                pinned_prefix_pages=args.pinned_prefix_pages,
            )
            overlap_row.append(metrics["page_recall"])
            mass_row.append(metrics["target_attention_mass_mean"])
            non_sink_mass_row.append(metrics["target_non_sink_mass_mean"])
        page_overlap_matrix.append(overlap_row)
        attention_mass_matrix.append(mass_row)
        non_sink_mass_matrix.append(non_sink_mass_row)

    pair_records = []
    aggregate_rows = []
    for lag in lags:
        by_factor: dict[float, list[dict[str, float]]] = {
            factor: [] for factor in overfetch_factors
        }
        for source in layers:
            target = source + lag
            if target not in layers:
                continue
            for factor in overfetch_factors:
                total_pages = min(pages, math.ceil(selected_pages * factor))
                prefetched = _prefetch_mask(
                    rankings[source],
                    pages=pages,
                    total_pages=total_pages,
                    pinned_prefix_pages=args.pinned_prefix_pages,
                )
                metrics = _set_metrics(
                    prefetched,
                    selections[target],
                    page_probabilities[target],
                    pinned_prefix_pages=args.pinned_prefix_pages,
                )
                by_factor[factor].append(metrics)
                pair_records.append(
                    {
                        "source_layer": source,
                        "target_layer": target,
                        "layer_lag": lag,
                        "overfetch_factor": factor,
                        "prefetched_pages_per_kv_head": total_pages,
                        **metrics,
                    }
                )
        for factor in overfetch_factors:
            aggregate_rows.append(
                {
                    "layer_lag": lag,
                    "overfetch_factor": factor,
                    "layer_pairs": len(by_factor[factor]),
                    **_mean_metrics(by_factor[factor]),
                }
            )

    adjacent = []
    adjacent_lookup = {
        (row["source_layer"], row["target_layer"]): row
        for row in pair_records
        if row["layer_lag"] == 1 and row["overfetch_factor"] == 1.0
    }
    for source in layers[:-1]:
        row = dict(adjacent_lookup[(source, source + 1)])
        row["post_rope_k_cka"] = cka_mean[source][source + 1]
        adjacent.append(row)

    payload = {
        "format": FORMAT,
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(sys.argv),
        "protocol": {
            "capture": str(capture_root),
            "capture_format": manifest.get("format"),
            "documents": documents,
            "sequence_length": tokens,
            "query_policy": manifest["calibration"].get("routing_query_policy"),
            "key_representation": "exact post-RoPE K",
            "query_representation": "exact post-RoPE last-token Q",
            "kv_heads": kv_heads,
            "head_dim": head_dim,
            "page_size": args.page_size,
            "token_budget": args.token_budget,
            "selected_pages_per_kv_head": selected_pages,
            "pinned_prefix_pages": args.pinned_prefix_pages,
            "selection": (
                "exact per-query page mass, max across each physical GQA group"
            ),
            "cka_samples": sample_count,
            "layer_lags": lags,
            "overfetch_factors": overfetch_factors,
        },
        "post_rope_k_linear_cka_mean": cka_mean,
        "post_rope_k_linear_cka_by_kv_head": cka_by_kv_head,
        "page_overlap_recall_matrix": page_overlap_matrix,
        "source_page_target_attention_mass_matrix": attention_mass_matrix,
        "source_page_target_non_sink_mass_matrix": non_sink_mass_matrix,
        "aggregate_prefetch": aggregate_rows,
        "pair_prefetch": pair_records,
        "adjacent_layers": adjacent,
        "elapsed_seconds": time.monotonic() - started,
        "environment": {
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_device": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
        },
    }
    output_json = args.output_json.expanduser().resolve()
    output_markdown = args.output_markdown.expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    temporary_json = output_json.with_suffix(output_json.suffix + ".tmp")
    temporary_markdown = output_markdown.with_suffix(output_markdown.suffix + ".tmp")
    temporary_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary_markdown.write_text(_markdown(payload), encoding="utf-8")
    os.replace(temporary_json, output_json)
    os.replace(temporary_markdown, output_markdown)
    print(f"[Interlayer K] result={output_json}", flush=True)


if __name__ == "__main__":
    main()
