#!/usr/bin/env python3
"""Combine the four frozen TP4 exact-K placement/routing RULER arms."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping


FORMAT = "basisserve.qwen3_8b.tp4_mapped_host_ruler.v1"


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    assert payload["format"] == FORMAT and payload["status"] == "complete"
    return payload


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _gib(value: int | float) -> float:
    return float(value) / 2**30


def _label(payload: Mapping[str, Any]) -> str:
    router = payload["protocol"]["router"]
    storage = payload["protocol"]["exact_key_storage"]
    labels = {
        ("c1_base16_r8", "gpu"): "C1 Base16+R8 GPU oracle",
        ("c1_base16_r8", "mapped_host"): "C1 Base16+R8 mapped host",
        ("quest", "mapped_host"): "QUEST mapped host",
        ("shadowkv", "mapped_host"): "ShadowKV-style mapped host",
    }
    return labels[(router, storage)]


def _markdown(payloads: list[Mapping[str, Any]]) -> str:
    first = payloads[0]
    lines = [
        "# Qwen3-8B TP4 Exact-K Offload on Five Hard RULER Samples",
        "",
        (
            "All arms use BF16 Qwen3-8B-Base, C1-V80, TP4, Page32, "
            "group-max fixed physical B4096 per KV head, the same selected-page "
            "CUDA exact-QK/online-softmax/V80 kernel, and the same five 64K YaRN4 samples."
        ),
        "",
        "| Arm | Hard-5 accuracy | Prefill tok/s | Decode tok/s | E2E model tok/s | K read/step | Total K read | GPU cache/rank | Host K/rank |",
        "|:---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for payload in payloads:
        summary = payload["summary"]
        cache = payload["cache"]
        physical_gib_per_step = float(
            summary["physical_exact_k_gib_per_decode_step"]
        )
        lines.append(
            f"| {_label(payload)} | "
            f"{100.0 * summary['task_balanced_accuracy']:.2f}% | "
            f"{summary['prefill_tokens_per_second']:.2f} | "
            f"{summary['decode_tokens_per_second']:.3f} | "
            f"{summary['end_to_end_model_tokens_per_second']:.2f} | "
            f"{physical_gib_per_step:.5f} GiB | "
            f"{summary['physical_exact_k_gib_read']:.3f} GiB | "
            f"{_gib(cache['gpu_bytes_per_rank_maximum']):.3f} GiB | "
            f"{_gib(cache['cpu_pinned_bytes_per_rank_maximum']):.3f} GiB |"
        )
    keys = [str(record["key"]) for record in first["records"]]
    lines.extend(
        [
            "",
            "| Sample | " + " | ".join(_label(payload) for payload in payloads) + " |",
            "|:---|" + "---:|" * len(payloads),
        ]
    )
    by_payload = [
        {str(record["key"]): record for record in payload["records"]}
        for payload in payloads
    ]
    for key in keys:
        scores = [100.0 * rows[key]["score"] for rows in by_payload]
        lines.append(
            f"| {key} | " + " | ".join(f"{score:.2f}%" for score in scores) + " |"
        )
    mapped_c1 = next(
        payload
        for payload in payloads
        if payload["protocol"]["router"] == "c1_base16_r8"
        and payload["protocol"]["exact_key_storage"] == "mapped_host"
    )
    agreement = mapped_c1["summary"]
    lines.extend(
        [
            "",
            "## Exact-K placement equivalence",
            "",
            (
                "C1 mapped-host versus the same-kernel GPU-resident oracle generated "
                f"identical token sequences on `{agreement['quality_oracle_sequence_matches']}/"
                f"{agreement['quality_oracle_samples']}` samples."
            ),
            "",
            "## Measurement definitions",
            "",
            "- `Group-max` first normalizes routable page scores independently for every Query head, then takes the maximum mass over the four Query heads sharing one physical KV head.",
            "- `K read/step` is the requested Page32 BF16 exact-K payload across all layers, physical KV heads, and ranks for one model decode step. `Total K read` also depends on when each generation reaches EOS. Neither value is a PCIe hardware-counter measurement.",
            "- `GPU cache/rank` includes C1-V80 plus router metadata; the GPU oracle additionally includes dense exact K.",
            "- C1 caches Base16, uses one fused CUDA historical router for Base16-to-post-RoPE QK plus R8 Page32 log-mass, and one fused CUDA selector for per-head normalization, four-head group-max, Top-K, and sorted physical page IDs.",
            "- The selected-page exact-QK/online-softmax/V80 attention is a separate fused CUDA kernel that directly reads CUDA-mapped host Keys.",
            "- `ShadowKV-style` isolates post-RoPE Page32 mean-landmark routing. It does not include ShadowKV's chunk8 outlier/local caches or online-SVD Key payload.",
            "- End-to-end model throughput counts prompt tokens and decode forward steps and excludes model loading.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-oracle", type=Path, required=True)
    parser.add_argument("--c1-mapped", type=Path, required=True)
    parser.add_argument("--quest-mapped", type=Path, required=True)
    parser.add_argument("--shadowkv-mapped", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payloads = [
        _load(args.gpu_oracle),
        _load(args.c1_mapped),
        _load(args.quest_mapped),
        _load(args.shadowkv_mapped),
    ]
    reference = payloads[0]["protocol"]
    comparable = (
        "model",
        "tp_size",
        "dtype",
        "sequence_length",
        "tasks",
        "sample_spec",
        "page_size",
        "physical_token_budget_per_kv_head",
        "pinned_prefix_pages",
        "force_current_page",
        "routing_aggregation",
        "value_cache",
        "exact_attention",
        "c1_collective",
        "rope_scaling",
    )
    assert all(
        all(payload["protocol"][name] == reference[name] for name in comparable)
        for payload in payloads[1:]
    )
    _atomic_text(args.output.expanduser().resolve(), _markdown(payloads))


if __name__ == "__main__":
    main()
