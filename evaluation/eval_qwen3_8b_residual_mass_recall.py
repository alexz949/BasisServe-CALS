#!/usr/bin/env python3
"""Frozen residual schedule versus uniform R8 on shared teacher queries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import sys
import time
from unittest.mock import patch

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basisserve.checkpoint.gqa_vo_qwen3 import install_qwen3_gqa_vo_als_export
from basisserve.core.c1_conditional_page_attention import _selected_pages
from basisserve.core.c1_k_routing_sidecar import RoutingDynamicCache
from basisserve.core.residual_kl_replay import fork_routing_prefix, prefix_signature, terminal_metrics
from basisserve.core.residual_mass_recall import (
    capture_teacher_queries, rank_sidecar, routing_mass_recall, teacher_page_probabilities,
)
from evaluation.fit_qwen3_8b_residual_kl_bank import sha256, write_json
from evaluation.profile_qwen3_8b_residual_two_sided_kl import (
    full_attention, install_rank, load_bank, make_sidecar, protocol as kl_protocol,
)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("smoke", "evaluate", "summarize"), required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--c1-checkpoint", type=Path, required=True)
    p.add_argument("--bank", type=Path, default=ROOT / "results/checkpoints/q8_residual_kl_bank")
    p.add_argument("--windows", type=Path, default=ROOT / "results/calibration/qwen3_8b_c4_64f16h_s32768/windows.safetensors")
    p.add_argument("--kl-root", type=Path, default=ROOT / "results/evaluation/q8_residual_kl_64x32k")
    p.add_argument("--output-dir", type=Path, default=ROOT / "results/evaluation/q8_residual_mass_16x32k")
    p.add_argument("--suffix-length", type=int, default=128)
    p.add_argument("--query-block-size", type=int, default=8)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=4)
    p.add_argument("--torch-num-threads", type=int, default=2)
    return p


def protocol(args):
    allocation = json.loads((args.kl_root / "schedule.json").read_text())
    source = kl_protocol(args)
    assert allocation["status"] == "complete" and allocation["protocol"] == source
    ranks = allocation["layer_ranks"]
    assert len(ranks) == 36 and sum(ranks) == 288 and set(ranks) <= {4, 8, 16}
    return {
        "format": "basisserve.residual_mass_recall.v1", "kl_protocol": source,
        "schedule_sha256": sha256(args.kl_root / "schedule.json"), "layer_ranks": ranks,
        "code_sha256": {name: sha256(ROOT / name) for name in (
            "basisserve/core/residual_mass_recall.py",
            "evaluation/eval_qwen3_8b_residual_mass_recall.py",
        )},
        "indices": list(range(64, 80)), "gpu": "NVIDIA L40S",
        "query_source": "same full-attention C1 teacher; no sparse feedback into hidden states",
        "teacher_mass": "FP32 exact QK of cached BF16 Q/K, causal full-support softmax",
        "non_sink_mass": "conditional softmax outside pinned page0; excludes unsupported rows",
        "selection": "unchanged native BF16 proxy, non-sink per-head Page-LSE normalization, GQA max, 64 pages including page0",
        "sidecar": "true post-RoPE residual, prefix and suffix built separately",
        "quantiles": "pooled query/head observations, not independent-document confidence intervals",
        "scope": "post-allocation diagnostics only; no refitting, no schedule selection, not full PPL or RULER",
    }


def describe(values, valid=None):
    values = values.double()
    if valid is not None:
        values = values[valid]
    values = values.flatten()
    assert bool(torch.isfinite(values).all())
    if not values.numel():
        return {"count": 0, "mean": None, "p01": None, "p10": None, "median": None, "min": None}
    quantiles = torch.quantile(values, torch.tensor([.01, .1, .5], dtype=torch.float64))
    return {"count": values.numel(), "mean": float(values.mean()), "p01": float(quantiles[0]),
            "p10": float(quantiles[1]), "median": float(quantiles[2]), "min": float(values.min())}


def describe_pair(raw):
    valid = raw["non_sink_valid"]
    result = {"sink_mass": describe(raw["sink_mass"]), "non_sink_unsupported": int((~valid).sum())}
    for arm in ("uniform", "adaptive"):
        result[arm] = {"mass": describe(raw[f"{arm}_mass"]),
                       "non_sink": describe(raw[f"{arm}_non_sink"], valid)}
    result["delta_mass"] = describe(raw["adaptive_mass"].double() - raw["uniform_mass"].double())
    result["delta_non_sink"] = describe(raw["adaptive_non_sink"].double() - raw["uniform_non_sink"].double(), valid)
    return result


@torch.inference_mode()
def verify_native_selection(module, tensors, rank, prefix, prefix_rotary, record, sidecar, observed, block_size):
    """Compare against a real attention-module forward, including cache appends."""
    install_rank(module, tensors, rank, block_size)
    cache = fork_routing_prefix(prefix)
    cache._ensure_routing_layer(module.layer_idx)
    cache._routing_sidecars[module.layer_idx] = make_sidecar(module, prefix, *prefix_rotary)
    captured = []

    def observe(*args, **kwargs):
        ids, valid = _selected_pages(*args, **kwargs)
        captured.append((ids.detach(), valid.detach()))
        return ids, valid

    with patch("basisserve.core.c1_conditional_page_attention._selected_pages", side_effect=observe):
        module(**dict(record["context"], past_key_values=cache))
    torch.testing.assert_close(cache.routing_sidecar(module.layer_idx), sidecar, atol=0, rtol=0)
    torch.testing.assert_close(torch.cat([x[0] for x in captured], dim=2), observed["page_ids"], atol=0, rtol=0)
    torch.testing.assert_close(torch.cat([x[1] for x in captured], dim=2), observed["page_valid"], atol=0, rtol=0)


@torch.inference_mode()
def evaluate_window(model, bank, tokens, args, settings, index):
    started = time.monotonic()
    modules = [layer.self_attn for layer in model.model.layers]
    prefix_length = tokens.shape[1] - args.suffix_length
    suffix = tokens[:, prefix_length:]
    full_attention(modules, "triton")
    prefix = RoutingDynamicCache()
    model.model(input_ids=tokens[:, :prefix_length], past_key_values=prefix, use_cache=True)
    torch.cuda.synchronize()
    prefill_seconds = time.monotonic() - started
    signature = prefix_signature(prefix)
    full_attention(modules, "sdpa")
    cache = fork_routing_prefix(prefix)
    hidden, records = capture_teacher_queries(model, suffix, cache)
    if args.stage == "smoke":
        reference = model.model(input_ids=suffix, past_key_values=fork_routing_prefix(prefix), use_cache=True).last_hidden_state
        torch.testing.assert_close(hidden, reference, atol=0, rtol=0)
        del reference
    logits = model.lm_head(hidden)
    teacher_metrics = terminal_metrics(logits, logits.float().log_softmax(dim=-1), suffix)
    del hidden, logits
    previous = json.loads((args.kl_root / "confirm" / f"window_{index:03d}.json").read_text())
    assert previous["protocol"] == settings["kl_protocol"]
    assert previous["gpu"] == settings["gpu"] and previous["schedule_sha256"] == settings["schedule_sha256"]
    nll_delta = teacher_metrics["nll_mean"] - previous["result"]["teacher"]["nll_mean"]
    assert abs(nll_delta) <= 1e-7
    prefix_positions = torch.arange(prefix_length, device=tokens.device)[None]
    prefix_rotary = model.model.rotary_emb(model.model.embed_tokens.weight[:1], prefix_positions)
    raw = {name: [] for name in (
        "uniform_mass", "adaptive_mass", "uniform_non_sink", "adaptive_non_sink", "sink_mass", "non_sink_valid",
    )}
    smoke_checks = []
    for layer, (module, tensors, record, rank) in enumerate(zip(modules, bank, records, settings["layer_ranks"], strict=True)):
        query = record["query"]
        shared = dict(page_size=32, pinned_prefix_pages=1, scale=module.scaling,
                      query_block_size=args.query_block_size, attention_mask=record["context"]["attention_mask"])
        teacher = teacher_page_probabilities(query, cache.layers[layer].keys, **shared)
        arms = {}
        for arm, arm_rank in (("uniform", 8), ("adaptive", rank)):
            if arm == "adaptive" and arm_rank == 8:
                arms[arm] = arms["uniform"]
                continue
            sidecar, projector = rank_sidecar(tensors, arm_rank, prefix.layers[layer], cache.layers[layer],
                                               prefix_rotary, record["context"]["position_embeddings"])
            check = args.stage == "smoke" and layer in (0, 1, 33)
            arms[arm] = routing_mass_recall(query, sidecar, projector, teacher, exact_token_budget=2048,
                                          return_pages=check, **shared)
            if check:
                verify_native_selection(module, tensors, arm_rank, prefix, prefix_rotary, record,
                                        sidecar, arms[arm], args.query_block_size)
                smoke_checks.append({"layer": layer, "rank": arm_rank, "native_pages_and_sidecar_exact": True})
            del sidecar, projector
        for arm in ("uniform", "adaptive"):
            raw[f"{arm}_mass"].append(arms[arm]["mass"][0].cpu())
            raw[f"{arm}_non_sink"].append(arms[arm]["non_sink_mass"][0].cpu())
        raw["sink_mass"].append(teacher["mass"][0, ..., :1].sum(dim=-1).cpu())
        raw["non_sink_valid"].append(teacher["non_sink_valid"][0].cpu())
        print(f"[mass window={index}] layer={layer:02d} R8->{rank} mass={float(arms['uniform']['mass'].mean()):.6f}->{float(arms['adaptive']['mass'].mean()):.6f}", flush=True)
        del teacher, arms
    raw = {name: torch.stack(parts).contiguous() for name, parts in raw.items()}
    assert prefix_signature(prefix) == signature
    for layer, rank in enumerate(settings["layer_ranks"]):
        if rank == 8:
            for name in ("mass", "non_sink"):
                torch.testing.assert_close(raw[f"uniform_{name}"][layer], raw[f"adaptive_{name}"][layer], atol=0, rtol=0)
    summary = describe_pair(raw)
    summary.update(index=index, prefix_unchanged=True, teacher_nll_delta_vs_kl_run=nll_delta,
                   prefill_seconds=prefill_seconds, wall_seconds=time.monotonic() - started,
                   maximum_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                   native_smoke_checks=smoke_checks)
    return raw, summary


def summarize(args, settings):
    raw_rows, rows = [], []
    for index in settings["indices"]:
        path = args.output_dir / "evaluate" / f"window_{index:03d}.json"
        payload = json.loads(path.read_text())
        assert payload["status"] == "complete" and payload["protocol"] == settings
        assert payload["gpu"] == settings["gpu"] and payload["result"]["index"] == index
        raw_path = path.with_suffix(".safetensors")
        assert sha256(raw_path) == payload["raw_sha256"]
        raw = load_file(str(raw_path))
        assert all(tuple(x.shape) == (36, 32, 128) for x in raw.values())
        raw_rows.append(raw)
        rows.append(payload["result"])
    pooled = {name: torch.stack([row[name] for row in raw_rows]) for name in raw_rows[0]}
    overall = describe_pair(pooled)
    by_layer = [dict(layer=layer, rank=rank, **describe_pair({name: value[:, layer] for name, value in pooled.items()}))
                for layer, rank in enumerate(settings["layer_ranks"])]
    paired = {}
    for name in ("mass", "non_sink"):
        deltas = torch.tensor([row[f"delta_{name}"]["mean"] for row in rows], dtype=torch.float64)
        paired[name] = {"mean": float(deltas.mean()), "median": float(deltas.quantile(.5)),
                        "standard_error": float(deltas.std(unbiased=True) / len(rows)**.5),
                        "improved_windows": int((deltas > 0).sum()), "windows": len(rows)}
    by_rank = {str(rank): describe_pair({name: value[:, torch.tensor(settings["layer_ranks"]) == rank]
                                       for name, value in pooled.items()}) for rank in (4, 8, 16)}
    result = {"status": "complete", "protocol": settings, "overall": overall, "by_layer": by_layer,
              "by_rank": by_rank, "by_window": rows, "paired_window_delta": paired,
              "command": shlex.join(sys.argv), "python": sys.executable, "torch": torch.__version__}
    write_json(args.output_dir / "result.json", result)
    lines = ["# Qwen3-8B residual attention-mass recall", "",
             "Frozen C1-V80 + Base16; Page32, B2048 including page0; uniform R8 versus the frozen average-R8 schedule.",
             "C4 confirmation indices 64–79: 16 packed 32768-token windows, final 128 queries, all 36 layers and 32 query heads.",
             "Shared full-attention C1 teacher Q/K/latent inputs; sparse outputs do not feed back into this diagnostic.", "",
             "Mass probabilities use FP32 exact QK from BF16 cached activations. Routing uses the unchanged native BF16 selector.",
             "Non-sink recall conditions on tokens outside page0. Unsupported rows are excluded, not treated as zero.",
             "P01/P10/median pool query-head observations; they are not independent-window confidence intervals.", "",
             "| Metric | Uniform mean | Adaptive mean | Uniform P01 | Adaptive P01 | Uniform P10 | Adaptive P10 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name in ("mass", "non_sink"):
        u, a = overall["uniform"][name], overall["adaptive"][name]
        lines.append(f"| {name} | {u['mean']:.8f} | {a['mean']:.8f} | {u['p01']:.8f} | {a['p01']:.8f} | {u['p10']:.8f} | {a['p10']:.8f} |")
    lines += ["", "## Per-layer means", "", "| Layer | Adaptive rank | Uniform mass | Adaptive mass | Uniform non-sink | Adaptive non-sink |",
              "|---|---:|---:|---:|---:|---:|"]
    for row in by_layer:
        u, a = row["uniform"], row["adaptive"]
        lines.append(f"| {row['layer']} | {row['rank']} | {u['mass']['mean']:.8f} | {a['mass']['mean']:.8f} | {u['non_sink']['mean']:.8f} | {a['non_sink']['mean']:.8f} |")
    lines += ["", "## Paired window means", "", "Deltas are adaptive minus uniform; positive means higher retained mass.", "",
              "| Window | Delta mass | Delta non-sink |", "|---|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['index']} | {row['delta_mass']['mean']:+.8f} | {row['delta_non_sink']['mean']:+.8f} |")
    lines += ["", "This is post-allocation analysis on previously inspected confirmation windows, not a new untouched test.",
              "It does not measure full-sequence sparse trajectories, full PPL, RULER, offload memory or latency.",
              "Environment: basis; formal GPU: NVIDIA L40S. Per-window JSON records commands, source hashes, raw-data hashes and timings.",
              "Protocol and commands: [protocol](../../../docs/q8_residual_mass_recall_protocol.md).", ""]
    (args.output_dir / "summary.md").write_text("\n".join(lines))
    print(json.dumps({"overall": overall, "paired_window_delta": paired}, indent=2), flush=True)


@torch.inference_mode()
def main():
    args = parser().parse_args()
    assert args.suffix_length == 128 and args.query_block_size == 8
    assert 0 <= args.shard_index < args.num_shards
    torch.set_num_threads(args.torch_num_threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    settings = protocol(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == "summarize":
        summarize(args, settings)
        return
    assert torch.cuda.is_available() and torch.cuda.get_device_name(0) == settings["gpu"]
    torch.cuda.set_device(0)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, local_files_only=True,
                                               low_cpu_mem_usage=True, attn_implementation="sdpa").to("cuda:0").eval()
    install_qwen3_gqa_vo_als_export(model, args.c1_checkpoint, attention_backend="triton")
    bank = load_bank(args.bank)
    windows = load_file(str(args.windows))["input_ids"]
    assert tuple(windows.shape) == (80, 32768)
    indices = [64] if args.stage == "smoke" else settings["indices"][args.shard_index::args.num_shards]
    for index in indices:
        output = args.output_dir / args.stage / f"window_{index:03d}.json"
        if output.exists():
            saved = json.loads(output.read_text())
            assert saved["status"] == "complete" and saved["protocol"] == settings and saved["gpu"] == settings["gpu"]
            assert sha256(output.with_suffix(".safetensors")) == saved["raw_sha256"]
            print(f"[resume] window={index} already complete", flush=True)
            continue
        torch.cuda.reset_peak_memory_stats()
        raw, result = evaluate_window(model, bank, windows[index:index+1].long().to("cuda:0"), args, settings, index)
        output.parent.mkdir(parents=True, exist_ok=True)
        raw_path = output.with_suffix(".safetensors")
        temporary = raw_path.with_suffix(".tmp")
        save_file(raw, str(temporary))
        temporary.replace(raw_path)
        write_json(output, {"status": "complete", "protocol": settings, "result": result,
                            "raw_sha256": sha256(raw_path), "command": shlex.join(sys.argv),
                            "python": sys.executable, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)})
        print(f"[complete] window={index} seconds={result['wall_seconds']:.2f} peak_GiB={result['maximum_allocated_gib']:.2f}", flush=True)
    write_json(args.output_dir / args.stage / f"shard_{args.shard_index}.json",
               {"status": "complete", "protocol": settings, "indices": indices, "command": shlex.join(sys.argv)})


if __name__ == "__main__":
    main()
