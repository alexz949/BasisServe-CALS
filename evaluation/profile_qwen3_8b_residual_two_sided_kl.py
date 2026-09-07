#!/usr/bin/env python3
"""Three-point residual-rank KL allocation after a shared dense-C1 prefix."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import shlex
import sys
import time

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basisserve.checkpoint.gqa_vo_qwen3 import install_qwen3_gqa_vo_als_export
from basisserve.core.c1_k_reverse_shadow import ReverseShadowConfig
from basisserve.core.c1_k_routing_sidecar import RoutingDynamicCache
from basisserve.core.c1_v_conditional_k_router import build_conditional_routing_sidecar
from basisserve.core.residual_kl_replay import (
    capture_cached_suffix, fork_routing_prefix, prefix_signature,
    replay_cached_suffix, terminal_metrics,
)
from evaluation.build_qwen3_8b_c1_two_sided_factorized_kl_schedule import allocate_layer_schedule
from evaluation.fit_qwen3_8b_residual_kl_bank import sha256, write_json

FORMAT = "basisserve.residual_two_sided_kl.v1"


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("smoke", "profile", "allocate", "confirm", "summarize"), required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--c1-checkpoint", type=Path, required=True)
    p.add_argument("--bank", type=Path, default=ROOT / "results/checkpoints/q8_residual_kl_bank")
    p.add_argument("--windows", type=Path, default=ROOT / "results/calibration/qwen3_8b_c4_64f16h_s32768/windows.safetensors")
    p.add_argument("--output-dir", type=Path, default=ROOT / "results/evaluation/q8_residual_kl_64x32k")
    p.add_argument("--suffix-length", type=int, default=128)
    p.add_argument("--query-block-size", type=int, default=8)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=4)
    p.add_argument("--torch-num-threads", type=int, default=2)
    return p


def protocol(args) -> dict:
    bank_sha = {}
    bank_protocol = None
    for layer in range(36):
        path = args.bank / f"layer_{layer:03d}.safetensors"
        record = json.loads(path.with_suffix(".json").read_text())
        assert record["status"] == "complete"
        bank_sha[str(layer)] = sha256(path)
        assert bank_sha[str(layer)] == record["sha256"]
        assert record["protocol"]["c1_manifest_sha256"] == sha256(args.c1_checkpoint / "results.json")
        assert record["protocol"]["windows_sha256"] == sha256(args.windows)
        if layer == 0:
            bank_protocol = record["protocol"]
        assert record["protocol"] == bank_protocol
    return {
        "format": FORMAT, "model_config_sha256": sha256(args.model / "config.json"),
        "c1_manifest_sha256": sha256(args.c1_checkpoint / "results.json"),
        "bank_sha256": bank_sha, "windows_sha256": sha256(args.windows),
        "factor_bank_protocol": bank_protocol,
        "code_sha256": {name: sha256(ROOT / name) for name in (
            "evaluation/profile_qwen3_8b_residual_two_sided_kl.py",
            "basisserve/core/residual_kl_replay.py",
            "basisserve/core/c1_conditional_page_attention.py",
            "basisserve/core/c1_v_conditional_k_router.py",
            "basisserve/checkpoint/gqa_vo_qwen3.py",
        )},
        "profile_indices": list(range(64)), "confirmation_indices": list(range(64, 80)),
        "sequence_length": 32768, "suffix_length": args.suffix_length,
        "prefix_length": 32768 - args.suffix_length,
        "window_type": "eight 4096-token C4 windows packed without inserted separators",
        "teacher": "same C1-V80, full exact-K, shared dense-C1 prefix",
        "payload_rank": 80, "base_rank": 16, "anchor_rank": 8, "candidate_ranks": [4, 8, 16],
        "rank_granularity": "one rank per layer, shared by all 8 KV groups",
        "cost": "signed measured terminal KL delta; no clipping or factorized interpolation",
        "total_layer_rank": 288, "page_size": 32, "physical_token_budget": 2048,
        "pinned_prefix_pages": 1, "query_block_size": args.query_block_size,
        "suffix_execution": "causal block teacher forcing; exact layer-suffix replay",
        "kl_readout": "all suffix positions, full vocabulary, FP32 log-softmax, FP64 reduction",
        "nll_readout": "suffix positions except final position (next target unavailable)",
        "factor_fit_overlap": "profile windows reused from residual fit; not held-out",
        "confirmation_scope": (
            "not used for terminal allocation; reused factor-development windows, not untouched; "
            "see factor_bank_protocol for factor fitting and selection use"
        ),
        "oracle_storage": "GPU exact K and materialized Base128+R sidecar; not a latency/offload benchmark",
    }


def load_bank(root: Path) -> list[dict]:
    return [load_file(str(root / f"layer_{layer:03d}.safetensors")) for layer in range(36)]


def install_rank(module, tensors: dict, rank: int, query_block_size: int) -> None:
    module.attention_backend = "native"
    module.set_reverse_shadow_config(None)
    module.set_conditional_routing_factors(
        base_left=tensors["base_left_b16"], base_right=tensors["base_right_b16"],
        base_bias=tensors["base_bias_b16"],
        residual_encoder=tensors[f"residual_encoder_b16_r{rank}"],
        residual_query_projector=tensors[f"residual_query_b16_r{rank}"],
    )
    module.set_conditional_page_query_block_size(query_block_size, collect_statistics=True)
    module.set_reverse_shadow_config(ReverseShadowConfig(
        page_size=32, exact_token_budget=2048, selector="kq_svd",
        quest_support="physical_shared", pinned_prefix_pages=1,
    ))


def full_attention(modules: list, backend: str) -> None:
    for module in modules:
        module.set_reverse_shadow_config(None)
        module.conditional_base_left = None
        module.conditional_base_right = None
        module.conditional_base_bias = None
        module.conditional_residual_encoder = None
        module.routing_query_projector = None
        module.attention_backend = backend


def make_sidecar(module, prefix, cos, sin):
    cache = prefix.layers[module.layer_idx]
    return build_conditional_routing_sidecar(
        cache.values, cache.keys, base_left=module.conditional_base_left,
        base_right=module.conditional_base_right, base_bias=module.conditional_base_bias,
        residual_encoder=module.conditional_residual_encoder, cos=cos, sin=sin,
    )


def routing_statistics(modules: list) -> dict:
    rows = [module.reverse_shadow_statistics() for module in modules]
    valid = sum(row["physical_valid_tokens"] for row in rows)
    selected = sum(row["selected_tokens"] for row in rows)
    assert valid > 0 and 0 < selected < valid
    return {"physical_valid_tokens": valid, "selected_tokens": selected,
            "selected_fraction": selected / valid}


@torch.inference_mode()
def evaluate_window(model, bank, tokens, args, *, index: int, schedule: list[int] | None) -> dict:
    modules = [layer.self_attn for layer in model.model.layers]
    prefix_length = int(tokens.shape[1]) - args.suffix_length
    suffix = tokens[:, prefix_length:]
    started = time.monotonic()
    full_attention(modules, "triton")
    prefix = RoutingDynamicCache()
    # C1 payload is already installed. Dense means full attention, not BF16 V.
    model.model(input_ids=tokens[:, :prefix_length], past_key_values=prefix, use_cache=True)
    torch.cuda.synchronize()
    prefill_seconds = time.monotonic() - started
    print(f"[window {index}] dense-C1 prefix ready seconds={prefill_seconds:.2f}", flush=True)
    assert all(prefix.get_seq_length(i) == prefix_length for i in range(36))
    original_signature = prefix_signature(prefix)
    full_attention(modules, "sdpa")
    teacher_cache = fork_routing_prefix(prefix)
    teacher_hidden = model.model(input_ids=suffix, past_key_values=teacher_cache, use_cache=True).last_hidden_state
    teacher_logits = model.lm_head(teacher_hidden)
    teacher_logp = teacher_logits.float().log_softmax(dim=-1)
    teacher_metrics = terminal_metrics(teacher_logits, teacher_logp, suffix)
    del teacher_cache, teacher_logits, teacher_hidden
    assert prefix_signature(prefix) == original_signature
    positions = torch.arange(prefix_length, device=tokens.device)[None]
    cos, sin = model.model.rotary_emb(model.model.embed_tokens.weight[:1], positions)
    for layer, module in enumerate(modules):
        install_rank(module, bank[layer], 8, args.query_block_size)
        prefix._ensure_routing_layer(layer)
        prefix._routing_sidecars[layer] = make_sidecar(module, prefix, cos, sin)
    original_signature = prefix_signature(prefix)
    anchor_cache = fork_routing_prefix(prefix)
    anchor = capture_cached_suffix(model, suffix, anchor_cache)
    anchor_metrics = terminal_metrics(model.lm_head(anchor.final_hidden), teacher_logp, suffix)
    anchor_metrics["routing"] = routing_statistics(modules)
    del anchor_cache
    assert prefix_signature(prefix) == original_signature
    print(f"[window {index}] R8 KL={anchor_metrics['kl_mean']:.8f}", flush=True)
    result = {"index": index, "teacher": teacher_metrics, "anchor": anchor_metrics,
              "probes": [], "prefill_seconds": prefill_seconds}

    if args.stage == "smoke":
        replay = replay_cached_suffix(model, anchor, fork_routing_prefix(prefix), intervention_layer=33)
        torch.testing.assert_close(replay, anchor.final_hidden, atol=0, rtol=0)
        result["anchor_replay_max_abs"] = float((replay - anchor.final_hidden).abs().max())
        del replay

    if schedule is None:
        probe_layers = (0, 33) if args.stage == "smoke" else range(36)
        for layer in probe_layers:
            for rank in (4, 16):
                probe_started = time.monotonic()
                for module in modules:
                    module.reset_reverse_shadow_statistics()
                install_rank(modules[layer], bank[layer], rank, args.query_block_size)
                cache = fork_routing_prefix(prefix)
                cache._routing_sidecars[layer] = make_sidecar(modules[layer], prefix, cos, sin)
                replay = replay_cached_suffix(model, anchor, cache, intervention_layer=layer)
                metrics = terminal_metrics(model.lm_head(replay), teacher_logp, suffix)
                metrics.update(layer=layer, rank=rank, delta_kl=metrics["kl_mean"] - anchor_metrics["kl_mean"])
                metrics["routing"] = routing_statistics(modules)
                if args.stage == "smoke" and layer == 33 and rank == 4:
                    reference_cache = fork_routing_prefix(prefix)
                    reference_cache._routing_sidecars[layer] = make_sidecar(modules[layer], prefix, cos, sin)
                    reference = model.model(input_ids=suffix, past_key_values=reference_cache, use_cache=True).last_hidden_state
                    torch.testing.assert_close(replay, reference, atol=0, rtol=0)
                    result["candidate_replay_max_abs"] = float((replay - reference).abs().max())
                    del reference, reference_cache
                del cache, replay
                assert prefix_signature(prefix) == original_signature
                install_rank(modules[layer], bank[layer], 8, args.query_block_size)
                metrics["wall_seconds"] = time.monotonic() - probe_started
                result["probes"].append(metrics)
                print(f"[window {index}] layer={layer:02d} R{rank} delta={metrics['delta_kl']:+.8f} seconds={metrics['wall_seconds']:.2f}", flush=True)
    else:
        assert len(schedule) == 36 and sum(schedule) == 288 and set(schedule) <= {4, 8, 16}
        cache = fork_routing_prefix(prefix)
        for layer, (module, rank) in enumerate(zip(modules, schedule, strict=True)):
            install_rank(module, bank[layer], rank, args.query_block_size)
            if rank != 8:
                cache._routing_sidecars[layer] = make_sidecar(module, prefix, cos, sin)
        hidden = model.model(input_ids=suffix, past_key_values=cache, use_cache=True).last_hidden_state
        result["adaptive"] = terminal_metrics(model.lm_head(hidden), teacher_logp, suffix)
        result["adaptive"]["routing"] = routing_statistics(modules)
        result["adaptive"]["delta_kl"] = result["adaptive"]["kl_mean"] - anchor_metrics["kl_mean"]
        del cache, hidden
        assert prefix_signature(prefix) == original_signature
    result["prefix_unchanged"] = True
    result["wall_seconds"] = time.monotonic() - started
    result["maximum_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
    return result


def load_rows(root: Path, stage: str, indices: range, settings: dict) -> list[dict]:
    rows = []
    for index in indices:
        row = json.loads((root / stage / f"window_{index:03d}.json").read_text())
        assert row["status"] == "complete" and row["protocol"] == settings
        assert row["result"]["index"] == index
        rows.append(row["result"])
    return rows


def average_metrics(rows: list[dict], arm: str) -> dict:
    kl_positions = sum(row[arm]["kl_positions"] for row in rows)
    nll_positions = sum(row[arm]["nll_positions"] for row in rows)
    kl = sum(row[arm]["kl_sum"] for row in rows) / kl_positions
    nll = sum(row[arm]["nll_sum"] for row in rows) / nll_positions
    return {"kl_mean": kl, "suffix_nll": nll, "suffix_ppl": math.exp(nll),
            "kl_positions": kl_positions, "nll_positions": nll_positions}


def allocate(args, settings) -> None:
    rows = load_rows(args.output_dir, "profile", range(64), settings)
    costs = [{8: 0.0} for _ in range(36)]
    for layer in range(36):
        for rank in (4, 16):
            deltas = []
            for row in rows:
                candidates = [p for p in row["probes"] if p["layer"] == layer and p["rank"] == rank]
                assert len(candidates) == 1
                deltas.append(candidates[0]["delta_kl"])
            costs[layer][rank] = sum(deltas) / len(deltas)
    schedule, predicted = allocate_layer_schedule(
        costs, candidate_ranks=(4, 8, 16), anchor_rank=8, target_average_rank=8,
    )
    assert sum(schedule) == 288
    write_json(args.output_dir / "schedule.json", {
        "status": "complete", "protocol": settings, "layer_ranks": list(schedule),
        "rank_counts": dict(Counter(schedule)), "measured_costs": costs,
        "predicted_additive_delta_kl": predicted,
        "profile_anchor": average_metrics(rows, "anchor"),
        "profile_teacher": average_metrics(rows, "teacher"),
    })
    print(f"[allocate] ranks={list(schedule)} predicted_delta={predicted:.8f}", flush=True)


def summarize(args, settings) -> None:
    schedule = json.loads((args.output_dir / "schedule.json").read_text())
    assert schedule["protocol"] == settings
    rows = load_rows(args.output_dir, "confirm", range(64, 80), settings)
    metrics = {arm: average_metrics(rows, arm) for arm in ("teacher", "anchor", "adaptive")}
    deltas = [row["adaptive"]["kl_mean"] - row["anchor"]["kl_mean"] for row in rows]
    mean_delta = sum(deltas) / len(deltas)
    se = math.sqrt(sum((x - mean_delta)**2 for x in deltas) / (len(deltas) - 1) / len(deltas))
    payload = {"status": "complete", "protocol": settings, "schedule": schedule,
               "confirmation": metrics, "paired_kl_delta_mean": mean_delta,
               "paired_window_kl_delta_standard_error": se,
               "confirmation_schedule_sha256": sha256(args.output_dir / "schedule.json")}
    write_json(args.output_dir / "result.json", payload)
    lines = ["# Qwen3-8B residual two-sided KL", "", "C1-V80 and Base16 frozen. C4 64×32768 profile, 16×32768 confirmation.",
             "Windows are packed from eight 4096-token C4 samples, not native 32K documents.", "",
             "Each window shares a full-attention C1 prefix across all arms; only the final 128 positions use the evaluated routing schedule.",
             "Teacher: same C1-V80 with full exact-K. Page32, B2048, page0 pinned, per-layer R4/R8/R16, average R8.", "",
             "| Arm | Confirmation teacher KL | Suffix NLL | Suffix PPL |",
             "|---|---:|---:|---:|"]
    for arm, metric in metrics.items():
        lines.append(f"| {arm} | {metric['kl_mean']:.8f} | {metric['suffix_nll']:.8f} | {metric['suffix_ppl']:.8f} |")
    lines += ["", f"Layer ranks: `{schedule['layer_ranks']}`", "",
              f"Predicted additive profile ΔKL: {schedule['predicted_additive_delta_kl']:.8f}.",
              f"Measured confirmation ΔKL: {mean_delta:+.8f}; paired-window standard error: {se:.8f}.", "",
              "The three ranks use signed measured costs directly; no local-error interpolation or negative-slope clipping.",
              "Profile windows overlap residual fitting. Confirmation windows were used for prior factor diagnostics, but not terminal allocation.",
              "KL uses all 128 suffix positions; NLL uses 127 positions with known next-token targets. This is not full-corpus PPL or RULER accuracy.",
              "GPU-resident exact K and materialized Base128+R sidecars are accuracy-oracle storage, not deployable memory or latency measurements.", ""]
    (args.output_dir / "summary.md").write_text("\n".join(lines))


@torch.inference_mode()
def main() -> None:
    args = parser().parse_args()
    assert 0 <= args.shard_index < args.num_shards
    assert args.suffix_length == 128 and args.query_block_size > 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.torch_num_threads)
    settings = protocol(args)
    if args.stage == "allocate":
        allocate(args, settings)
        return
    if args.stage == "summarize":
        summarize(args, settings)
        return
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    assert torch.cuda.is_available()
    torch.cuda.set_device(0)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, local_files_only=True,
        low_cpu_mem_usage=True, attn_implementation="sdpa",
    ).to("cuda:0").eval()
    install_qwen3_gqa_vo_als_export(model, args.c1_checkpoint, attention_backend="triton")
    assert len(model.model.layers) == 36
    bank = load_bank(args.bank)
    windows = load_file(str(args.windows))["input_ids"]
    assert tuple(windows.shape) == (80, 32768)
    schedule = None
    schedule_hash = None
    if args.stage == "confirm":
        path = args.output_dir / "schedule.json"
        allocation = json.loads(path.read_text())
        assert allocation["protocol"] == settings
        schedule = allocation["layer_ranks"]
        schedule_hash = sha256(path)
    indices = [0] if args.stage == "smoke" else list(range(64) if args.stage == "profile" else range(64, 80))
    if args.stage != "smoke":
        indices = indices[args.shard_index::args.num_shards]
    for index in indices:
        output = args.output_dir / args.stage / f"window_{index:03d}.json"
        if output.exists():
            saved = json.loads(output.read_text())
            assert saved["protocol"] == settings and saved["status"] == "complete"
            assert saved["schedule_sha256"] == schedule_hash
            print(f"[resume] {args.stage} window={index} already complete", flush=True)
            continue
        torch.cuda.reset_peak_memory_stats()
        result = evaluate_window(model, bank, windows[index:index+1].long().to("cuda:0"), args,
                                 index=index, schedule=schedule)
        write_json(output, {
            "status": "complete", "protocol": settings, "result": result,
            "schedule_sha256": schedule_hash, "command": shlex.join(sys.argv),
            "python": sys.executable, "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0),
        })
        print(f"[complete] {args.stage} window={index} seconds={result['wall_seconds']:.2f} peak_GiB={result['maximum_allocated_gib']:.2f}", flush=True)
    write_json(args.output_dir / args.stage / f"shard_{args.shard_index}.json", {
        "status": "complete", "protocol": settings, "indices": indices,
        "command": shlex.join(sys.argv),
    })


if __name__ == "__main__":
    main()
