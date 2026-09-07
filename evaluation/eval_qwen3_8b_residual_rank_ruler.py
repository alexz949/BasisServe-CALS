#!/usr/bin/env python3
"""Paired RULER pilot: frozen Base16 + uniform Fisher R8 vs exact K."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import sys
import time
from unittest.mock import patch

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basisserve.checkpoint.gqa_vo_qwen3 import install_qwen3_gqa_vo_als_export
from basisserve.core.c1_conditional_page_attention import _selected_pages
from basisserve.core.c1_k_routing_sidecar import RoutingDynamicCache
from basisserve.core.residual_kl_replay import fork_routing_prefix, prefix_signature
from evaluation.eval_qwen3_c1_quest_ruler import _build_work, _eos_ids, _load_dataset_manifest
from evaluation.fit_qwen3_8b_residual_kl_bank import sha256, write_json
from evaluation.profile_qwen3_8b_residual_two_sided_kl import full_attention, install_rank, load_bank, make_sidecar
from evaluation.ruler_v1 import paired_summary, parse_tasks, ruler_prompt, sample_score, summarize_arm

TASKS = "niah_single_1,niah_single_2,niah_single_3,niah_multikey_1,niah_multikey_2,niah_multiquery,niah_multivalue,vt,fwe,qa_1,qa_2"
ARMS = ("c1_exact_k", "uniform_r8")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("preflight", "smoke", "evaluate", "summarize"), required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--c1-checkpoint", type=Path, required=True)
    p.add_argument("--bank", type=Path, default=ROOT / "results/checkpoints/q8_qbase_fisher_bank")
    p.add_argument("--data-dir", type=Path, default=ROOT / "results/datasets/qwen3_8b_base_ruler_v1_32k_shadowkv11_s8")
    p.add_argument("--output-dir", type=Path, default=ROOT / "results/evaluation/q8_qbase_r8_ruler32k")
    p.add_argument("--samples-per-task", type=int, default=8)
    p.add_argument("--sequence-length", type=int, default=32768)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=4)
    p.add_argument("--torch-num-threads", type=int, default=2)
    return p


def arm_ranks(arm: str, layers: int) -> list[int] | None:
    assert arm in ARMS
    if arm == "c1_exact_k":
        return None
    return [8] * layers


def base_description(source):
    if source["format"] == "basisserve.closed_form_base_fisher_bank.v1":
        assert source["base_kind"] == "closed_form_rrr" and source["base_optimizer"] is None
        assert source["base_query_positions"] == []
        count = source.get('residual_query_count',len(source['residual_query_positions']))
        return f"closed-form MSE-RRR Base16 + Q{count} Page-Fisher R8"
    if source["format"] == "basisserve.residual_kl_bank.v1":
        assert source["fit_query"] == "last token of each window"
        return "closed-form MSE-RRR Base16 + Q1 Page-Fisher R8"
    assert source["format"] == "basisserve.qaware_base_fisher_bank.v1"
    return f"Q-aware Base16 + Q{len(source['residual_query_positions'])} Page-Fisher R8"


def validate_closed_form_bank(args, source):
    """Audit the original Base artifacts without relabeling the old bank."""
    assert source["fit_windows"] == 64 and source["diagnostic_windows"] == 16
    assert source["sequence_length"] == 32768 and source["sweeps"] == 40
    assert not source["validation_selects_factors"]
    assert Path(source["c1_checkpoint"]).resolve() == args.c1_checkpoint.resolve()
    audits = {}
    for layer in range(36):
        paths = sorted(Path(source["base_root"]).glob(f"shard_*/layer_{layer:03d}.safetensors"))
        assert len(paths) == 1
        original_path = paths[0]
        original_record = json.loads((original_path.parent / "result.json").read_text())
        assert original_record["status"] == "complete"
        assert original_record["format"] == "basisserve.qwen3_8b.v80_base16_page32_budget_sweep.v1"
        original_model = Path(original_record["protocol"]["model"])
        assert sha256(original_model / "config.json") == sha256(args.model / "config.json")
        assert any(row["layer"] == layer for row in original_record["layers"])
        original = load_file(str(original_path))
        bank = load_file(str(args.bank / f"layer_{layer:03d}.safetensors"))
        for name, shape in (
            ("base_left_b16", (8, 80, 16)), ("base_right_b16", (8, 16, 128)),
            ("base_bias_b16", (8, 128)), ("residual_encoder_b16_r8", (8, 128, 8)),
            ("residual_query_b16_r8", (32, 128, 8)),
        ):
            assert bank[name].shape == shape and bank[name].dtype == torch.float32
            assert torch.isfinite(bank[name]).all()
            if name.startswith("base_"):
                assert torch.equal(bank[name], original[name])
        audits[str(layer)] = {
            "original_base_sha256": sha256(original_path),
            "original_record_sha256": sha256(original_path.parent / "result.json"),
            "base_bitwise_equal": True,
        }
    return audits


def protocol(args):
    source = None
    bank_sha = {}
    for layer in range(36):
        path = args.bank / f"layer_{layer:03d}.safetensors"
        record = json.loads(path.with_suffix(".json").read_text())
        assert record["status"] == "complete" and record["layer"] == layer
        if layer == 0:
            source = record["protocol"]
        assert record["protocol"] == source
        bank_sha[str(layer)] = sha256(path)
        assert bank_sha[str(layer)] == record["sha256"]
    description = base_description(source)
    assert source["base_rank"] == 16 and 8 in source["ranks"]
    assert source["page_size"] == 32 and source["excluded_prefix_pages"] == 1
    assert sha256(args.c1_checkpoint / "results.json") == source["c1_manifest_sha256"]
    c1 = json.loads((args.c1_checkpoint / "results.json").read_text())
    closed_form_audit = None
    if source["format"] == "basisserve.residual_kl_bank.v1":
        closed_form_audit = validate_closed_form_bank(args, source)
    else:
        assert sha256(args.model / "config.json") == source["model_config_sha256"]
        for layer in range(36):
            assert sha256(args.c1_checkpoint / c1["artifacts"][str(layer)]["file"]) == source["inputs"][str(layer)]["c1_layer_sha256"]
            if source["format"] == "basisserve.closed_form_base_fisher_bank.v1":
                original_path = Path(source["frozen_base_bank"]) / f"layer_{layer:03d}.safetensors"
                assert sha256(original_path) == source["inputs"][str(layer)]["initial_bank_sha256"]
                original = load_file(str(original_path))
                current = load_file(str(args.bank / f"layer_{layer:03d}.safetensors"))
                for name, shape in (
                    ("base_left_b16", (8, 80, 16)), ("base_right_b16", (8, 16, 128)),
                    ("base_bias_b16", (8, 128)), ("residual_encoder_b16_r8", (8, 128, 8)),
                    ("residual_query_b16_r8", (32, 128, 8)),
                ):
                    assert current[name].shape == shape and current[name].dtype == torch.float32
                    assert torch.isfinite(current[name]).all()
                    if name.startswith("base_"):
                        assert torch.equal(current[name], original[name])
    tasks = parse_tasks(TASKS)
    manifest, manifest_hash = _load_dataset_manifest(
        args.data_dir, sequence_length=args.sequence_length, samples=args.samples_per_task,
        tasks=tasks, tokenizer_path=args.model,
    )
    assert manifest["protocol"]["tasks"] == [task.name for task in tasks]
    return {
        "format": "basisserve.uniform_residual_ruler.v1", "factor_bank_protocol": source,
        "base_description": description, "closed_form_bank_audit": closed_form_audit,
        "c1_layer_sha256": {str(layer): sha256(args.c1_checkpoint / c1["artifacts"][str(layer)]["file"])
                            for layer in range(36)},
        "bank_sha256": bank_sha, "layer_ranks": [8] * 36, "arms": list(ARMS),
        "rank_allocation": "none; uniform R8 without terminal KL or schedule loading",
        "dataset_manifest_sha256": manifest_hash, "ruler_revision": manifest["ruler"]["revision"],
        "tasks": [task.name for task in tasks], "samples_per_task": args.samples_per_task,
        "sequence_length": args.sequence_length, "gpu": "NVIDIA L40S",
        "model": str(args.model.resolve()), "c1_checkpoint": str(args.c1_checkpoint.resolve()),
        "bank": str(args.bank.resolve()), "data_dir": str(args.data_dir.resolve()),
        "model_dtype": "bfloat16", "generation": "greedy; task-specific cap; tokenizer/model EOS",
        "prompt": "official base input + answer_prefix; add_special_tokens=True; no chat template",
        "prefill": "single shared full-attention C1-V80 Triton prompt prefill",
        "first_token": "shared full-attention C1 prefill argmax",
        "decode": "independent immutable-prefix forks; true incremental post-RoPE residual sidecars",
        "sparse_backend": "native BF16 Page32 selector/attention; explicit full-support 4D decode mask disables alternate fused path",
        "dense_backend": "C1-V80 SDPA with the same explicit full-support decode mask",
        "page_size": 32, "physical_token_budget": 2048, "pinned_prefix_pages": 1,
        "force_current_page": False, "adaptive_budget": False,
        "scope": "88 paired samples, 11-task subset, previously used dataset; no refit or schedule tuning; not full RULER suite",
        "storage": "GPU exact K and materialized Base128+R sidecar; accuracy oracle, not PCIe/latency benchmark",
        "old_baseline_reused": False,
        "code_sha256": {name: sha256(ROOT / name) for name in (
            "evaluation/eval_qwen3_8b_residual_rank_ruler.py", "evaluation/ruler_v1.py",
            "evaluation/eval_qwen3_c1_quest_ruler.py", "basisserve/core/c1_k_routing_sidecar.py",
            "evaluation/profile_qwen3_8b_residual_two_sided_kl.py", "basisserve/core/residual_kl_replay.py",
            "basisserve/core/c1_conditional_page_attention.py", "basisserve/core/c1_v_conditional_k_router.py",
            "basisserve/checkpoint/gqa_vo_qwen3.py",
        )},
    }


@torch.inference_mode()
def prepare_arm(model, bank, prefix, ranks):
    modules = [layer.self_attn for layer in model.model.layers]
    cache = fork_routing_prefix(prefix)
    if ranks is None:
        full_attention(modules, "sdpa")
        return cache
    assert len(ranks) == len(modules) == len(bank)
    device = model.model.embed_tokens.weight.device
    positions = torch.arange(prefix.get_seq_length(), device=device)[None]
    cos, sin = model.model.rotary_emb(model.model.embed_tokens.weight[:1], positions)
    for layer, (module, tensors, rank) in enumerate(zip(modules, bank, ranks, strict=True)):
        install_rank(module, tensors, rank, 1)
        module.set_conditional_page_query_block_size(1, collect_statistics=False)
        assert module.conditional_residual_encoder.shape[-1] == rank
        cache._ensure_routing_layer(layer)
        cache._routing_sidecars[layer] = make_sidecar(module, prefix, cos, sin)
        assert cache.routing_sidecar(layer).shape[-1] == module.head_dim + rank
    return cache


@torch.inference_mode()
def greedy_decode(model, cache, first_token, *, maximum_tokens, eos_ids, trace=False):
    """One-token decoding with explicit full causal support at every layer.

    At each decode step all cached tokens and the current token are valid.
    Passing the prepared mask mapping preserves a non-None mask in attention,
    locking the sparse path to the same native implementation used by KL.
    """
    assert maximum_tokens > 0 and not model.model.has_sliding_layers
    device = model.model.embed_tokens.weight.device
    ids = [int(first_token)]
    logits_trace = []
    while len(ids) < maximum_tokens and ids[-1] not in eos_ids:
        length = cache.get_seq_length() + 1
        valid = torch.ones(1, 1, 1, length, dtype=torch.bool, device=device)
        output = model(
            input_ids=torch.tensor([[ids[-1]]], device=device), past_key_values=cache, use_cache=True,
            attention_mask={"full_attention": valid}, logits_to_keep=1,
        )
        cache = output.past_key_values
        logits = output.logits[0, -1]
        assert bool(torch.isfinite(logits).all())
        ids.append(int(logits.argmax()))
        if trace:
            logits_trace.append(logits.detach().cpu())
    return ids, cache, logits_trace


def decode_record(model, bank, prefix, first_token, task, tokenizer, references, ranks, maximum_tokens, *, smoke):
    started = time.monotonic()
    cache = prepare_arm(model, bank, prefix, ranks)
    native_calls = 0

    def observe(*args, **kwargs):
        nonlocal native_calls
        native_calls += 1
        ids, valid = _selected_pages(*args, **kwargs)
        assert ids.shape[-1] == 64 and bool(valid.all())
        assert bool((ids[..., 0] == 0).all())
        assert bool((ids.sort(dim=-1).values.diff(dim=-1) > 0).all())
        return ids, valid

    eos = _eos_ids(tokenizer, model)
    if smoke and ranks is not None:
        with patch("basisserve.core.c1_conditional_page_attention._selected_pages", side_effect=observe):
            ids, cache, trace = greedy_decode(model, cache, first_token, maximum_tokens=maximum_tokens, eos_ids=eos, trace=True)
        assert native_calls == (len(ids) - 1) * len(ranks)
    else:
        ids, cache, trace = greedy_decode(model, cache, first_token, maximum_tokens=maximum_tokens, eos_ids=eos, trace=smoke)
    assert cache.get_seq_length() == prefix.get_seq_length() + len(ids) - 1
    if ranks is not None:
        for layer, rank in enumerate(ranks):
            sidecar = cache.routing_sidecar(layer)
            assert sidecar.shape[-2] == cache.get_seq_length(layer)
            assert sidecar.shape[-1] == model.model.layers[layer].self_attn.head_dim + rank
    prediction = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    result = {
        "prediction": prediction, "generated_token_ids": ids, "generated_tokens": len(ids),
        "stopped_on_eos": ids[-1] in eos, "score": sample_score(prediction, references, task.match_type),
        "maximum_tokens": maximum_tokens, "elapsed_seconds": time.monotonic() - started,
        "native_selector_calls_smoke": native_calls if smoke and ranks is not None else None,
    }
    return result, trace


@torch.inference_mode()
def evaluate_sample(model, tokenizer, bank, work_row, args, settings):
    index, task, ordinal, source = work_row
    started = time.monotonic()
    tokens = tokenizer(ruler_prompt(source), add_special_tokens=True, return_tensors="pt")["input_ids"].to("cuda:0")
    prompt_length = int(tokens.shape[-1])
    assert prompt_length + task.tokens_to_generate <= args.sequence_length
    modules = [layer.self_attn for layer in model.model.layers]
    full_attention(modules, "triton")
    prefix = RoutingDynamicCache()
    output = model(input_ids=tokens, past_key_values=prefix, use_cache=True, logits_to_keep=1)
    assert bool(torch.isfinite(output.logits).all())
    first = int(output.logits[0, -1].argmax())
    del output, tokens
    torch.cuda.synchronize()
    prefill_seconds = time.monotonic() - started
    signature = prefix_signature(prefix)
    smoke = args.stage == "smoke"
    maximum_tokens = min(4, task.tokens_to_generate) if smoke else task.tokens_to_generate
    arms, traces = {}, {}
    print(f"[RULER {index:03d} {task.name}:{ordinal}] prefill={prefill_seconds:.2f}s prompt={prompt_length}", flush=True)
    for arm in ARMS:
        ranks = arm_ranks(arm, len(modules))
        arms[arm], trace = decode_record(model, bank, prefix, first, task, tokenizer, source["outputs"], ranks,
                                         maximum_tokens, smoke=smoke)
        if smoke:
            traces[arm] = trace
        assert prefix_signature(prefix) == signature
        print(f"[RULER {index:03d}] {arm} score={arms[arm]['score']:.4f} tokens={arms[arm]['generated_tokens']} seconds={arms[arm]['elapsed_seconds']:.2f}", flush=True)
    smoke_checks = None
    if smoke:
        repeated, trace = decode_record(model, bank, prefix, first, task, tokenizer, source["outputs"],
                                        [8] * 36, maximum_tokens, smoke=True)
        assert repeated["generated_token_ids"] == arms["uniform_r8"]["generated_token_ids"]
        for observed, expected in zip(trace, traces["uniform_r8"], strict=True):
            torch.testing.assert_close(observed, expected, atol=0, rtol=0)
        assert prefix_signature(prefix) == signature
        assert all(row["generated_tokens"] > 1 for row in arms.values())
        smoke_checks = {"uniform_repeat_logits_exact": True,
                        "native_selector_exercised": True, "rank_width_and_cache_alignment": True,
                        "prefix_unchanged": True, "generation_cap": 4, "not_official_accuracy": True}
    result = {
        "index": index, "key": f"{task.name}:{ordinal}", "task": task.name, "sample_ordinal": ordinal,
        "task_family": task.family, "match_type": task.match_type, "source_index": source.get("index"),
        "references": list(source["outputs"]), "prompt_tokens": prompt_length,
        "task_maximum_tokens": task.tokens_to_generate, "first_token_from_shared_prefill": first,
        "arms": arms, "prefill_seconds": prefill_seconds, "prefix_unchanged": True,
        "smoke_checks": smoke_checks, "wall_seconds": time.monotonic() - started,
        "maximum_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
    }
    return result


def summarize(args, settings, work):
    records = []
    for index, task, ordinal, source in work:
        payload = json.loads((args.output_dir / "evaluate" / f"sample_{index:03d}.json").read_text())
        assert payload["status"] == "complete" and payload["protocol"] == settings
        assert payload["gpu"] == settings["gpu"]
        row = payload["result"]
        assert row["index"] == index and row["key"] == f"{task.name}:{ordinal}"
        assert row["references"] == source["outputs"] and row["prefix_unchanged"]
        for arm in ARMS:
            arm_row = row["arms"][arm]
            assert arm_row["maximum_tokens"] == task.tokens_to_generate
            assert arm_row["score"] == sample_score(arm_row["prediction"], source["outputs"], task.match_type)
            assert arm_row["generated_token_ids"][0] == row["first_token_from_shared_prefill"]
        records.append(row)
    assert len(records) == 88 and len({row["key"] for row in records}) == 88
    tasks = parse_tasks(TASKS)
    summaries = {arm: summarize_arm(records, arm, tasks) for arm in ARMS}
    paired = {"uniform_vs_exact": paired_summary(records, "c1_exact_k", "uniform_r8", tasks)}
    write_json(args.output_dir / "result.json", {
        "status": "complete", "protocol": settings, "arms": summaries, "paired": paired,
        "records": records, "command": shlex.join(sys.argv), "python": sys.executable,
    })
    lines = [f"# Qwen3-8B {settings['base_description']}: RULER 32K pilot", "",
             f"Frozen C1-V80 + {settings['base_description']}; C1 full exact-K reference rerun. No KL allocation or adaptive schedule.",
             "11 tasks × 8 samples = 88 paired prompts. This is the existing 11-task subset, not the complete 13-task RULER suite.",
             "Official base completion prompts, greedy generation, task-specific caps and EOS. No old checkpoint accuracy is reused.", "",
             "| Task | Samples | C1 exact-K | Uniform R8 | R8-exact, pp | Improvements | Regressions |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for task in tasks:
        scores = [100 * summaries[arm]["tasks"][task.name]["accuracy"] for arm in ARMS]
        pair = paired["uniform_vs_exact"]["tasks"][task.name]
        lines.append(f"| {task.name} | 8 | {scores[0]:.4f}% | {scores[1]:.4f}% | {scores[1]-scores[0]:+.4f} | {pair['sparse_improvements']} | {pair['sparse_regressions']} |")
    means = [100 * summaries[arm]["task_balanced_accuracy"] for arm in ARMS]
    pair = paired["uniform_vs_exact"]["all_samples"]
    lines += [f"| Task-balanced mean | 88 | {means[0]:.4f}% | {means[1]:.4f}% | {means[1]-means[0]:+.4f} | {pair['sparse_improvements']} | {pair['sparse_regressions']} |", "",
              "## Measurement scope", "",
              "Both arms share one full-attention C1 Triton prefill. The first generated token is common; routing applies to subsequent decode forwards.",
              "Every arm has an independent immutable-prefix cache fork. Page32, B2048 including page0, no forced current page and no adaptive budget.",
              "An explicit full-support 4D decode mask locks routing to the native BF16 selector/attention used by the KL oracle, not the alternate fused decode path.",
              "Ranks/factors were not refitted or selected using RULER. The dataset has been used in previous experiments, so this is not an untouched final benchmark.",
              "Scores use the existing RULER case-insensitive substring metric: fraction of reference answers recovered for match_type=all; any-reference hit for match_type=part.",
              "Eight examples per task constitute a pilot, not strong evidence about small accuracy differences. Aggregate scores are task-balanced means.",
              "C1 exact-K is a reference for the added routing effect; this run does not include a new dense-V128 baseline.",
              "Exact K remains GPU-resident and Base128+R coordinates are materialized. This is not an offload-memory or PCIe-latency benchmark.", "",
              "Environment: basis, NVIDIA L40S, BF16. The result JSON records the factor provenance and exact evaluation commands.", ""]
    (args.output_dir / "summary.md").write_text("\n".join(lines))
    print(json.dumps({"accuracy": {arm: summaries[arm]["task_balanced_accuracy"] for arm in ARMS},
                      "paired_uniform_vs_exact": pair}, indent=2), flush=True)


@torch.inference_mode()
def main():
    args = parser().parse_args()
    assert args.samples_per_task == 8 and args.sequence_length == 32768
    assert 0 <= args.shard_index < args.num_shards
    torch.set_num_threads(args.torch_num_threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    settings = protocol(args)
    if args.stage == "preflight":
        print(json.dumps({"status": "preflight_passed", "protocol": settings}, indent=2), flush=True)
        return
    work = _build_work(args.data_dir, parse_tasks(TASKS), args.samples_per_task)
    assert len(work) == 88
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == "summarize":
        summarize(args, settings, work)
        return
    assert torch.cuda.is_available() and torch.cuda.get_device_name(0) == settings["gpu"]
    torch.cuda.set_device(0)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, local_files_only=True,
        low_cpu_mem_usage=True, attn_implementation="sdpa").to("cuda:0").eval()
    install_qwen3_gqa_vo_als_export(model, args.c1_checkpoint, attention_backend="triton")
    assert len(model.model.layers) == 36 and not model.model.has_sliding_layers
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    bank = load_bank(args.bank)
    assigned = work[:1] if args.stage == "smoke" else work[args.shard_index::args.num_shards]
    for row in assigned:
        path = args.output_dir / args.stage / f"sample_{row[0]:03d}.json"
        if path.exists():
            saved = json.loads(path.read_text())
            assert saved["status"] == "complete" and saved["protocol"] == settings and saved["gpu"] == settings["gpu"]
            assert saved["result"]["key"] == f"{row[1].name}:{row[2]}"
            print(f"[resume] {saved['result']['key']}", flush=True)
            continue
        torch.cuda.reset_peak_memory_stats()
        result = evaluate_sample(model, tokenizer, bank, row, args, settings)
        write_json(path, {"status": "complete", "protocol": settings, "result": result,
                          "gpu": torch.cuda.get_device_name(0), "python": sys.executable,
                          "torch": torch.__version__, "command": shlex.join(sys.argv)})
        print(f"[complete] sample={row[0]} seconds={result['wall_seconds']:.2f} peak_GiB={result['maximum_allocated_gib']:.2f}", flush=True)
    write_json(args.output_dir / args.stage / f"shard_{args.shard_index}.json", {
        "status": "complete", "protocol": settings, "indices": [row[0] for row in assigned],
        "command": shlex.join(sys.argv),
    })


if __name__ == "__main__":
    main()
