#!/usr/bin/env python3
"""C1-V80 RULER comparison: conditional R8, QUEST and Loki PCA routing."""

from contextlib import nullcontext
import json
from pathlib import Path
import shlex
import sys
import time

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basisserve.checkpoint.gqa_vo_qwen3 import install_qwen3_gqa_vo_als_export
from basisserve.core.c1_k_routing_sidecar import RoutingDynamicCache
from basisserve.core.c1_routing_comparison import BASELINES, RoutingBaseline
from basisserve.core.residual_kl_replay import prefix_signature
from evaluation import eval_qwen3_8b_residual_rank_ruler as previous
from evaluation.fit_qwen3_8b_residual_kl_bank import sha256, write_json
from evaluation.profile_qwen3_8b_residual_two_sided_kl import full_attention, load_bank
from evaluation.ruler_v1 import paired_summary, parse_tasks, ruler_prompt, sample_score, summarize_arm

ARMS = ("c1_exact_k", "uniform_r8", *BASELINES)


def parser():
    p = previous.parser()
    p.description = __doc__
    p.set_defaults(bank=ROOT / "results/checkpoints/q8_qbase_fisher16_r8",
        output_dir=ROOT / "results/evaluation/routing_baselines_ruler32k")
    p.add_argument("--loki-checkpoint", type=Path,
        default=ROOT / "results/checkpoints/qwen3_8b_loki_r32_c4_64f_s32768")
    return p


def protocol(args):
    settings = previous.protocol(args)
    manifest = json.loads((args.loki_checkpoint / "result.json").read_text())
    assert manifest["status"] == "complete" and manifest["geometry"]["rank"] == 32
    assert manifest["model"]["config_sha256"] == sha256(args.model / "config.json")
    calibration = manifest["calibration"]
    assert calibration["samples"] == 64 and calibration["sequence_length"] == 32768
    assert calibration["windows_sha256"] == settings["factor_bank_protocol"]["windows_sha256"]
    factors = args.loki_checkpoint / manifest["artifacts"]["factors"]["file"]
    assert sha256(factors) == manifest["artifacts"]["factors"]["sha256"]
    settings.update({"format": "basisserve.routing_baselines_ruler.v1", "arms": list(ARMS),
        "loki_checkpoint": str(args.loki_checkpoint.resolve()), "loki_manifest": manifest,
        "loki_manifest_sha256": sha256(args.loki_checkpoint / "result.json"),
        "sparse_layers": list(range(36)),
        "quest": "exact post-RoPE min/max; raw-bound GQA max; 64 Page32 including pinned page0",
        "loki_page": "Loki PCA proxy + existing normalized Page-LSE GQA max; 64 Page32 including pinned page0; adapted baseline",
        "loki_token": "original per-query-head token Top2048; no pinned prefix; physical GQA union measured; NOT equal physical budget",
        "comparison": "same C1-V80 and full prefill; all layers sparse; no original-paper accuracy/latency reproduction",
        "storage": "exact K remains on GPU; actual materialized routing metadata reported separately; no measured PCIe traffic",
        "sparse_backend": "native BF16 selected exact QK and C1-V; explicit full-support decode mask",
        "old_baseline_reused": False})
    for name in ("evaluation/eval_qwen3_8b_routing_baselines_ruler.py",
                 "basisserve/core/c1_routing_comparison.py", "basisserve/core/c1_loki_attention.py",
                 "basisserve/core/c1_k_reverse_shadow.py"):
        settings["code_sha256"][name] = sha256(ROOT / name)
    return settings


@torch.inference_mode()
def decode_arm(model, bank, projector, prefix, first, task, tokenizer, references, arm, *, smoke):
    modules = [layer.self_attn for layer in model.model.layers]
    started = time.monotonic()
    maximum = min(4, task.tokens_to_generate) if smoke else task.tokens_to_generate
    if arm in BASELINES:
        context = RoutingBaseline(model, prefix, projector, arm)
        cache = context.cache
    else:
        cache = previous.prepare_arm(model, bank, prefix, None if arm == "c1_exact_k" else [8] * len(modules))
        if arm == "uniform_r8":
            for module in modules:
                module.set_conditional_page_query_block_size(1, collect_statistics=True)
        context = nullcontext(None)
    with context as runtime:
        ids, cache, trace = previous.greedy_decode(model, cache, first,
            maximum_tokens=maximum, eos_ids=previous._eos_ids(tokenizer, model), trace=smoke)
        if runtime is not None:
            statistics = runtime.statistics()
        elif arm == "uniform_r8":
            selected = sum(module.reverse_shadow_statistics()["selected_tokens"] for module in modules)
            count = (len(ids) - 1) * sum(module.num_key_value_heads for module in modules)
            statistics = {"selector_calls": (len(ids) - 1) * len(modules),
                "physical_selected_tokens_sum": selected, "kv_group_query_count": count,
                "mean_physical_tokens_per_kv_group": selected / count if count else None,
                "final_resident_routing_metadata_bytes": sum(x.numel() * x.element_size()
                    for x in cache._routing_sidecars if x is not None)}
        else:
            statistics = {"selector_calls": 0, "final_resident_routing_metadata_bytes": 0}
    assert cache.get_seq_length() == prefix.get_seq_length() + len(ids) - 1
    if arm != "c1_exact_k":
        assert statistics["selector_calls"] == (len(ids) - 1) * len(modules)
        mean = statistics["mean_physical_tokens_per_kv_group"]
        if mean is not None:
            assert 0 < mean <= (8192 if arm == "loki_token_r32" else 2048)
    prediction = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return {"prediction": prediction, "generated_token_ids": ids, "generated_tokens": len(ids),
        "stopped_on_eos": ids[-1] in previous._eos_ids(tokenizer, model),
        "score": sample_score(prediction, references, task.match_type), "maximum_tokens": maximum,
        "routing_statistics": statistics, "elapsed_seconds": time.monotonic() - started}, trace


@torch.inference_mode()
def evaluate_sample(model, tokenizer, bank, projector, work_row, args):
    index, task, ordinal, source = work_row
    started = time.monotonic()
    tokens = tokenizer(ruler_prompt(source), add_special_tokens=True, return_tensors="pt")["input_ids"].to("cuda:0")
    prompt_length = tokens.shape[-1]
    assert prompt_length + task.tokens_to_generate <= args.sequence_length
    full_attention([layer.self_attn for layer in model.model.layers], "triton")
    prefix = RoutingDynamicCache()
    output = model(input_ids=tokens, past_key_values=prefix, use_cache=True, logits_to_keep=1)
    assert bool(torch.isfinite(output.logits).all())
    first = int(output.logits[0, -1].argmax())
    del tokens, output
    signature = prefix_signature(prefix)
    arms = {}
    smoke = args.stage == "smoke"
    print(f"[sample {index:03d}] {task.name}:{ordinal} prompt={prompt_length}", flush=True)
    for arm in ARMS:
        arms[arm], trace = decode_arm(model, bank, projector, prefix, first, task,
            tokenizer, source["outputs"], arm, smoke=smoke)
        assert prefix_signature(prefix) == signature
        if smoke:
            repeated, repeated_trace = decode_arm(model, bank, projector, prefix, first, task,
                tokenizer, source["outputs"], arm, smoke=True)
            assert repeated["generated_token_ids"] == arms[arm]["generated_token_ids"]
            for a, b in zip(trace, repeated_trace, strict=True):
                torch.testing.assert_close(a, b, atol=0, rtol=0)
            assert prefix_signature(prefix) == signature
        print(f"[sample {index:03d}] {arm} score={arms[arm]['score']:.4f} tokens={arms[arm]['generated_tokens']} seconds={arms[arm]['elapsed_seconds']:.2f}", flush=True)
    return {"index": index, "key": f"{task.name}:{ordinal}", "task": task.name,
        "sample_ordinal": ordinal, "references": list(source["outputs"]),
        "prompt_tokens": prompt_length, "first_token_from_shared_prefill": first,
        "prefix_unchanged": True, "arms": arms, "smoke_repeat_logits_exact": smoke,
        "wall_seconds": time.monotonic() - started,
        "maximum_allocated_gib": torch.cuda.max_memory_allocated() / 2**30}


def summarize(args, settings, work):
    records = []
    for index, task, ordinal, source in work:
        saved = json.loads((args.output_dir / "evaluate" / f"sample_{index:03d}.json").read_text())
        assert saved["status"] == "complete" and saved["protocol"] == settings
        assert saved["gpu"] == settings["gpu"]
        row = saved["result"]
        assert row["key"] == f"{task.name}:{ordinal}" and row["references"] == source["outputs"]
        assert row["prefix_unchanged"]
        for arm in ARMS:
            arm_row = row["arms"][arm]
            assert arm_row["maximum_tokens"] == task.tokens_to_generate
            assert arm_row["score"] == sample_score(arm_row["prediction"], source["outputs"], task.match_type)
            assert arm_row["generated_token_ids"][0] == row["first_token_from_shared_prefill"]
        records.append(row)
    assert len(records) == 88
    tasks = parse_tasks(previous.TASKS)
    means = {arm: summarize_arm(records, arm, tasks) for arm in ARMS}
    pairs = {arm: paired_summary(records, "uniform_r8", arm, tasks) for arm in BASELINES}
    traffic = {}
    for arm in ARMS[1:]:
        rows = [row["arms"][arm]["routing_statistics"] for row in records]
        traffic[arm] = {"mean_physical_tokens_per_kv_group":
            sum(x["physical_selected_tokens_sum"] for x in rows) / sum(x["kv_group_query_count"] for x in rows),
            "maximum_resident_routing_metadata_bytes": max(x["final_resident_routing_metadata_bytes"] for x in rows)}
    result = {"status": "complete", "protocol": settings, "arms": means,
        "paired_vs_uniform_r8": pairs, "routing": traffic, "records": records,
        "python": sys.executable, "command": shlex.join(sys.argv)}
    write_json(args.output_dir / "result.json", result)
    lines = ["# C1-V80 routing comparison: RULER 32K", "",
        "Same 88 previously used prompts, full C1 prefill, shared first token, 36 sparse layers, BF16 on L40S.",
        "Uniform R8 uses the existing Q8 Base16 and Q16 Page-Fisher residual. Loki uses the existing C4 64x32K PCA checkpoint.",
        "QUEST uses raw-bound GQA max. Loki-page uses normalized Page-LSE GQA max and is an adaptation of Loki.",
        "The first three sparse methods use 64 Page32 including page0. Loki-token selects 2048 tokens independently per query head; its physical budget is larger.",
        "These are C1 selector controls, not full reproductions of the original papers. Exact K stays on GPU; elapsed time is not an optimized offload benchmark.", "",
        "| Task | Exact K | Base16+R8 | QUEST-page | Loki-page R32 | Loki-token R32 |",
        "|---|---:|---:|---:|---:|---:|"]
    for task in tasks:
        values = [100 * means[arm]["tasks"][task.name]["accuracy"] for arm in ARMS]
        lines.append("| " + task.name + " | " + " | ".join(f"{v:.4f}%" for v in values) + " |")
    lines.append("| Task-balanced mean | " + " | ".join(f"{100 * means[arm]['task_balanced_accuracy']:.4f}%" for arm in ARMS) + " |")
    lines += ["", "| Method | Mean selected physical tokens / KV group | Peak routing metadata, MiB |",
        "|---|---:|---:|"]
    for arm, values in traffic.items():
        lines.append(f"| {arm} | {values['mean_physical_tokens_per_kv_group']:.3f} | {values['maximum_resident_routing_metadata_bytes'] / 2**20:.3f} |")
    lines += ["", "Routing storage above reports actual materialized caches: our Base128+R8, Loki R32, or QUEST min/max. It does not equate the deployed residual-only R8 storage with this oracle's materialized Base.",
        "Eight examples per task do not establish small accuracy differences reliably. No configuration was selected using these new results.", ""]
    (args.output_dir / "summary.md").write_text("\n".join(lines))
    print(json.dumps({"accuracy": {arm: x["task_balanced_accuracy"] for arm, x in means.items()}, "routing": traffic}, indent=2), flush=True)


@torch.inference_mode()
def main():
    args = parser().parse_args()
    assert args.samples_per_task == 8 and args.sequence_length == 32768
    assert 0 <= args.shard_index < args.num_shards
    torch.set_num_threads(args.torch_num_threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    settings = protocol(args)
    work = previous._build_work(args.data_dir, parse_tasks(previous.TASKS), args.samples_per_task)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == "summarize":
        summarize(args, settings, work)
        return
    assert torch.cuda.is_available() and torch.cuda.get_device_name(0) == settings["gpu"]
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
        local_files_only=True, low_cpu_mem_usage=True, attn_implementation="sdpa").to("cuda:0").eval()
    install_qwen3_gqa_vo_als_export(model, args.c1_checkpoint, attention_backend="triton")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    bank = load_bank(args.bank)
    projector = load_file(str(args.loki_checkpoint / "factors.safetensors"))["key_projector"]
    assert projector.shape == (36, 8, 128, 32) and bool(torch.isfinite(projector).all())
    assigned = work[:1] if args.stage == "smoke" else work[args.shard_index::args.num_shards]
    for row in assigned:
        path = args.output_dir / args.stage / f"sample_{row[0]:03d}.json"
        if path.exists():
            saved = json.loads(path.read_text())
            assert saved["status"] == "complete" and saved["protocol"] == settings
            continue
        torch.cuda.reset_peak_memory_stats()
        result = evaluate_sample(model, tokenizer, bank, projector, row, args)
        write_json(path, {"status": "complete", "protocol": settings, "result": result,
            "gpu": torch.cuda.get_device_name(0), "python": sys.executable, "command": shlex.join(sys.argv)})
    write_json(args.output_dir / args.stage / f"shard_{args.shard_index}.json",
        {"status": "complete", "protocol": settings, "indices": [row[0] for row in assigned]})


if __name__ == "__main__":
    main()
