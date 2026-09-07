#!/usr/bin/env python3
"""Official QUEST accuracy forward versus C1 routing, with two full layers."""

import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basisserve.checkpoint.gqa_vo_qwen3 import install_qwen3_gqa_vo_als_export
from basisserve.core.c1_k_routing_sidecar import RoutingDynamicCache
from basisserve.core.c1_official_quest import load_quest, OfficialQuest
from basisserve.core.residual_kl_replay import prefix_signature
from evaluation import eval_qwen3_8b_residual_rank_ruler as previous
from evaluation.fit_qwen3_8b_residual_kl_bank import sha256, write_json
from evaluation.profile_qwen3_8b_residual_two_sided_kl import full_attention, load_bank
from evaluation.ruler_v1 import paired_summary, parse_tasks, ruler_prompt, sample_score, summarize_arm

ARMS = ("c1_exact_k", "uniform_r8_full2", "quest_official_full2")
QUEST_COMMIT = "01c1623bf9395009520874e989e29f683203b357"


def parser():
    p = previous.parser()
    p.description = __doc__
    p.set_defaults(bank=ROOT / "results/checkpoints/q8_qbase_fisher16_r8",
        output_dir=ROOT / "results/evaluation/quest_official_ruler32k")
    p.add_argument("--quest-repo", type=Path, default=ROOT.parent / "Quest")
    return p


def protocol(args):
    result = previous.protocol(args)
    revision = subprocess.check_output(["git", "-C", str(args.quest_repo), "rev-parse", "HEAD"], text=True).strip()
    assert revision == QUEST_COMMIT
    source = args.quest_repo / "evaluation/quest_attention.py"
    original = subprocess.check_output(["git", "-C", str(args.quest_repo), "show", "HEAD:evaluation/quest_attention.py"])
    assert source.read_bytes() == original
    result.update({"format": "basisserve.official_quest_ruler.v1", "arms": list(ARMS),
        "quest_commit": revision, "quest_source_sha256": sha256(source),
        "quest_source": str(source.resolve()), "full_layers": [0, 1], "sparse_layers": list(range(2, 36)),
        "physical_token_budget": None,
        "budgets": {"uniform_r8_full2": "2048 physical tokens per KV group; Page32 including page0",
            "quest_official_full2": "2048 tokens per Q head; Page32; no pin; measure physical GQA union"},
        "pinned_prefix_pages": {"uniform_r8_full2": 1, "quest_official_full2": 0},
        "decode": "independent cache forks; first two layers full in both sparse arms",
        "sparse_backend": "QUEST: unmodified upstream accuracy forward; ours: native conditional Page-LSE",
        "bridge": "post-norm post-RoPE Q/K passed unchanged via identity projections/RoPE; C1-V zero-padded to K width and sliced on output",
        "storage": "all exact K and C1-V resident on GPU; upstream computes full QK then masks; not a sparse-kernel or PCIe speed benchmark",
        "scope": "same previously used RULER 11 tasks x 8 prompts; C1 payload adaptation, not original-model paper reproduction"})
    for name in ("basisserve/core/c1_official_quest.py", "evaluation/eval_qwen3_8b_official_quest_ruler.py"):
        result["code_sha256"][name] = sha256(ROOT / name)
    return result


@torch.inference_mode()
def decode(model, tokenizer, bank, upstream, prefix, first, task, references, arm, smoke):
    started = time.monotonic()
    modules = [layer.self_attn for layer in model.model.layers]
    maximum = min(4, task.tokens_to_generate) if smoke else task.tokens_to_generate
    eos = previous._eos_ids(tokenizer, model)
    if arm == "quest_official_full2":
        with OfficialQuest(model, prefix, upstream) as runtime:
            ids, cache, trace = previous.greedy_decode(model, runtime.cache, first,
                maximum_tokens=maximum, eos_ids=eos, trace=smoke)
            stats = runtime.statistics()
            assert stats["layer_calls"] == [0, 0] + [len(ids) - 1] * (len(modules) - 2)
    else:
        ranks = [8] * len(modules) if arm == "uniform_r8_full2" else None
        cache = previous.prepare_arm(model, bank, prefix, ranks)
        if ranks is not None:
            full_attention(modules[:2], "sdpa")
            for layer, module in enumerate(modules):
                if layer < 2:
                    module.set_routing_projectors(None, None)
                    cache._routing_sidecars[layer] = None
                else:
                    module.set_conditional_page_query_block_size(1, collect_statistics=True)
        ids, cache, trace = previous.greedy_decode(model, cache, first,
            maximum_tokens=maximum, eos_ids=eos, trace=smoke)
        if ranks is None:
            stats = {}
        else:
            count = (len(ids) - 1) * sum(module.num_key_value_heads for module in modules[2:])
            selected = sum(module.reverse_shadow_statistics()["selected_tokens"] for module in modules[2:])
            stats = {"physical_selected_tokens_sum": selected, "kv_group_query_count": count,
                "mean_physical_tokens_per_sparse_kv_group": selected / count if count else None}
    assert cache.get_seq_length() == prefix.get_seq_length() + len(ids) - 1
    prediction = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return {"prediction": prediction, "generated_token_ids": ids, "generated_tokens": len(ids),
        "maximum_tokens": maximum, "stopped_on_eos": ids[-1] in eos,
        "score": sample_score(prediction, references, task.match_type), "routing_statistics": stats,
        "elapsed_seconds": time.monotonic() - started}, trace


@torch.inference_mode()
def evaluate_sample(model, tokenizer, bank, upstream, work_row, args):
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
    del output, tokens
    signature = prefix_signature(prefix)
    arms = {}
    smoke = args.stage == "smoke"
    print(f"[sample {index:03d}] {task.name}:{ordinal} prompt={prompt_length}", flush=True)
    for arm in ARMS:
        arms[arm], trace = decode(model, tokenizer, bank, upstream, prefix, first, task, source["outputs"], arm, smoke)
        assert prefix_signature(prefix) == signature
        if smoke:
            repeated, repeated_trace = decode(model, tokenizer, bank, upstream, prefix, first, task, source["outputs"], arm, True)
            assert repeated["generated_token_ids"] == arms[arm]["generated_token_ids"]
            for a, b in zip(trace, repeated_trace, strict=True):
                torch.testing.assert_close(a, b, atol=0, rtol=0)
            assert prefix_signature(prefix) == signature
        print(f"[sample {index:03d}] {arm} score={arms[arm]['score']:.4f} tokens={arms[arm]['generated_tokens']} seconds={arms[arm]['elapsed_seconds']:.2f}", flush=True)
    return {"index": index, "key": f"{task.name}:{ordinal}", "task": task.name,
        "sample_ordinal": ordinal, "references": source["outputs"], "prompt_tokens": prompt_length,
        "first_token_from_shared_prefill": first, "prefix_unchanged": True,
        "smoke_repeat_logits_exact": smoke, "arms": arms, "wall_seconds": time.monotonic() - started,
        "maximum_allocated_gib": torch.cuda.max_memory_allocated() / 2**30}


def summarize(args, settings, work):
    records = []
    for index, task, ordinal, source in work:
        saved = json.loads((args.output_dir / "evaluate" / f"sample_{index:03d}.json").read_text())
        assert saved["status"] == "complete" and saved["protocol"] == settings
        row = saved["result"]
        assert row["key"] == f"{task.name}:{ordinal}" and row["prefix_unchanged"]
        assert row["references"] == source["outputs"]
        for arm in ARMS:
            data = row["arms"][arm]
            assert data["maximum_tokens"] == task.tokens_to_generate
            assert data["score"] == sample_score(data["prediction"], source["outputs"], task.match_type)
        records.append(row)
    assert len(records) == 88 and len({r["key"] for r in records}) == 88
    tasks = parse_tasks(previous.TASKS)
    means = {arm: summarize_arm(records, arm, tasks) for arm in ARMS}
    traffic = {}
    for arm in ARMS[1:]:
        stats = [r["arms"][arm]["routing_statistics"] for r in records]
        traffic[arm] = sum(r["physical_selected_tokens_sum"] for r in stats) / sum(r["kv_group_query_count"] for r in stats)
    write_json(args.output_dir / "result.json", {"status": "complete", "protocol": settings,
        "arms": means, "mean_physical_tokens_per_sparse_kv_group": traffic, "records": records,
        "paired": paired_summary(records, "uniform_r8_full2", "quest_official_full2", tasks),
        "python": sys.executable, "command": shlex.join(sys.argv)})
    lines = ["# Official QUEST accuracy forward: C1-V80 RULER 32K", "",
        "Qwen3-8B-Base, same 88 reused prompts, shared full C1 prefill/first token, greedy generation, L40S BF16.",
        "Both sparse arms retain full attention in layers 0 and 1. No factors are refitted.",
        "QUEST directly calls unmodified mit-han-lab/Quest evaluation/quest_attention.py at commit " + QUEST_COMMIT + ".",
        "Page32, 2048 tokens per Q head, no pinned page. Our Q8 Base16/Q16 Fisher R8 selects 2048 physical tokens per KV group including page0.",
        "The model bridge preserves Qwen3 norm/RoPE and C1-V80; it is not original-model paper accuracy or an optimized offload/kernel benchmark.", "",
        "| Task | Full exact K | Base16+R8, full2 | Official QUEST, full2 |", "|---|---:|---:|---:|"]
    for task in tasks:
        lines.append("| " + task.name + " | " + " | ".join(f"{100 * means[a]['tasks'][task.name]['accuracy']:.4f}%" for a in ARMS) + " |")
    lines.append("| Task-balanced mean | " + " | ".join(f"{100 * means[a]['task_balanced_accuracy']:.4f}%" for a in ARMS) + " |")
    lines += ["", "Mean unique physical tokens / sparse-layer KV group (weighted by generated decode steps):"]
    lines += [f"- {arm}: {value:.3f}" for arm, value in traffic.items()]
    lines += ["", "QUEST's per-head budget is not an equal physical GQA budget. Neither method's count includes the two full layers.", ""]
    (args.output_dir / "summary.md").write_text("\n".join(lines))
    print(json.dumps({"accuracy": {a: means[a]["task_balanced_accuracy"] for a in ARMS}, "physical_tokens": traffic}, indent=2), flush=True)


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
    if args.stage == "summarize":
        summarize(args, settings, work)
        return
    assert torch.cuda.is_available() and torch.cuda.get_device_name(0) == settings["gpu"]
    upstream = load_quest(args.quest_repo / "evaluation/quest_attention.py")
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
        local_files_only=True, low_cpu_mem_usage=True, attn_implementation="sdpa").to("cuda:0").eval()
    install_qwen3_gqa_vo_als_export(model, args.c1_checkpoint, attention_backend="triton")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    bank = load_bank(args.bank)
    assigned = work[:1] if args.stage == "smoke" else work[args.shard_index::args.num_shards]
    for row in assigned:
        path = args.output_dir / args.stage / f"sample_{row[0]:03d}.json"
        if path.exists():
            saved = json.loads(path.read_text())
            assert saved["status"] == "complete" and saved["protocol"] == settings
            continue
        torch.cuda.reset_peak_memory_stats()
        result = evaluate_sample(model, tokenizer, bank, upstream, row, args)
        write_json(path, {"status": "complete", "protocol": settings, "result": result,
            "gpu": torch.cuda.get_device_name(0), "python": sys.executable, "command": shlex.join(sys.argv)})
    write_json(args.output_dir / args.stage / f"shard_{args.shard_index}.json",
        {"status": "complete", "protocol": settings, "indices": [r[0] for r in assigned]})


if __name__ == "__main__":
    main()
