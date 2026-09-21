"""Paired Dense-V RULER-64K evaluation for the Section 4 routing ablations."""

import argparse
import csv
from pathlib import Path
import shlex
import sys
import time

import torch
from transformers import AutoTokenizer

from evaluation import eval_k_routing_ruler as common


METHODS = (
    "dense_full",
    "exact_k",
    "r16_only",
    "r20_only",
    "b4r16",
    "b16_only",
    "b16r16",
    "r32_only",
    "residual_mse",
    "score_mse",
)
TASK_NAMES = common.DEFAULT_TASK_NAMES


def bank_path(root, section4, method):
    return {
        "dense_full": root / "ours_b16r16",
        "exact_k": root / "ours_b16r16",
        "r16_only": root / "b0r16/ours_b0r16",
        "r20_only": root / "b0r20/ours_b0r20",
        "b4r16": root / "b4r16/ours_b4r16",
        "b16_only": section4 / "base_rank_sweep/b16r0",
        "b16r16": root / "ours_b16r16",
        "r32_only": root / "b0r32/ours_b0r32",
        "residual_mse": section4 / "objectives/residual_mse/ours_b16r16",
        "score_mse": section4 / "objectives/score_mse/ours_b16r16",
    }[method]


def runtime_arm(method):
    if method == "dense_full":
        return "full"
    if method == "exact_k":
        return "exact_sparse"
    return "ours"


def method_description(method):
    return {
        "dense_full": "full exact attention, Dense V128",
        "exact_k": "Exact-K Page32 oracle, sink32/recent64 inside B2048",
        "r16_only": "Page-Fisher R16-only",
        "r20_only": "Page-Fisher R20-only",
        "b4r16": "affine Base4 plus Page-Fisher R16",
        "b16_only": "affine Base16-only",
        "b16r16": "affine Base16 plus Page-Fisher R16",
        "r32_only": "Page-Fisher R32-only",
        "residual_mse": "affine Base16 plus Residual-MSE R16",
        "score_mse": "affine Base16 plus Score-MSE R16",
    }[method]


def inputs(args):
    identity_path = args.root / "manifests/v128.json"
    bank = bank_path(args.root, args.section4, args.method)
    namespace = argparse.Namespace(
        identity=identity_path,
        data=args.root / "ruler64k",
        bank=bank,
        output=args.output,
        sequence_length=65536,
        rope="native",
        dense_v=True,
        native_audit=None,
        full_smoke=None,
        wo_bank=None,
        arm="ours",
        stage="evaluate",
        official_lrqk=False,
        chat_template=True,
    )
    identity = common.read_json(identity_path)
    tokenizer = AutoTokenizer.from_pretrained(identity["model"], local_files_only=True)
    identity, manifest, rows, factors, hashes, protocol = common.inputs(
        namespace, tokenizer, task_names=TASK_NAMES, samples_per_task=8
    )
    protocol["section4_method"] = args.method
    protocol["section4_method_description"] = method_description(args.method)
    protocol["value_mode"] = "dense original V and Wo for every method"
    protocol["budget"] = {
        "tokens": 2048,
        "page_size": 32,
        "sink_tokens_inside_budget": 32,
        "recent_tokens_inside_budget": 64,
        "historical_pages": 62,
        "freely_routed_historical_pages": 61,
    }
    protocol["final_attention"] = "exact selected post-RoPE K and exact QK; proxy selects support only"
    protocol["source_sha256"]["evaluation/eval_llama_section4_ruler.py"] = common.sha256(Path(__file__))
    return tokenizer, identity, manifest, rows, factors, hashes, protocol


def verify(saved, row, protocol, hashes, method):
    assert saved["status"] == "complete" and saved["sample"] == row and saved["protocol"] == protocol
    assert saved["bank_sha256"] == ({} if method in ("dense_full", "exact_k") else hashes)
    result = saved["result"]
    assert 0 < len(result["ids"]) <= row["maximum_tokens"]
    assert common.sample_score(result["prediction"], row["answers"], row["match_type"]) == result["score"]


@torch.inference_mode()
def run(args, *, smoke):
    tokenizer, identity, manifest, rows, factors, hashes, protocol = inputs(args)
    arm = runtime_arm(args.method)
    assert torch.cuda.device_count() == 1 and "L40S" in torch.cuda.get_device_name(0)
    model = common.load_evaluation_model(identity, common.routing_config(identity, rope="native", sequence_length=65536))
    common.install(model, Path(identity["checkpoint"]), manifest, arm, factors, dense_v=True)
    selected = [rows[0], rows[56]] if smoke else rows[args.shard_index::args.num_shards]
    stage = "smoke" if smoke else "evaluate"
    for row in selected:
        path = args.output / args.method / stage / f"sample_{row['index']:03d}.json"
        if path.exists():
            verify(common.read_json(path), row, protocol, hashes, args.method)
            continue
        cap = min(4, row["maximum_tokens"]) if smoke else row["maximum_tokens"]
        torch.manual_seed(0)
        started = time.monotonic()
        ids, first, statistics, stopped = common.generate(model, tokenizer, row, arm, cap)
        if smoke:
            again, other, _, _ = common.generate(model, tokenizer, row, arm, cap)
            assert ids == again
            torch.testing.assert_close(first, other, atol=0, rtol=0)
        prediction = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        saved = {
            "status": "complete",
            "sample": row,
            "protocol": protocol,
            "bank_sha256": {} if args.method in ("dense_full", "exact_k") else hashes,
            "command": shlex.join(sys.argv),
            "python": sys.executable,
            "gpu": torch.cuda.get_device_name(0),
            "result": {
                "ids": ids,
                "prediction": prediction,
                "score": common.sample_score(prediction, row["answers"], row["match_type"]),
                "stopped": stopped,
                "routing": statistics,
                "seconds": time.monotonic() - started,
                "peak_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
            },
        }
        verify(saved, row, protocol, hashes, args.method)
        common.write_json(path, saved)
        print("COMPLETE", args.method, row["index"], saved["result"]["score"], saved["result"]["seconds"], flush=True)
    if smoke:
        common.write_json(
            args.output / args.method / "smoke_audit.json",
            {"status": "complete", "method": args.method, "protocol": protocol, "samples": [0, 56]},
        )


def summarize_method(args):
    _, _, _, rows, _, hashes, protocol = inputs(args)
    gate = common.read_json(args.output / args.method / "smoke_audit.json")
    assert gate["status"] == "complete" and gate["protocol"] == protocol
    results = []
    for row in rows:
        saved = common.read_json(args.output / args.method / "evaluate" / f"sample_{row['index']:03d}.json")
        verify(saved, row, protocol, hashes, args.method)
        results.append(saved["result"])
    tasks = {}
    for task in dict.fromkeys(row["task"] for row in rows):
        selected = [result["score"] for row, result in zip(rows, results, strict=True) if row["task"] == task]
        assert len(selected) == 8
        tasks[task] = 100 * sum(selected) / len(selected)
    summary = {
        "status": "complete",
        "method": args.method,
        "protocol": protocol,
        "samples": 88,
        "mean": 100 * sum(result["score"] for result in results) / 88,
        "tasks": tasks,
        "results": results,
    }
    common.write_json(args.output / args.method / "summary.json", summary)
    print(args.method, summary["mean"], flush=True)


def aggregate(args):
    summaries = {method: common.read_json(args.output / method / "summary.json") for method in METHODS}
    assert all(summary["status"] == "complete" and summary["samples"] == 88 for summary in summaries.values())
    reference = summaries["dense_full"]["results"]
    for method, summary in summaries.items():
        assert all(a["ids"][0] == b["ids"][0] for a, b in zip(reference, summary["results"], strict=True)), method
    tasks = list(summaries["dense_full"]["tasks"])
    combined = {
        "status": "complete",
        "samples_per_method": 88,
        "methods": list(METHODS),
        "means": {method: summary["mean"] for method, summary in summaries.items()},
        "tasks": {task: {method: summary["tasks"][task] for method, summary in summaries.items()} for task in tasks},
        "first_token_agreement": True,
    }
    common.write_json(args.output / "summary.json", combined)
    args.repo_output.mkdir(parents=True, exist_ok=True)
    csv_path = args.repo_output / "section4_ruler88.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["task", *METHODS])
        for task in tasks:
            writer.writerow([task, *[combined["tasks"][task][method] for method in METHODS]])
        writer.writerow(["unweighted_11_task_mean", *[combined["means"][method] for method in METHODS]])
    lines = [
        "# Section 4 Dense-V RULER-64K",
        "",
        "11 tasks × 8 paired prompts. Dense V128 and original W_O for every arm.",
        "",
        "| Task | " + " | ".join(METHODS) + " |",
        "|---|" + "---:|" * len(METHODS),
    ]
    for task in [*tasks, "Mean"]:
        values = combined["means"] if task == "Mean" else combined["tasks"][task]
        lines.append("| " + task + " | " + " | ".join(f"{values[method]:.4f}" for method in METHODS) + " |")
    (args.repo_output / "section4_ruler88.md").write_text("\n".join(lines) + "\n")
    print(combined["means"], flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=("smoke", "evaluate", "summarize", "aggregate"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--section4", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-output", type=Path, default=Path("results/section4_ablation"))
    parser.add_argument("--method", choices=METHODS, default="dense_full")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=4)
    args = parser.parse_args()
    common.configure()
    assert 0 <= args.shard_index < args.num_shards
    if args.stage == "smoke":
        run(args, smoke=True)
    elif args.stage == "evaluate":
        gate = common.read_json(args.output / args.method / "smoke_audit.json")
        assert gate["status"] == "complete"
        run(args, smoke=False)
    elif args.stage == "summarize":
        summarize_method(args)
    else:
        aggregate(args)


if __name__ == "__main__":
    main()
