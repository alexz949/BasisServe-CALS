"""Paired 300-prompt hard-task RULER comparison of B16R16 objectives."""

import argparse
import csv
import json
from pathlib import Path
import random
import shlex
import sys
import time

import torch
from transformers import AutoTokenizer

from evaluation import eval_k_routing_ruler as common


METHODS = ("page_fisher", "score_mse", "score_only", "qgram_score_only")
METHOD_LABELS = {
    "page_fisher": "Page-Fisher",
    "score_mse": "Score-MSE",
    "score_only": "Fixed Score-only",
    "qgram_score_only": "QGram Score-only",
}
TASK_NAMES = ("niah_multiquery", "vt", "cwe", "fwe", "qa_1", "qa_2")
SAMPLES_PER_TASK = 50
TOTAL_SAMPLES = len(TASK_NAMES) * SAMPLES_PER_TASK
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260918


def bank_path(root, section4, method):
    return {
        "page_fisher": root / "ours_b16r16",
        "score_mse": section4 / "objectives/score_mse/ours_b16r16",
        "score_only": section4 / "score_only/checkpoint/ours_b16r16",
        "qgram_score_only": section4 / "qgram_score_only/checkpoint/ours_b16r16",
    }[method]


def inputs(args):
    identity_path = args.root / "manifests/v128.json"
    namespace = argparse.Namespace(
        identity=identity_path,
        data=args.data,
        bank=bank_path(args.root, args.section4, args.method),
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
        namespace,
        tokenizer,
        task_names=TASK_NAMES,
        samples_per_task=SAMPLES_PER_TASK,
    )
    assert len(rows) == TOTAL_SAMPLES
    protocol["section4_method"] = args.method
    protocol["section4_method_description"] = {
        "page_fisher": "affine Base16 plus Page-Fisher Residual16",
        "score_mse": "affine Base16 plus Score-MSE Residual16",
        "score_only": "affine Base16 plus clean Score-only Residual16",
        "qgram_score_only": "affine Base16 plus Query-Gram-selected Fisher-free Score-only Residual16",
    }[args.method]
    protocol["experiment"] = "independent-seed hard-task objective selection"
    protocol["value_mode"] = "dense original V and Wo"
    protocol["budget"] = {
        "tokens": 2048,
        "page_size": 32,
        "sink_tokens_inside_budget": 32,
        "recent_tokens_inside_budget": 64,
        "historical_pages": 62,
        "freely_routed_historical_pages": 61,
    }
    protocol["final_attention"] = "exact selected post-RoPE K and exact QK; proxy selects support only"
    protocol["source_sha256"]["evaluation/eval_llama_section4_hard_ruler.py"] = common.sha256(Path(__file__))
    return tokenizer, identity, manifest, rows, factors, hashes, protocol


def verify(saved, row, protocol, hashes):
    assert saved["status"] == "complete" and saved["sample"] == row
    assert saved["protocol"] == protocol and saved["bank_sha256"] == hashes
    result = saved["result"]
    assert 0 < len(result["ids"]) <= row["maximum_tokens"]
    assert common.sample_score(result["prediction"], row["answers"], row["match_type"]) == result["score"]


@torch.inference_mode()
def run(args, *, smoke):
    tokenizer, identity, manifest, rows, factors, hashes, protocol = inputs(args)
    assert torch.cuda.device_count() == 1 and "L40S" in torch.cuda.get_device_name(0)
    model = common.load_evaluation_model(
        identity,
        common.routing_config(identity, rope="native", sequence_length=65536),
    )
    common.install(
        model,
        Path(identity["checkpoint"]),
        manifest,
        "ours",
        factors,
        dense_v=True,
    )
    selected = [rows[0], rows[-SAMPLES_PER_TASK]] if smoke else rows[args.shard_index::args.num_shards]
    stage = "smoke" if smoke else "evaluate"
    for row in selected:
        path = args.output / args.method / stage / f"sample_{row['index']:03d}.json"
        if path.exists():
            verify(common.read_json(path), row, protocol, hashes)
            continue
        cap = min(4, row["maximum_tokens"]) if smoke else row["maximum_tokens"]
        torch.manual_seed(0)
        started = time.monotonic()
        ids, first, statistics, stopped = common.generate(model, tokenizer, row, "ours", cap)
        if smoke:
            again, other, _, _ = common.generate(model, tokenizer, row, "ours", cap)
            assert ids == again
            torch.testing.assert_close(first, other, atol=0, rtol=0)
        prediction = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        saved = {
            "status": "complete",
            "sample": row,
            "protocol": protocol,
            "bank_sha256": hashes,
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
        verify(saved, row, protocol, hashes)
        common.write_json(path, saved)
        print("COMPLETE", args.method, row["index"], row["task"], saved["result"]["score"], flush=True)
    if smoke:
        common.write_json(
            args.output / args.method / "smoke_audit.json",
            {
                "status": "complete",
                "method": args.method,
                "protocol": protocol,
                "samples": [rows[0]["index"], rows[-SAMPLES_PER_TASK]["index"]],
            },
        )


def summarize_method(args):
    _, _, _, rows, _, hashes, current_protocol = inputs(args)
    gate = common.read_json(args.output / args.method / "smoke_audit.json")
    assert gate["status"] == "complete" and gate["method"] == args.method
    first = common.read_json(args.output / args.method / "evaluate" / "sample_000.json")
    protocol = first["protocol"]
    frozen_settings = {key: value for key, value in protocol.items() if key != "source_sha256"}
    current_settings = {
        key: value for key, value in current_protocol.items() if key != "source_sha256"
    }
    smoke_settings = {
        key: value for key, value in gate["protocol"].items() if key != "source_sha256"
    }
    assert frozen_settings == current_settings == smoke_settings
    results = []
    for row in rows:
        saved = common.read_json(args.output / args.method / "evaluate" / f"sample_{row['index']:03d}.json")
        verify(saved, row, protocol, hashes)
        results.append(saved["result"])
    tasks = {}
    for task in TASK_NAMES:
        selected = [result["score"] for row, result in zip(rows, results, strict=True) if row["task"] == task]
        assert len(selected) == SAMPLES_PER_TASK
        tasks[task] = 100 * sum(selected) / SAMPLES_PER_TASK
    summary = {
        "status": "complete",
        "method": args.method,
        "protocol": protocol,
        "samples": TOTAL_SAMPLES,
        "mean": 100 * sum(result["score"] for result in results) / TOTAL_SAMPLES,
        "tasks": tasks,
        "results": results,
    }
    common.write_json(args.output / args.method / "summary.json", summary)
    print(args.method, summary["mean"], flush=True)


def stratified_bootstrap(differences):
    assert len(differences) == TOTAL_SAMPLES
    generator = random.Random(BOOTSTRAP_SEED)
    estimates = []
    for _ in range(BOOTSTRAP_REPLICATES):
        total = 0.0
        for task_index in range(len(TASK_NAMES)):
            start = task_index * SAMPLES_PER_TASK
            group = differences[start : start + SAMPLES_PER_TASK]
            total += sum(group[generator.randrange(SAMPLES_PER_TASK)] for _ in range(SAMPLES_PER_TASK))
        estimates.append(100 * total / TOTAL_SAMPLES)
    estimates.sort()
    return (
        estimates[int(0.025 * (BOOTSTRAP_REPLICATES - 1))],
        estimates[int(0.975 * (BOOTSTRAP_REPLICATES - 1))],
    )


def aggregate(args):
    methods = tuple(method for method in METHODS if (args.output / method / "summary.json").is_file())
    assert methods[0] == "page_fisher" and len(methods) >= 2
    summaries = {method: common.read_json(args.output / method / "summary.json") for method in methods}
    assert all(summary["status"] == "complete" and summary["samples"] == TOTAL_SAMPLES for summary in summaries.values())
    page = summaries["page_fisher"]
    assert all(list(summary["tasks"]) == list(TASK_NAMES) for summary in summaries.values())

    def paired_record(candidate_name):
        candidate = summaries[candidate_name]
        assert all(
            left["ids"][0] == right["ids"][0]
            for left, right in zip(page["results"], candidate["results"], strict=True)
        )
        differences = [
            right["score"] - left["score"]
            for left, right in zip(page["results"], candidate["results"], strict=True)
        ]
        confidence_interval = stratified_bootstrap(differences)
        return {
            "wins": sum(value > 0 for value in differences),
            "losses": sum(value < 0 for value in differences),
            "ties": sum(value == 0 for value in differences),
            "delta_candidate_minus_page_fisher": candidate["mean"] - page["mean"],
            "stratified_paired_bootstrap_95_ci": list(confidence_interval),
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "significantly_worse_than_page_fisher": confidence_interval[1] < 0,
        }

    paired = {f"{method}_vs_page_fisher": paired_record(method) for method in methods[1:]}
    combined = {
        "status": "complete",
        "samples_per_method": TOTAL_SAMPLES,
        "samples_per_task": SAMPLES_PER_TASK,
        "methods": list(methods),
        "means": {method: summaries[method]["mean"] for method in methods},
        "tasks": {
            task: {method: summaries[method]["tasks"][task] for method in methods}
            for task in TASK_NAMES
        },
        "paired": paired,
        "first_token_agreement": True,
    }
    common.write_json(args.output / "summary.json", combined)
    args.repo_output.mkdir(parents=True, exist_ok=True)
    csv_path = args.repo_output / "section4_ruler_hard300.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["task", *methods])
        for task in TASK_NAMES:
            values = combined["tasks"][task]
            writer.writerow([task, *[values[method] for method in methods]])
        writer.writerow(["six_task_mean", *[combined["means"][method] for method in methods]])
    lines = [
        "# Section 4 Hard-Task RULER-64K Objective Selection",
        "",
        "Six hard tasks × 50 independent seed-43 prompts. Dense V128; B16R16; B2048; Page32; sink32/recent64.",
        "",
        "| Task | " + " | ".join(METHOD_LABELS[method] for method in methods) + " |",
        "|---|" + "---:|" * len(methods),
    ]
    for task in TASK_NAMES:
        values = combined["tasks"][task]
        lines.append("| " + task + " | " + " | ".join(f"{values[method]:.4f}" for method in methods) + " |")
    lines += [
        "| Mean | " + " | ".join(f"{combined['means'][method]:.4f}" for method in methods) + " |",
        "",
        "## Paired comparisons against Page-Fisher",
        "",
        "```json",
        json.dumps(paired, indent=2),
        "```",
    ]
    (args.repo_output / "section4_ruler_hard300.md").write_text("\n".join(lines) + "\n")
    print(combined, flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=("smoke", "evaluate", "summarize", "aggregate"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--section4", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-output", type=Path, default=Path("results/section4_ablation"))
    parser.add_argument("--method", choices=METHODS, default="page_fisher")
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
