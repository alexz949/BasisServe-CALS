"""Paired Dense-V LongBench evaluation for the Section 4 routing objectives."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import shlex
import sys
import time

from safetensors.torch import load_file
import torch
from transformers import AutoTokenizer

from evaluation import eval_k_routing_ruler as common
from evaluation.eval_longbench_c1_fourarm import official_scorer, score_prediction


METHODS = ("dense_full", "exact_k", "page_fisher", "score_mse", "score_only", "qgram_score_only")
METHOD_LABELS = {
    "dense_full": "Dense Full",
    "exact_k": "Exact-K",
    "page_fisher": "Page-Fisher",
    "score_mse": "Score-MSE",
    "score_only": "Fixed Score-only",
    "qgram_score_only": "QGram Score-only",
}
TASKS = ("qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "qmsum")
SAMPLES_PER_TASK = 32
TOTAL_SAMPLES = len(TASKS) * SAMPLES_PER_TASK
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260918


def runtime_arm(method):
    if method == "dense_full":
        return "full"
    if method == "exact_k":
        return "exact_sparse"
    return "ours"


def bank_path(root, section4, method):
    return {
        "page_fisher": root / "ours_b16r16",
        "score_mse": section4 / "objectives/score_mse/ours_b16r16",
        "score_only": section4 / "score_only/checkpoint/ours_b16r16",
        "qgram_score_only": section4 / "qgram_score_only/checkpoint/ours_b16r16",
    }[method]


def load_bank(identity_path, checkpoint_manifest, root, section4, method):
    if method not in ("page_fisher", "score_mse", "score_only", "qgram_score_only"):
        return {}, {}
    folder = bank_path(root, section4, method)
    factors = {}
    hashes = {}
    for record in checkpoint_manifest["layers"]:
        layer = record["layer"]
        path = folder / f"layer_{layer:03d}.safetensors"
        audit = common.read_json(path.with_suffix(".json"))
        assert audit["status"] == "complete" and audit["layer"] == layer
        assert audit["v_rank"] == record["ranks"][0] == 128
        assert audit["identity_sha256"] == common.sha256(identity_path)
        assert audit["protocol"]["sequence_length"] == 65536
        assert audit["protocol"]["fit_queries"] == 64
        assert audit["protocol"].get("diagnostic_queries", audit["protocol"].get("heldout_queries")) == 32
        assert audit["protocol"]["fit_ids"] == list(range(64))
        assert audit["protocol"].get("diagnostic_ids", audit["protocol"].get("heldout_ids")) == list(range(64, 80))
        assert audit["sweeps"] == 40 and audit["pcg_iterations"] == 100
        assert audit["sha256"] == common.sha256(path)
        factors[layer] = load_file(str(path))
        assert all(value.dtype == torch.float32 and torch.isfinite(value).all() for value in factors[layer].values())
        hashes[str(layer)] = audit["sha256"]
    return factors, hashes


def inputs(args):
    identity_path = args.root / "manifests/v128.json"
    identity = common.read_json(identity_path)
    checkpoint = Path(identity["checkpoint"])
    checkpoint_manifest = common.read_json(checkpoint / "manifest.json")
    assert common.sha256(checkpoint / "manifest.json") == identity["manifest_sha256"]
    assert common.sha256(Path(identity["model"]) / "config.json") == identity["model_config_sha256"]
    for record in checkpoint_manifest["layers"]:
        assert common.sha256(checkpoint / record["file"]) == record["sha256"]
    dataset = common.read_json(args.data / "manifest.json")
    dataset_protocol = dataset["protocol"]
    assert dataset["status"] == "complete"
    assert dataset_protocol["tasks"] == list(TASKS)
    assert dataset_protocol["samples_per_task"] == SAMPLES_PER_TASK
    assert dataset_protocol["sequence_length"] == 32768
    assert dataset_protocol["model_config_sha256"] == identity["model_config_sha256"]
    assert common.sha256(args.data / "tokens.safetensors") == dataset["tokens_sha256"]
    assert common.sha256(args.data / "samples.json") == dataset["samples_sha256"]
    rows = common.read_json(args.data / "samples.json")
    tokens = load_file(str(args.data / "tokens.safetensors"))
    assert len(rows) == len(tokens) == TOTAL_SAMPLES
    for index, row in enumerate(rows):
        tensor = tokens[f"sample_{index:03d}"]
        assert row["index"] == index and len(tensor) == row["prompt_tokens"]
        assert len(tensor) + row["maximum_tokens"] <= 32768
        assert hashlib.sha256(tensor.numpy().tobytes()).hexdigest() == row["input_ids_sha256"]
    official = Path(dataset_protocol["official_root"])
    for name, digest in dataset_protocol["official_sha256"].items():
        assert common.sha256(official / "LongBench" / name) == digest
    factors, bank_hashes = load_bank(identity_path, checkpoint_manifest, args.root, args.section4, args.method)
    runtime_config = common.routing_config(identity, rope="native", sequence_length=32768)
    protocol = {
        "format": "basisserve.section4.llama_longbench.v1",
        "method": args.method,
        "runtime_arm": runtime_arm(args.method),
        "identity_sha256": common.sha256(identity_path),
        "dataset_manifest_sha256": common.sha256(args.data / "manifest.json"),
        "dataset_protocol": dataset_protocol,
        "bank_sha256": bank_hashes,
        "runtime_config": runtime_config.to_dict(),
        "dtype": "bfloat16",
        "sequence_length": 32768,
        "value_mode": "dense original V128 and Wo",
        "prefill": "shared full exact attention through the same installed runtime",
        "decode": "full attention, exact-K Page32 oracle, or B16R16 proxy routing",
        "budget": {
            "tokens": 2048,
            "page_size": 32,
            "sink_tokens_inside_budget": 32,
            "recent_tokens_inside_budget": 64,
        },
        "final_attention": "exact selected post-RoPE K and exact QK; proxy selects support only",
        "generation": "greedy, native EOS, official task caps",
        "metrics": "official LongBench-v1 QA F1 or ROUGE-L; maximum over alternate references",
        "source_sha256": {
            name: common.sha256(Path(name))
            for name in (
                "evaluation/eval_llama_section4_longbench.py",
                "evaluation/eval_k_routing_ruler.py",
                "evaluation/eval_longbench_c1_fourarm.py",
                "basisserve/core/c1_conditional_page_attention.py",
                "basisserve/core/c1_v_conditional_k_router.py",
                "basisserve/kernels/compressed_v_decode_attention.py",
            )
        },
    }
    protocol = json.loads(json.dumps(protocol, allow_nan=False))
    tokenizer = AutoTokenizer.from_pretrained(identity["model"], local_files_only=True)
    return tokenizer, identity, checkpoint_manifest, rows, tokens, factors, bank_hashes, protocol, official_scorer(official)


def verify(saved, row, protocol, bank_hashes, scorer):
    assert saved["status"] == "complete" and saved["sample"] == row
    assert saved["protocol"] == protocol and saved["bank_sha256"] == bank_hashes
    result = saved["result"]
    assert 0 < len(result["ids"]) <= row["maximum_tokens"]
    assert result["prediction"] and result["score"] == score_prediction(
        scorer, row["task"], result["prediction"], row["answers"], row["all_classes"]
    )


@torch.inference_mode()
def run(args, *, smoke):
    tokenizer, identity, manifest, rows, tokens, factors, hashes, protocol, scorer = inputs(args)
    assert torch.cuda.device_count() == 1 and "L40S" in torch.cuda.get_device_name(0)
    model = common.load_evaluation_model(
        identity,
        common.routing_config(identity, rope="native", sequence_length=32768),
    )
    arm = runtime_arm(args.method)
    common.install(model, Path(identity["checkpoint"]), manifest, arm, factors, dense_v=True)
    selected = [min(rows, key=lambda row: row["prompt_tokens"]), max(rows, key=lambda row: row["prompt_tokens"])] if smoke else rows[args.shard_index::args.num_shards]
    stage = "smoke" if smoke else "evaluate"
    for row in selected:
        path = args.output / args.method / stage / f"sample_{row['index']:03d}.json"
        if path.exists():
            verify(common.read_json(path), row, protocol, hashes, scorer)
            continue
        generation_row = {**row, "input_ids": tokens[f"sample_{row['index']:03d}"].tolist()}
        cap = min(4, row["maximum_tokens"]) if smoke else row["maximum_tokens"]
        torch.manual_seed(0)
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        ids, first, statistics, stopped = common.generate(model, tokenizer, generation_row, arm, cap)
        if smoke:
            again, other, _, _ = common.generate(model, tokenizer, generation_row, arm, cap)
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
                "score": score_prediction(scorer, row["task"], prediction, row["answers"], row["all_classes"]),
                "stopped": stopped,
                "routing": statistics,
                "seconds": time.monotonic() - started,
                "peak_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
            },
        }
        verify(saved, row, protocol, hashes, scorer)
        common.write_json(path, saved)
        print("COMPLETE", args.method, row["index"], row["task"], saved["result"]["score"], flush=True)
    if smoke:
        common.write_json(
            args.output / args.method / "smoke_audit.json",
            {
                "status": "complete",
                "method": args.method,
                "protocol": protocol,
                "samples": [row["index"] for row in selected],
            },
        )


def summarize_method(args):
    _, _, _, rows, _, _, hashes, protocol, scorer = inputs(args)
    gate = common.read_json(args.output / args.method / "smoke_audit.json")
    assert gate["status"] == "complete" and gate["protocol"] == protocol
    results = []
    for row in rows:
        saved = common.read_json(args.output / args.method / "evaluate" / f"sample_{row['index']:03d}.json")
        verify(saved, row, protocol, hashes, scorer)
        results.append(saved["result"])
    tasks = {}
    for task in TASKS:
        selected_rows = [row for row in rows if row["task"] == task]
        selected_results = [result for row, result in zip(rows, results, strict=True) if row["task"] == task]
        assert len(selected_results) == SAMPLES_PER_TASK
        mean = 100 * sum(result["score"] for result in selected_results) / SAMPLES_PER_TASK
        official = scorer.scorer(
            task,
            [result["prediction"] for result in selected_results],
            [row["answers"] for row in selected_rows],
            selected_rows[0]["all_classes"],
        )
        assert round(mean, 2) == official
        tasks[task] = mean
    summary = {
        "status": "complete",
        "method": args.method,
        "protocol": protocol,
        "samples": TOTAL_SAMPLES,
        "mean": sum(tasks.values()) / len(tasks),
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
        task_means = []
        for task_index in range(len(TASKS)):
            start = task_index * SAMPLES_PER_TASK
            group = differences[start : start + SAMPLES_PER_TASK]
            task_means.append(sum(group[generator.randrange(SAMPLES_PER_TASK)] for _ in range(SAMPLES_PER_TASK)) / SAMPLES_PER_TASK)
        estimates.append(100 * sum(task_means) / len(task_means))
    estimates.sort()
    return [
        estimates[int(0.025 * (BOOTSTRAP_REPLICATES - 1))],
        estimates[int(0.975 * (BOOTSTRAP_REPLICATES - 1))],
    ]


def paired_record(reference, candidate):
    differences = [
        right["score"] - left["score"]
        for left, right in zip(reference["results"], candidate["results"], strict=True)
    ]
    return {
        "wins": sum(value > 0 for value in differences),
        "losses": sum(value < 0 for value in differences),
        "ties": sum(value == 0 for value in differences),
        "macro_mean_delta": candidate["mean"] - reference["mean"],
        "stratified_paired_bootstrap_95_ci": stratified_bootstrap(differences),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }


def aggregate(args):
    methods = tuple(method for method in METHODS if (args.output / method / "summary.json").is_file())
    assert methods[:3] == ("dense_full", "exact_k", "page_fisher")
    summaries = {method: common.read_json(args.output / method / "summary.json") for method in methods}
    assert all(summary["status"] == "complete" and summary["samples"] == TOTAL_SAMPLES for summary in summaries.values())
    dense = summaries["dense_full"]
    for method in methods[1:]:
        assert all(
            left["ids"][0] == right["ids"][0]
            for left, right in zip(dense["results"], summaries[method]["results"], strict=True)
        )
    paired = {f"{method}_vs_dense_full": paired_record(dense, summaries[method]) for method in methods[1:]}
    for method in methods[3:]:
        paired[f"{method}_vs_page_fisher"] = paired_record(summaries["page_fisher"], summaries[method])
    combined = {
        "status": "complete",
        "samples_per_method": TOTAL_SAMPLES,
        "methods": list(methods),
        "means": {method: summaries[method]["mean"] for method in methods},
        "tasks": {
            task: {method: summaries[method]["tasks"][task] for method in methods}
            for task in TASKS
        },
        "paired": paired,
        "first_token_agreement": True,
    }
    common.write_json(args.output / "summary.json", combined)
    args.repo_output.mkdir(parents=True, exist_ok=True)
    with (args.repo_output / "section4_longbench192.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["task", *methods])
        for task in TASKS:
            writer.writerow([task, *[combined["tasks"][task][method] for method in methods]])
        writer.writerow(["six_task_macro_mean", *[combined["means"][method] for method in methods]])
    lines = [
        "# Section 4 Dense-V LongBench-32K",
        "",
        "Six tasks × 32 paired seed-43 prompts. Llama-3.1-8B-Instruct; Dense V128; B2048; Page32; sink32/recent64.",
        "",
        "| Task | " + " | ".join(METHOD_LABELS[method] for method in methods) + " |",
        "|---|" + "---:|" * len(methods),
    ]
    for task in TASKS:
        values = combined["tasks"][task]
        lines.append("| " + task + " | " + " | ".join(f"{values[method]:.4f}" for method in methods) + " |")
    lines.append("| Mean | " + " | ".join(f"{combined['means'][method]:.4f}" for method in methods) + " |")
    lines += ["", "## Paired comparisons", "", "```json", json.dumps(paired, indent=2), "```"]
    (args.repo_output / "section4_longbench192.md").write_text("\n".join(lines) + "\n")
    print(combined, flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=("smoke", "evaluate", "summarize", "aggregate"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--section4", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
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
