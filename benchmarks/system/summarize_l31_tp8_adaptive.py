"""Summary of the Llama-3.1-8B TP8 adaptive grid written by scripts/run_l31_tp8_adaptive.sh.

One row per method and batch from the formal request cells. Timing is the measured cohort:
common release to the last request's final token. Memory is reported separately as measured
peak allocation, parameter storage, the pre-allocated pool, and the effective K/V bytes that
the deployed per-layer ranks put in the cache for the same number of tokens.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

GIB = 2 ** 30
LAYERS, KV_HEADS, HEAD_DIM, BLOCK = 32, 8, 128, 32
METHODS = ["dense", "basiskv", "palu_a", "palu_b"]
LABELS = dict(dense="Dense", basiskv="BasisKV adaptive V96", palu_a="PaLU-A G-LRD4 V96 (replicated)",
              palu_b="PaLU-B G-LRD4 V96 (sharded)")
TOP1_MIN, LOGPROB_MAX = .97, .05
QUALITY = Path("ICLR-results/llama31-8b-instruct/quality")
QUALITY_RUN = dict(dense="L31-8B-I-Dense", basiskv="L31-8B-I-C1-R96", palu_a="L31-8B-I-PALUG4-R96",
                   palu_b="L31-8B-I-PALUG4-R96")


def quality(method):
    """Main-table quality result of the checkpoint a method serves."""
    result = json.loads((QUALITY / QUALITY_RUN[method] / "result.json").read_text())
    return dict(run_id=result["run_id"], manifest_sha256=result.get("checkpoint", {}).get("manifest_sha256"),
                model_revision=result["model"]["revision"], wikitext2_ppl=result["metrics"]["wikitext2_ppl"],
                c4_ppl=result["metrics"]["c4_validation_128_ppl"],
                average_accuracy=result["metrics"]["average_accuracy"],
                rank_sum=result.get("compression", {}).get("rank_sum_across_layers"))


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def validation(root, method):
    path = root / "validation" / f"compare_{method}.json"
    if not path.exists():
        return dict(status="missing")
    result = json.loads(path.read_text())
    passed = (result["same_prompts"] and result["all_finite"]
              and result["teacher_forced_top1_agreement"] >= TOP1_MIN
              and result["teacher_forced_logprob_mean_abs_difference"] < LOGPROB_MAX)
    return dict(status="pass" if passed else "FAIL", top1=result["teacher_forced_top1_agreement"],
                mean_abs_logprob=result["teacher_forced_logprob_mean_abs_difference"],
                all_finite=result["all_finite"])


def cache_layout(method, ranks, tp):
    """Per TP rank: effective and physically stored K/V bytes per token over all layers, and
    the Value ranks this rank holds per layer."""
    local_heads = KV_HEADS // tp
    rows = []
    for rank in ranks:
        deployment = rank.get("deployment") or {}
        latent = deployment.get("latent_bytes_per_token") or [local_heads * HEAD_DIM * 2] * LAYERS
        effective = LAYERS * local_heads * HEAD_DIM * 2 + sum(latent)
        # A layer's cache is [blocks, ...]; everything after the block axis is one block of 32 tokens.
        physical = sum(sum(int(np.prod(shape[1:])) * 2 for shape in layer) // BLOCK
                       for layer in rank["gpu_cache_layer_shapes"])
        rows.append(dict(effective=effective, physical=physical,
                         layer_ranks=deployment.get("layer_ranks") or [HEAD_DIM] * LAYERS,
                         latent_width=[value // 2 for value in latent]))
    return rows


def retention(method, layout, tp):
    """Distinct Value coordinates kept, over the dense 32 layers x 8 heads x 128."""
    total = sum(sum(row["layer_ranks"]) for row in layout) * (KV_HEADS // tp)
    # Every TP rank of a PaLU group reports the whole group's rank.
    if method.startswith("palu"):
        total //= tp // 2
    return total / (LAYERS * KV_HEADS * HEAD_DIM)


def phases(cell):
    """Wall time of steps that schedule prompt tokens and of steps that only decode."""
    steps = lines(cell / "scheduler_steps.jsonl")[0]["steps"]
    wall = np.load(cell / "steps_0.npz")["wall_ms"]
    result = dict(scheduler_steps=len(steps), max_active_requests=max(len(s["requests"]) for s in steps),
                  final_step_active_requests=len(steps[-1]["requests"]))
    if len(steps) == len(wall):
        prefill = np.array([any(r["is_prefill"] for r in s["requests"]) for s in steps])
        result.update(prefill_step_wall_s=float(wall[prefill].sum() / 1000),
                      decode_only_step_wall_s=float(wall[~prefill].sum() / 1000),
                      prefill_steps=int(prefill.sum()), decode_only_steps=int((~prefill).sum()))
    return result


def read_cell(root, cell):
    method, batch = cell.parent.name, int(cell.name[1:])
    config = json.loads((cell / "config.json").read_text())
    row = dict(method=method, label=LABELS[method], tp=config["tp"], batch=batch, status=config["status"],
               graph=config["graph"], nccl_p2p_disabled=config["nccl_p2p_disabled"],
               factor_manifest_sha256=None if method == "dense" else config["factor_manifest_sha256"],
               validation=validation(root, method), command=config["command"], quality=quality(method),
               software=config["software"], gpu_memory_utilization=config["configuration"]["gpu_memory_utilization"])
    if config["status"] != "complete":
        return row | dict(failure_reason=config.get("failure_reason"))
    trial = lines(cell / "request_trials.jsonl")[-1]
    requests = lines(cell / "requests.jsonl")
    memory = {record["phase"]: record["ranks"] for record in lines(cell / "memory_by_rank.jsonl")}
    ranks = memory["request_end"]
    layout = cache_layout(method, ranks, config["tp"])
    tokens = batch * (config["prompt_tokens"] + config["generation_tokens"])
    effective = max(r["effective"] for r in layout)
    row.update(
        status=trial["status"], preemptions=trial["preemptions"],
        value_retention=1. if method == "dense" else retention(method, layout, config["tp"]),
        e2e_s=trial["cohort_e2e_s"], engine_loop_s=trial["engine_loop_s"],
        output_tokens_s=trial["cohort_output_tokens_s"],
        ttft_mean_ms=trial["ttft_mean_ms"], ttft_p50_ms=trial["ttft_p50_ms"], ttft_p95_ms=trial["ttft_p95_ms"],
        tpot_mean_ms=trial["tpot_mean_ms"], tpot_p50_ms=trial["tpot_p50_ms"], tpot_p95_ms=trial["tpot_p95_ms"],
        output_tokens_per_request=sorted({len(r["token_ids"]) for r in requests}),
        prompt_cohort_sha256=trial["prompt_cohort_sha256"], attention_impl=ranks[0]["attention_impl"],
        peak_allocated_gib_max_rank=max(r["peak_allocated_bytes"] for r in ranks) / GIB,
        peak_allocated_gib_by_rank=[r["peak_allocated_bytes"] / GIB for r in ranks],
        model_parameter_gib_max_rank=max(r["model_parameter_storage_bytes"] for r in ranks) / GIB,
        kv_pool_gib_max_rank=max(r["gpu_cache_pool_bytes"] for r in ranks) / GIB,
        kv_pool_free_blocks=config["kv_capacity"]["free_blocks"],
        kv_required_blocks=config["kv_capacity"]["required_blocks"],
        effective_kv_bytes_per_token_rank=effective,
        effective_kv_bytes_per_token_by_rank=[r["effective"] for r in layout],
        physical_kv_bytes_per_token_rank=max(r["physical"] for r in layout),
        effective_kv_gib_per_rank=tokens * effective / GIB, cached_tokens=tokens,
        layer_value_ranks_rank0=layout[0]["layer_ranks"], stored_value_width_rank0=layout[0]["latent_width"],
        **phases(cell))
    return row


def number(value, digits=2):
    return "-" if value is None else f"{value:.{digits}f}"


def table(header, rows):
    text = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    return "\n".join(text + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    cells = [c for m in METHODS for c in sorted((args.root / "formal" / m).glob("b*")) if (c / "config.json").exists()]
    rows = sorted((read_cell(args.root, c) for c in cells), key=lambda r: (r["batch"], METHODS.index(r["method"])))
    dense = {r["batch"]: r.get("e2e_s") for r in rows if r["method"] == "dense" and r["status"] == "complete"}
    for row in rows:
        # A speedup is only stated for a complete cell against a complete Dense cell of the same batch.
        valid = row["status"] == "complete" and dense.get(row["batch"]) is not None
        row["e2e_speedup_vs_dense"] = dense[row["batch"]] / row["e2e_s"] if valid else None
    assert len({r["tp"] for r in rows}) <= 1 and len({r["prompt_cohort_sha256"] for r in rows if "e2e_s" in r}) <= len(dense)
    (args.root / "summary.json").write_text(json.dumps(dict(
        thresholds=dict(top1_min=TOP1_MIN, mean_abs_logprob_max=LOGPROB_MAX), rows=rows), indent=2) + "\n")
    columns = ["method", "tp", "batch", "value_retention", "e2e_s", "e2e_speedup_vs_dense", "ttft_mean_ms", "ttft_p50_ms",
               "tpot_mean_ms", "tpot_p50_ms", "output_tokens_s", "peak_allocated_gib_max_rank",
               "model_parameter_gib_max_rank", "kv_pool_gib_max_rank", "effective_kv_bytes_per_token_rank",
               "physical_kv_bytes_per_token_rank", "effective_kv_gib_per_rank", "prefill_step_wall_s",
               "decode_only_step_wall_s", "max_active_requests", "preemptions", "status", "validation_status"]
    with (args.root / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(row | dict(validation_status=row["validation"]["status"]) for row in rows)
    text = ["# Llama-3.1-8B-Instruct TP8: Dense, adaptive BasisKV, adaptive PaLU G-LRD4", "",
            "BF16, eager, 4096-token prompts, 128 generated tokens per request (EOS ignored), dense Keys and full "
            "attention in every method. One separate warmup cohort, then one measured cohort per cell; E2E is "
            "common release to the last request's 128th token.", "",
            table(["Method", "TP", "Batch", "V retention", "E2E s", "Speedup vs Dense", "TTFT mean ms", "TPOT mean ms",
                   "Output tok/s", "Peak alloc GiB/rank", "Effective KV GiB/rank", "Preemptions", "Validation"],
                  [[r["label"], r["tp"], r["batch"], number(r.get("value_retention"), 4), number(r.get("e2e_s")),
                    number(r["e2e_speedup_vs_dense"], 3), number(r.get("ttft_mean_ms"), 0),
                    number(r.get("tpot_mean_ms"), 1), number(r.get("output_tokens_s"), 0),
                    number(r.get("peak_allocated_gib_max_rank")), number(r.get("effective_kv_gib_per_rank")),
                    r.get("preemptions", "-"), r["validation"]["status"] if r["status"] == "complete"
                    else f"{r['status']}: not a valid timing"] for r in rows]), "",
            "## Memory", "",
            "Peak allocation includes the pool vLLM pre-allocates under the same 0.8 budget, so it is not the "
            "K/V data size. Effective bytes follow each layer's loaded rank on the rank that stores the most; "
            "physical bytes are the stored cache width, padding included.", "",
            table(["Method", "Batch", "Model params GiB/rank", "KV pool GiB/rank", "Effective KV B/token/rank",
                   "Physical KV B/token/rank", "Effective KV GiB/rank", "Pool blocks free / needed"],
                  [[r["label"], r["batch"], number(r["model_parameter_gib_max_rank"]), number(r["kv_pool_gib_max_rank"]),
                    r["effective_kv_bytes_per_token_rank"], r["physical_kv_bytes_per_token_rank"],
                    number(r["effective_kv_gib_per_rank"]), f"{r['kv_pool_free_blocks']} / {r['kv_required_blocks']}"]
                   for r in rows if "e2e_s" in r]), "",
            "## Phases", "",
            "Steps that schedule prompt tokens also decode already-prefilled requests, so the two columns "
            "partition the engine loop and are not pure prefill and pure decode costs.", "",
            table(["Method", "Batch", "Steps with prefill s", "Decode-only steps s", "TTFT p50 ms", "TPOT p50 ms",
                   "Max active requests"],
                  [[r["label"], r["batch"], number(r.get("prefill_step_wall_s")), number(r.get("decode_only_step_wall_s")),
                    number(r["ttft_p50_ms"], 0), number(r["tpot_p50_ms"], 1), r["max_active_requests"]]
                   for r in rows if "e2e_s" in r]), ""]
    first = {r["method"]: r for r in reversed(rows)}
    software = rows[0]["software"]
    text += ["## Checkpoints", "",
             "Each compressed method serves the main quality table's checkpoint; the manifest hash loaded by the "
             "benchmark is the one recorded with that quality result. PaLU-A and PaLU-B load the same factors.", "",
             table(["Method", "Quality run", "Manifest sha256", "Rank sum", "V retention", "WT2 PPL", "C4 PPL",
                    "Avg accuracy", "Attention impl"],
                   [[r["label"], r["quality"]["run_id"], (r["factor_manifest_sha256"] or "-")[:16],
                     r["quality"]["rank_sum"] or LAYERS * KV_HEADS * HEAD_DIM, number(r.get("value_retention"), 4),
                     number(r["quality"]["wikitext2_ppl"], 3), number(r["quality"]["c4_ppl"], 3),
                     number(r["quality"]["average_accuracy"], 4), r.get("attention_impl", "-")]
                    for r in (first[m] for m in METHODS if m in first)]), "",
             "Per-layer Value ranks loaded on TP rank 0 (BasisKV: per KV head; PaLU: per 4-head group):", ""]
    text += [f"- {first[m]['label']}: {first[m]['layer_value_ranks_rank0']}" for m in METHODS[1:]
             if m in first and "layer_value_ranks_rank0" in first[m]]
    text += ["", "## Conditions", "",
             f"- Model revision {rows[0]['quality']['model_revision']}; torch {software['torch']}, vLLM {software['vllm']}, "
             f"transformers {software['transformers']}, triton {software['triton']}; GPU memory utilization "
             f"{rows[0]['gpu_memory_utilization']}; graph mode {sorted({r['graph'] for r in rows})}.",
             f"- NCCL P2P disabled in every cell: {all(r['nccl_p2p_disabled'] for r in rows)}. Evidence for this node "
             "is in hardware/nccl_p2p_check.",
             "- Dense runs vLLM's default attention backend while the compressed methods run Triton DiffKV kernels "
             "(see Attention impl), so speed differences against Dense include the kernel implementation.",
             "- One measured cohort per cell: differences of a few percent are within what a single trial can show.",
             "- Numerical validation compares each TP8 deployment with a single-GPU reference of the same factors "
             f"(gate: top-1 >= {TOP1_MIN}, mean |logprob difference| < {LOGPROB_MAX}); see validation/.", ""]
    (args.root / "SUMMARY.md").write_text("\n".join(text) + "\n")
    print("\n".join(text))


if __name__ == "__main__":
    main()
