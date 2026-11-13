"""Tables for the PaLU G-LRD4 TP4/TP8 grid produced by run_palu_g4_grid.sh.

Reads every cell's config.json, summary.json, request_trials.jsonl and memory_by_rank.jsonl.
Memory is reported two ways because vLLM pre-allocates its KV pool up to the utilization
limit: bytes per token per rank from the deployed cache layout (the quantity the paper's
Table 4 talks about), and the pool capacity in tokens that this layout gives on the same GPU.
"""

import argparse
import csv
import json
from pathlib import Path
import statistics

GIB = 2 ** 30
LAYERS, KV_HEADS, HEAD_DIM = 32, 8, 128
ORDER = ["dense", "basis_v96", "basis_v64", "palu_g4_r96", "palu_g4_r64",
         "palu_g4_sharded_r96", "palu_g4_sharded_r64"]


def label(config):
    arm, factors = config["arm"], Path(config["configuration"]["hf_overrides"].get(
        "basisserve_palu_factor_dir", config["configuration"]["hf_overrides"].get(
            "basisserve_joint_factor_dir", ""))).name
    rank = "r96" if "R96" in factors or "v96" in factors else "r64" if "R64" in factors or "v64" in factors else ""
    return {"dense": "dense", "als_full": f"basis_{rank.replace('r', 'v')}", "palu_g4": f"palu_g4_{rank}",
            "palu_g4_sharded": f"palu_g4_sharded_{rank}"}[arm]


def per_token_bytes(name, deployment, tp):
    """Per layer, per rank: Keys plus Value state in BF16."""
    local_heads = KV_HEADS // tp
    keys = local_heads * HEAD_DIM * 2
    if name == "dense":
        values = [local_heads * HEAD_DIM * 2] * LAYERS
    elif name.startswith("basis"):
        values = [local_heads * int(name[-2:]) * 2] * LAYERS
    else:
        values = list(deployment["latent_bytes_per_token"])
    return keys, values


def read_cell(path):
    config = json.loads((path / "config.json").read_text())
    summary = json.loads((path / "summary.json").read_text())
    trials = [json.loads(line) for line in (path / "request_trials.jsonl").read_text().splitlines()]
    memory = [json.loads(line) for line in (path / "memory_by_rank.jsonl").read_text().splitlines()]
    phases = {record["phase"]: record["ranks"] for record in memory}
    end = phases.get("request_end") or phases.get("decode_end") or memory[-1]["ranks"]
    start = phases.get("decode_ready") or phases.get("request_start") or memory[0]["ranks"]
    rank0 = end[0]
    name = label(config)
    tp = config["configuration"]["tensor_parallel_size"]
    keys, values = per_token_bytes(name, rank0.get("deployment") or {}, tp)
    per_token = sum(keys + value for value in values)
    tokens = config["batch"] * (config["prompt_tokens"] + config["generation_tokens"])
    layer_bytes = rank0.get("gpu_cache_layer_bytes") or []
    capacity = min(layer / (keys + value) for layer, value in zip(layer_bytes, values)) if layer_bytes else None
    row = dict(arm=name, tp=tp, graph=config.get("graph", "full"), experiment=config["experiment"], batch=config["batch"],
        status=summary["trials"][-1]["status"] if summary.get("trials") else config.get("status"),
        repetitions=len(trials), e2e_s=summary["medians"].get("cohort_e2e_s"),
        ttft_p50_ms=summary["medians"].get("ttft_p50_ms"), tpot_p50_ms=summary["medians"].get("tpot_p50_ms"),
        output_tokens_s=summary["medians"].get("cohort_output_tokens_s"),
        decode_step_p50_ms=summary["medians"].get("decode_step_p50_ms"),
        decode_step_mean_ms=summary["medians"].get("decode_step_mean_ms"),
        decode_tokens_s=summary["medians"].get("decode_tokens_s"),
        decode_step_p50_ms_each=[t.get("decode_step_p50_ms") for t in trials],
        e2e_s_each=[t.get("cohort_e2e_s") for t in trials],
        key_bytes_per_token_rank=keys * LAYERS, value_bytes_per_token_rank=sum(values),
        kv_bytes_per_token_rank=per_token, mean_value_width_per_rank=sum(values) / LAYERS / 2,
        workload_kv_gib_per_rank=tokens * per_token / GIB,
        workload_value_gib_all_ranks=tokens * sum(
            sum(per_token_bytes(name, rank.get("deployment") or {}, tp)[1]) for rank in end) / GIB,
        pool_gib_per_rank=rank0["gpu_cache_pool_bytes"] / GIB, pool_capacity_tokens=capacity,
        peak_allocated_gib_max_rank=max(r["peak_allocated_bytes"] for r in end) / GIB,
        peak_reserved_gib_max_rank=max(r["peak_reserved_bytes"] for r in end) / GIB,
        device_used_gib_max_rank=max(r["nvml_sampled_device_bytes"] for r in end) / GIB,
        model_gib_per_rank=(rank0["model_parameter_storage_bytes"] + rank0["model_buffer_storage_bytes"]) / GIB,
        attention_impl=rank0.get("attention_impl"))
    deployment_end, deployment_start = rank0.get("deployment") or {}, start[0].get("deployment") or {}
    if "group_traffic_bytes_since_load" in deployment_end:
        delta = {k: deployment_end["group_traffic_bytes_since_load"][k] - deployment_start.get(
            "group_traffic_bytes_since_load", {}).get(k, 0) for k in deployment_end["group_traffic_bytes_since_load"]}
        steps = max(delta["decode_steps"] / LAYERS, 1)
        row.update(group_decode_weight_mib_per_step_per_rank=delta["decode_weight_bytes"] / steps / 2 ** 20,
                   group_decode_partial_mib_per_step_per_rank=delta["decode_partial_bytes"] / steps / 2 ** 20,
                   group_prefill_shard_mib_total_per_rank=delta["prefill_shard_bytes"] / 2 ** 20)
    return row


def number(value, digits=2):
    return "-" if value is None else f"{value:.{digits}f}"


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    return "\n".join(lines + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True, help="phase/graph directory of the grid")
    args = parser.parse_args()
    cells = sorted(p.parent for p in args.root.glob("**/summary.json"))
    rows = [read_cell(cell) for cell in cells]
    tp_sizes = {row["tp"] for row in rows}
    assert len(tp_sizes) == 1, "Summarize each TP size separately"
    tp = tp_sizes.pop()
    rows.sort(key=lambda r: (r["experiment"], r["batch"], ORDER.index(r["arm"]) if r["arm"] in ORDER else 99))
    with (args.root / "cells.csv").open("w", newline="") as handle:
        columns = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("arm", "graph", "experiment", "batch"), k))
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    text = [f"# PaLU G-LRD4 TP{tp} grid: {args.root}", "",
            f"Llama-3.1-8B-Instruct BF16, TP{tp}, 4096-token prompts, 128 generated tokens, greedy. Medians over "
            "the repetitions inside one process after a warmup trial. Per-token bytes come from the deployed "
            "cache layout per rank (Keys are dense in every arm); pool capacity is the number of tokens the "
            "pre-allocated KV pool holds for that layout.", ""]
    for experiment in sorted({r["experiment"] for r in rows}):
        text += [f"## {experiment}", ""]
        for batch in sorted({r["batch"] for r in rows if r["experiment"] == experiment}):
            subset = [r for r in rows if r["experiment"] == experiment and r["batch"] == batch]
            if experiment == "decode":
                header = ["Arm", "Graph", "Decode step p50 ms", "Decode step mean ms", "decode tok/s", "steps each (p50 ms)"]
                timing = lambda r: [number(r["decode_step_p50_ms"], 2), number(r["decode_step_mean_ms"], 2),
                                    number(r["decode_tokens_s"], 0),
                                    " / ".join(number(x, 1) for x in r["decode_step_p50_ms_each"])]
            else:
                header = ["Arm", "Graph", "E2E s", "TTFT p50 ms", "TPOT p50 ms", "tok/s"]
                timing = lambda r: [number(r["e2e_s"]), number(r["ttft_p50_ms"], 0), number(r["tpot_p50_ms"], 1),
                                    number(r["output_tokens_s"], 0)]
            text += [f"### batch {batch}", "",
                     table(header + ["KV B/token/rank", "mean V width/rank", "workload KV GiB/rank", "pool tokens",
                                     "peak alloc GiB"],
                           [[f"`{r['arm']}`", r["graph"], *timing(r),
                             r["kv_bytes_per_token_rank"], number(r["mean_value_width_per_rank"], 0),
                             number(r["workload_kv_gib_per_rank"]), number(r["pool_capacity_tokens"], 0),
                             number(r["peak_allocated_gib_max_rank"])] for r in subset]), ""]
            traffic = [r for r in subset if "group_decode_weight_mib_per_step_per_rank" in r]
            if traffic:
                text += ["Group exchange of the sharded arm (logical bytes sent to peers per rank, "
                         "excluding self and transport overhead):", "",
                         table(["Arm", "Weights MiB/step", "Partial outputs MiB/step", "Prefill shards MiB total"],
                               [[f"`{r['arm']}`", number(r["group_decode_weight_mib_per_step_per_rank"]),
                                 number(r["group_decode_partial_mib_per_step_per_rank"], 3),
                                 number(r["group_prefill_shard_mib_total_per_rank"])] for r in traffic]), ""]
    (args.root / "SUMMARY.md").write_text("\n".join(text) + "\n")
    print("\n".join(text))


if __name__ == "__main__":
    main()
