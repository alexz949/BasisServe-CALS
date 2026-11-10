"""Compare matching real-input vLLM trials before and after padding elimination."""

import json
from pathlib import Path
import statistics


def read(path):
    return json.loads(path.read_text())


def requests(folder):
    return {(r["trial"], r["request_id"]): r for line in (folder / "requests.jsonl").read_text().splitlines()
            for r in [json.loads(line)]}


def main():
    root = Path("results/system_benchmarks/tp4_router_opt")
    rows = []
    for batch in (32, 128):
        old = Path(f"results/system_benchmarks/vllm_tp4/p4096_b{batch}/basis_joint/decode")
        new = root / f"p4096_b{batch}"
        old_config, new_config = read(old / "config.json"), read(new / "config.json")
        assert old_config["status"] == new_config["status"] == "complete"
        for key in ("tp", "prompt_tokens", "batch", "generation_tokens", "conditioning_steps",
                    "measured_decode_steps", "prompt_cohorts", "software", "factor_manifest_sha256",
                    "router_sha256", "visible_gpus", "host"):
            assert old_config[key] == new_config[key], key
        assert (old / "topology.txt").read_bytes() == (new / "topology.txt").read_bytes()
        a, b = requests(old), requests(new)
        assert a.keys() == b.keys()
        assert all(a[k]["prompt"] == b[k]["prompt"] for k in a)
        equal = sum(a[k]["token_ids"] == b[k]["token_ids"] for k in a)
        old_trials = read(old / "summary.json")["trials"]
        new_trials = read(new / "summary.json")["trials"]
        old_by_cohort = {r["prompt_cohort_sha256"]: r for r in old_trials}
        speedups = [old_by_cohort[r["prompt_cohort_sha256"]]["decode_step_mean_ms"] /
                    r["decode_step_mean_ms"] for r in new_trials]
        rows.append(dict(batch=batch, baseline_ms=[t["decode_step_mean_ms"] for t in old_trials],
                         optimized_ms=[t["decode_step_mean_ms"] for t in new_trials],
                         paired_speedup_median=statistics.median(speedups),
                         identical_generated_requests=equal, total_requests=len(a),
                         generated_tokens_identical=equal == len(a)))
    data = dict(scope="TP4 full-model decode, P4096, CUDA Graph, three matched real-text cohorts",
                gpu_placement="node6 GPU2,3,4,5; cross-NUMA; NCCL_P2P_DISABLE=1",
                conda_environment="/home/yl1139/basis_artifacts/envs/vllm/basis", cases=rows)
    (root / "comparison.json").write_text(json.dumps(data, indent=2) + "\n")
    lines = ["# TP4 routing: skip unused fine candidates", "",
             "## Change", "",
             "Fine routing launches min(512, capacity_pages) candidate CTAs per request/KV group. "
             "Empty candidates write -inf without executing matrix products; unlaunched columns "
             "are initialized to -inf. Valid candidate arithmetic, selection, the 2048-token cap "
             "and cache placement are unchanged. The original kernel is preserved in baseline_source/.", "",
             "## Full-model results", "",
             "node6 L40S TP4 on GPUs 2,3,4,5 (cross-NUMA); vLLM 0.30.0+cu129; "
             "CUDA Graph FULL_AND_PIECEWISE; NCCL_P2P_DISABLE=1; BF16; TF32 off. "
             "Each of three matched real-text cohorts uses 16 conditioning and 128 measured steps. "
             "Latencies below are medians of full-model mean wall-step latency. "
             "Speedup is the median of paired per-cohort ratios.", "",
             "| P | B | Before ms | After ms | Speedup | Identical generated requests |",
             "|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| 4096 | {row['batch']} | {statistics.median(row['baseline_ms']):.3f} | "
                     f"{statistics.median(row['optimized_ms']):.3f} | {row['paired_speedup_median']:.3f}x | "
                     f"{row['identical_generated_requests']}/{row['total_requests']} |")
    lines += ["", "## Validation and scope", "",
              "check/rank*.json contains concurrent four-rank synthetic kernel comparisons at "
              "4K B1/B32/B128 and 16K B8. New and old coarse scores, candidate IDs, fine scores, "
              "selected pages and support match exactly through five changing-query/length graph "
              "replays, including empty rows, tail-only rows and partial pages. "
              "routing_smoke/ retains independent PyTorch reference checks, exact selected attention, "
              "slot reuse and request recycling tests on all four ranks.", "",
              "This is one local optimization. No hardware SM-utilization claim is made. "
              "Synthetic single-layer GPU event times are not full-model latency. "
              "The full TP4 workload matrix and further kernel/TP tuning remain unfinished. "
              "Original baseline results are preserved under ../vllm_tp4/.", "",
              "## Environment and commands", "",
              "Conda prefix: /home/yl1139/basis_artifacts/envs/vllm/basis. "
              "GPU work runs through Slurm; see slurm-210457.log for the exact expanded commands.", "",
              "```bash",
              "bash benchmarks/system/launch_paged_cache_smoke.sh results/system_benchmarks/tp4_router_opt/check benchmarks.system.check_router_padding",
              "bash benchmarks/system/launch_paged_cache_smoke.sh results/system_benchmarks/tp4_router_opt/routing_smoke benchmarks.system.smoke_paged_joint_routing",
              "bash benchmarks/system/launch_vllm_bench.sh 4 results/system_benchmarks/tp4_router_opt/p4096_b32 basis_joint 4096 32 decode",
              "bash benchmarks/system/launch_vllm_bench.sh 4 results/system_benchmarks/tp4_router_opt/p4096_b128 basis_joint 4096 128 decode",
              "python -m benchmarks.system.summarize_router_padding",
              "```", ""]
    (root / "README.md").write_text("\n".join(lines))
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
