"""Summary of a run_tp1_128k_retest.py result root (request-mode formal runs, probes and checks):
runs.csv, summary.csv, manifest.json and SUMMARY.md.

  python benchmarks/system/summarize_tp1_request_retest.py --root results/rebuttal/v2q8_l40s_sparse/tp1_128k_retest
"""

import argparse
import csv
import json
from pathlib import Path
import statistics

from run_tp1_128k_retest import LABELS
from summarize_lrqk_tp1_retest import load, row

# (key, title, unit, digits)
METRICS = (
    ("ttft_s", "Prefill end (TTFT)", "s", 2),
    ("placement_s", "Placement after prefill (ShadowKV H2D)", "s", 2),
    ("decode_ready_s", "Decode-ready", "s", 2),
    ("request_e2e_s", "Request end to end, 128 tokens", "s", 2),
    ("request_decode_mean_ms", "Request: mean decode step (127 steps)", "ms", 2),
    ("request_gpu_resident_gib", "GPU resident, allocated", "GiB", 2),
    ("request_prefill_peak_allocated_gib", "Prefill peak allocated", "GiB", 2),
    ("request_prefill_peak_reserved_gib", "Prefill peak reserved", "GiB", 2),
    ("request_decode_peak_allocated_gib", "Decode peak allocated", "GiB", 2),
    ("request_decode_peak_reserved_gib", "Decode peak reserved", "GiB", 2),
    ("request_process_rss_gib", "Process RSS", "GiB", 2),
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    rows = sorted((row(folder) for folder in sorted((args.root / "formal").glob("*_c[0-9]"))),
                  key=lambda item: (item["arm"], item["cohort"]))
    arms = list(dict.fromkeys(item["arm"] for item in rows))
    columns = list(dict.fromkeys(key for item in rows for key in item))
    with (args.root / "runs.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for arm in arms:
        for key, title, unit, digits in METRICS:
            by_cohort = {item["cohort"]: item[key] for item in rows if item["arm"] == arm and key in item}
            values = list(by_cohort.values())
            summary.append(dict(arm=arm, metric=key, unit=unit, runs=len(values),
                                **{f"cohort_{cohort}": by_cohort.get(cohort) for cohort in (0, 1, 2)},
                                median=statistics.median(values) if values else None,
                                minimum=min(values, default=None), maximum=max(values, default=None)))
    with (args.root / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    manifests = [load(path) for path in sorted(args.root.glob("manifest_*.json"))]
    stopped = load(args.root / "stopped_arms.json")
    (args.root / "manifest.json").write_text(json.dumps(dict(
        invocations=manifests, stopped_arms=stopped,
        formal_runs=[dict(name=f"{item['arm']}_{item['mode']}_c{item['cohort']}", status=item["status"],
                          physical_gpu=item.get("physical_gpu")) for item in rows],
        deliverables=["correctness-summary.json", "manifest.json", "runs.csv", "summary.csv", "SUMMARY.md"]),
        indent=2) + "\n")

    correctness = load(args.root / "correctness-summary.json")
    latest = manifests[-1] if manifests else {}
    length = latest.get("context_tokens")
    cell = lambda value, digits: "n/a" if value is None else f"{value:.{digits}f}"
    lines = [f"# TP1 request-mode retest, {length} prompt tokens", "",
             f"Formal runs complete: {sum(item['status'] == 'complete' for item in rows)} of {len(rows)}. Request mode "
             "only (same-shape warmup request, fresh state, one measured request of 128 tokens); no steady runs, so "
             "there are no per-step percentiles and no host-memory inventory. The three cohorts are three different "
             "prompts.", ""]
    if latest:
        gpus = sorted({item.get("physical_gpu") for item in rows if item.get("physical_gpu")})
        lines += [f"- Model revision `{latest['model']['revision']}`, TP1, B1, BF16, original V128/Wo; inputs "
                  f"`{Path(latest['inputs']['path']).parent.parent.name}/inputs/prompts.safetensors` "
                  f"(sha256 `{latest['inputs']['sha256'][:16]}`).",
                  f"- Host `{latest['host']}`, {latest['gpu']['name']}, driver {latest['gpu']['driver_version']}; "
                  f"formal runs on physical GPU(s) at PCI {', '.join(gpus)}; {latest['cpu_model']}, "
                  f"{latest['cpu']['threads']} threads on {latest['cpu']['physical_cores']} physical cores of the "
                  "GPU's NUMA node.",
                  f"- conda env `{latest['software']['conda_environment']}`, PyTorch {latest['software']['torch']}, "
                  f"Transformers {latest['software']['transformers']}; repository `{latest['repository']['head'][:7]}` "
                  "plus uncommitted changes.", ""]
    lines += ["## Arms", ""] + [f"- `{arm}`: {LABELS.get(arm, arm)}. CUDA allocator setting: "
                                f"{next((item['allocator'] for item in rows if item['arm'] == arm and 'allocator' in item), 'n/a')}."
                                for arm in arms] + [""]
    lines += ["## Results", "", "Median of the cohorts, range in brackets.", "",
              "| | " + " | ".join(f"`{arm}`" for arm in arms) + " |", "|---|" + "---|" * len(arms)]
    for key, title, unit, digits in METRICS:
        cells = []
        for arm in arms:
            item = next(entry for entry in summary if entry["arm"] == arm and entry["metric"] == key)
            cells.append("n/a" if not item["runs"] else f"{cell(item['median'], digits)} {unit} "
                         f"({cell(item['minimum'], digits)} to {cell(item['maximum'], digits)}; n={item['runs']})")
        lines += [f"| {title} | " + " | ".join(cells) + " |"]
    lines += ["", "Per run:", "", "| Arm | Cohort | Status | Physical GPU | TTFT s | Placement s | Decode-ready s | E2E s | "
              "Step ms | Others on pinned CPUs, max |", "|---|---|---|---|---|---|---|---|---|---|"]
    for item in rows:
        lines += [f"| `{item['arm']}` | {item['cohort']} | {item['status']} | {item.get('physical_gpu', '')} | "
                  f"{cell(item.get('ttft_s'), 2)} | {cell(item.get('placement_s'), 2)} | "
                  f"{cell(item.get('decode_ready_s'), 2)} | {cell(item.get('request_e2e_s'), 2)} | "
                  f"{cell(item.get('request_decode_mean_ms'), 2)} | {item.get('load_pinned_busy_by_others_max')} |"]
    lines += ["", "## Checks, probes and failures", ""]
    scope = correctness.get("shadowkv_cleanup_scope")
    if scope:
        for arm, item in scope["arms"].items():
            lines += [f"- Cleanup scope (8192 tokens, call counts, not timed), `{arm}`: inside `get_svd` "
                      f"{item['calls_inside_get_svd']}; inside `H2D` {item['calls_inside_h2d']}; bindings restored "
                      f"after an exception inside `get_svd`: {item['bindings_restored_after_exception']}; passed: "
                      f"{item['passed']}."]
    order = correctness.get("lrqk_offload_order")
    if order:
        lines += [f"- LRQK stream-order check: ordered copy wrong elements "
                  f"{ {name: order['ordered'][name]['wrong'] for name in ('K', 'V', 'A_K')} }; passed: {order['passed']}."]
    for stage in sorted(key for key in correctness if key.startswith(("smoke_", "probe_"))):
        for group, block in correctness[stage].items():
            for arm, item in block["arms"].items():
                failure = item.get("failure")
                lines += [f"- {stage}, `{arm}`: {item['status']}" + (
                    f"; TTFT {item['ttft_seconds']:.2f} s, decode-ready {item['decode_ready_seconds']:.2f} s"
                    if "ttft_seconds" in item else "") + (
                    f". **Failed, kind `{failure['kind']}`, in phase `{failure['last_phase_reached']}`**: "
                    f"{failure['message'][:330]} (innermost upstream frame: "
                    f"{next((frame for frame in reversed(failure['innermost_repository_frames']) if 'external/' in frame), '')}; "
                    f"log `{failure['log']}`)" if failure else ".")]
            comparison = block.get("comparison", {})
            if comparison.get("done"):
                lines += [f"- {stage}, `{comparison['teacher_forced']}` fed the tokens of `{comparison['reference']}`: "
                          f"argmax equal at {comparison['argmax_matches']} of {comparison['positions']} positions, "
                          f"max abs log-probability difference {comparison['logprob_max_abs_difference']}."]
    for arm, item in stopped.items():
        lines += [f"- Formal run stopped for `{arm}` after `{item['run']}`: {item['kind']}; {item['message'][:300]}"]
    lines += ["", "A probe is one short run (8 generated tokens) on cohort 0; probe timings are not results.", ""]
    (args.root / "SUMMARY.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
