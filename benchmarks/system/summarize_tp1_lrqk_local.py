"""Validate and compare the GPU-local supplement with the frozen CPU baseline."""

import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/system_benchmarks/tp1_lrqk_local"
OLD = ROOT / "results/system_benchmarks/tp1_offline"


def main():
    rows = json.loads((OUT / "outcomes.json").read_text())
    assert len(rows) == 15
    assert len({(r["context"],r["output_tokens"],r["cohort"]) for r in rows}) == 15
    warnings = []
    complete = []
    for row in rows:
        folder = Path(row["directory"])
        log = (folder/"run.log").read_text()
        if row["status"] == "gpu_oom":
            assert "out of memory" in log.lower()
            continue
        assert row["status"] == "complete"
        data = json.loads((folder/"result.json").read_text())
        measured = data["measurement"]
        assert data["configuration"]["storage"] == "GPU-local exact K/V"
        assert measured["final_logits_finite"]
        assert measured["actual_decode_calls"] == row["output_tokens"]-1
        assert len(measured["generated_tokens"][0]) == row["output_tokens"]
        assert measured["post_prefill_ready_seconds"] == 0
        assert not measured["ready_overlaps_first_decode"]
        assert abs(measured["request_seconds"]-sum(measured[k] for k in (
            "prefill_inclusive_seconds","post_prefill_phase_seconds","decode_phase_seconds"))) < 1e-8
        if row["output_tokens"] == 128:
            assert data["build_profile"]["counts_match"] and data["build_profile"]["first_two_tokens_match"]
        primary = log.split('"phase": "measured_request"',1)[1].split('"phase": "separate_',1)[0]
        if "AUTOTUNE" in primary or "recompile_limit" in primary:
            warnings.append(str(folder/"run.log"))
        complete.append(data)
    for path in (OUT/"source").glob("*.py"):
        assert path.read_bytes() == (ROOT/"benchmarks/system"/path.name).read_bytes()
    validation = dict(status="passed",trials=15,complete=len(complete),gpu_oom=15-len(complete),
        primary_compile_warning_logs=warnings,source_comparison="direct bytes, no hashes")
    (OUT/"verification.json").write_text(json.dumps(validation,indent=2)+"\n")
    baseline = json.loads((OLD/"formal_outcomes.json").read_text())
    token_checks = []
    for data in complete:
        matches = [x for x in baseline if x["method"] == "lrqk" and x["context"] == data["context_tokens"]
            and x["output_tokens"] == data["output_tokens"] and x["cohort"] == data["cohort"] and x["status"] == "complete"]
        if matches:
            old = json.loads((Path(matches[0]["directory"])/"result.json").read_text())
            token_checks.append(dict(context=data["context_tokens"],cohort=data["cohort"],
                output_tokens=data["output_tokens"],all_tokens_match=data["measurement"]["generated_tokens"] == old["measurement"]["generated_tokens"]))
    validation["paired_generation_checks"] = token_checks
    (OUT/"verification.json").write_text(json.dumps(validation,indent=2)+"\n")
    lines = ["# LRQK GPU-local Comparison", "", f"15 formal trials: {len(complete)} complete, {15-len(complete)} GPU OOM. Record/source checks passed.",
        "Three-cohort medians; 128 output tokens. Decode is the request tail wall mean, including greedy selection.", "",
        "| Context | CPU-offload request s | GPU-local request s | CPU-offload decode ms | GPU-local decode ms | GPU-local build s |",
        "|---:|---:|---:|---:|---:|---:|"]
    for length in (32768,65536,130048):
        old = [json.loads((Path(x["directory"])/"result.json").read_text()) for x in baseline
            if x["method"] == "lrqk" and x["context"] == length and x["output_tokens"] == 128 and x["status"] == "complete"]
        local = [d for d in complete if d["context_tokens"] == length and d["output_tokens"] == 128]
        def metric(group,key):
            return f"{statistics.median(d['measurement'][key] for d in group):.3f}" if len(group) == 3 else f"{len(group)}/3 complete"
        build = f"{statistics.median(d['build_profile']['construction_including_preparation_seconds'] for d in local):.3f}" if len(local) == 3 else "-"
        lines.append(f"| {length} | {metric(old,'request_seconds')} | {metric(local,'request_seconds')} | {metric(old,'steady_tail_wall_mean_ms')} | {metric(local,'steady_tail_wall_mean_ms')} | {build} |")
    lines += ["", "GPU-local changes only storage/placement and equivalent gather: exact K/V and routing state remain on GPU; fitting, rank, support and hit/miss policy remain unchanged.",
        "70 synthetic decode steps matched CPU/GPU selected indices, selected K/V, routing factors and full cache exactly, including lite-buffer turnover. Full-model 8K smoke matched generated tokens.",
        f"Paired full-request generation checks: {sum(x['all_tokens_match'] for x in token_checks)}/{len(token_checks)} matched all generated token IDs. Details are in verification.json.",
        "GPU-local retains the upstream 1.5x allocation policy and long-context fit intermediates; OOM is not an algorithmic capacity claim.",
        "In this run, 64K failures occur at prefill MLP temporary allocation; approximately 128K failures occur inside online fitting. Both happen during warmup, before primary request timing. No MLP or allocation-policy optimization was introduced.",
        "This storage control does not remove LRQK's prompt-specific fit or its decode-time factor updates.",
        "No quality equivalence is claimed from short correctness checks. Compiler warnings remain in logs; absence during primary timing is not proof that every call was compilation-free.",
        "See [SUMMARY.md](SUMMARY.md) for all configurations, reproduction command and failure phases.", ""]
    (OUT/"COMPARISON.md").write_text("\n".join(lines))
    print(json.dumps(validation))


if __name__ == "__main__":
    main()
