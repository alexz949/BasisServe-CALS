"""Audit completed TP1 records and frozen sources without content hashes."""

import json
import math
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "results/system_benchmarks/tp1_offline"


def main():
    outcomes = json.loads((OUTPUT / "formal_outcomes.json").read_text())
    assert len(outcomes) == 78
    keys = {(t["mode"], t["method"], t["context"], t["cohort"], t["output_tokens"]) for t in outcomes}
    assert len(keys) == 78
    successes = 0
    primary_compile_logs = []
    for trial in outcomes:
        folder = Path(trial["directory"])
        if trial["status"] == "gpu_oom":
            assert "out of memory" in (folder / "run.log").read_text().lower()
            continue
        assert trial["status"] == "complete"
        result = json.loads((folder / "result.json").read_text())
        assert result["status"] == "complete" and result["method"] == trial["method"]
        assert result["context_tokens"] == trial["context"] and result["cohort"] == trial["cohort"]
        measured = result["measurement"]
        if trial["mode"] == "request":
            assert measured["actual_decode_calls"] == trial["output_tokens"]-1
            assert len(measured["generated_tokens"][0]) == trial["output_tokens"]
            assert measured["final_logits_finite"]
            phases = [measured[k] for k in ("prefill_inclusive_seconds", "post_prefill_phase_seconds", "decode_phase_seconds")]
            assert all(math.isfinite(x) and x >= 0 for x in phases)
            assert abs(sum(phases)-measured["request_seconds"]) < 1e-8
            assert 0 <= measured["post_prefill_ready_seconds"] <= sum(phases[1:])
            assert measured["steady_tail_decode_steps"] == trial["output_tokens"]-17
            if trial["method"] in ("dense", "basis"):
                assert measured["post_prefill_ready_seconds"] == 0
                assert result["build_profile"]["prompt_specific_fit_seconds"] == 0
            elif trial["output_tokens"] == 128:
                assert result["build_profile"]["counts_match"]
                assert result["build_profile"]["first_two_tokens_match"]
        else:
            assert len(measured["cuda_ms"]) == 128 and len(measured["wall_ms"]) == 128
            assert len(measured["generated_tokens"][0]) == 144
            assert measured["all_logits_finite"]
            profile = result["attention_profile"]
            assert profile["prefix_tokens_match"] and profile["all_logits_finite"]
            assert len(profile["components"]) == 32
            assert all(r["attention_block"] > 0 for r in profile["components"])
        log = (folder / "run.log").read_text()
        marker = '"phase": "measured_' + trial["mode"] + '"'
        primary = log.split(marker, 1)[1].split('"phase": "separate_', 1)[0]
        if "AUTOTUNE" in primary or "recompile_limit" in primary:
            primary_compile_logs.append(str(folder / "run.log"))
        successes += 1
    source_checks = 0
    for path in (OUTPUT / "formal/source").glob("*.py"):
        assert path.read_bytes() == (ROOT / "benchmarks/system" / path.name).read_bytes()
        source_checks += 1
    with tarfile.open(OUTPUT / "freeze/dependencies.tar.gz") as archive:
        for member in archive.getmembers():
            if member.isfile():
                assert archive.extractfile(member).read() == (ROOT / member.name).read_bytes()
                source_checks += 1
    factor_checks = 0
    with tarfile.open(OUTPUT / "freeze/factors.tar.gz") as archive:
        for member in archive.getmembers():
            if member.isfile():
                path = Path("/workspace/runs/l31-router-source/v128-router") / member.name
                assert archive.extractfile(member).read() == path.read_bytes()
                factor_checks += 1
    result = dict(status="passed", trials=78, completed=successes,
                  gpu_oom=78-successes, source_files_unchanged=source_checks,
                  factor_files_unchanged=factor_checks,
                  primary_timing_compile_warning_logs=primary_compile_logs,
                  comparison="direct bytes, no SHA256 calculation",
                  warning="Absence of compile warnings is not a profiler proof that every primary call was compilation-free.")
    (OUTPUT / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
