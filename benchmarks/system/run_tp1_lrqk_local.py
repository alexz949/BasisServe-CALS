"""Serial GPU-local LRQK supplement; preserve the original frozen grid."""

import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/system_benchmarks/tp1_lrqk_local"
OLD = ROOT / "results/system_benchmarks/tp1_offline"
BENCH = ROOT / "benchmarks/system/bench_tp1_lrqk_local.py"


def report(rows):
    (OUT / "outcomes.json").write_text(json.dumps(rows, indent=2) + "\n")
    lines = ["# LRQK GPU-local TP1 Supplement", "",
        "Llama-3.1-8B-Instruct, L40S GPU 0, TP1/B1, basis environment. Three fixed prompt cohorts.",
        "Exact K/V and routing state remain on GPU. Rank 32, active 2048, lite 64, fitting and hit/miss logic unchanged.",
        "CPU gather is replaced by GPU index_select. Upstream 1.5x allocation growth remains unchanged.",
        "Same frozen request protocol: same-shape warmup, fresh request state, greedy fixed-length output.",
        "Request includes prefill/build and N-1 decode calls; excludes model loading, tokenization and input transfer.",
        "Construction is a separate synchronized profile nested inside prefill, not additive to request latency.",
        "No post-prefill restoration is needed. This is a storage control, not a quality evaluation.", "",
        "| Context | Output | Complete | Request s | Decode tail ms/token | Build profile s | Peak allocated GiB |",
        "|---:|---:|---:|---:|---:|---:|---:|"]
    for length, count in [(32768,128),(65536,128),(130048,128),(130048,32),(130048,512)]:
        group = [x for x in rows if x["context"] == length and x["output_tokens"] == count]
        ok = [x for x in group if x["status"] == "complete"]
        cells = ["-"]*4
        if len(ok) == 3:
            data = [json.loads((Path(x["directory"])/"result.json").read_text()) for x in ok]
            cells = [f"{statistics.median(d['measurement'][k] for d in data):.3f}" for k in ("request_seconds", "steady_tail_wall_mean_ms")]
            cells.append(f"{statistics.median(d['build_profile']['construction_including_preparation_seconds'] for d in data):.3f}" if count == 128 else "-")
            cells.append(f"{statistics.median(d['peak_allocated_gib'] for d in data):.3f}")
        lines.append(f"| {length} | {count} | {len(ok)}/{len(group)} | " + " | ".join(cells) + " |")
    lines += ["", "## Failures", ""]
    lines += [f"- {x['context']} / {x['output_tokens']} outputs / cohort {x['cohort']}: {x['status']} ({x['phase']})." for x in rows if x["status"] != "complete"]
    lines += ["", "GPU OOM is specific to this runtime/allocation policy, not proof of an algorithmic limit.",
        "Original CPU-offload results are unchanged in ../tp1_offline/SUMMARY.md.",
        "Raw logs retain graph-break/recompilation warnings. No claim of optimal LRQK implementation.", "",
        "## Command", "", "```bash",
        "CUDA_VISIBLE_DEVICES=0 CUDA_HOME=/usr/local/cuda MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=8.9 /workspace/miniforge3/bin/conda run --no-capture-output -n basis python benchmarks/system/run_tp1_lrqk_local.py",
        "```", "", "No GitHub upload or commit performed.", ""]
    (OUT / "SUMMARY.md").write_text("\n".join(lines))


def trial(length, count, cohort, smoke=False):
    folder = OUT / ("smoke" if smoke else f"t{length}_n{count}_c{cohort}")
    folder.mkdir(parents=True, exist_ok=True)
    saved = folder / "outcome.json"
    if saved.exists():
        return json.loads(saved.read_text())
    command = [sys.executable, str(BENCH), "--method", "lrqk", "--mode", "request",
        "--length", str(length), "--cohort", str(cohort), "--output-tokens", str(count),
        "--inputs", str(OLD/"inputs/prompts.safetensors"), "--output", str(folder)]
    if smoke:
        command.append("--smoke")
    print(json.dumps(dict(command=command)), flush=True)
    with (folder/"run.log").open("w") as log:
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    path = folder / "result.json"
    log = (folder/"run.log").read_text()
    status = json.loads(path.read_text())["status"] if result.returncode == 0 and path.exists() else (
        "gpu_oom" if "out of memory" in log.lower() else "error")
    phase = json.loads((folder/"progress.json").read_text())["phase"]
    row = dict(context=length, output_tokens=count, cohort=cohort, status=status,
        phase=phase, returncode=result.returncode, directory=str(folder))
    saved.write_text(json.dumps(row, indent=2) + "\n")
    print(json.dumps(row), flush=True)
    return row


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source = OUT / "source"
    source.mkdir(exist_ok=True)
    for original in (BENCH, Path(__file__), ROOT/"benchmarks/system/bench_tp1_offline_request.py", ROOT/"benchmarks/system/audit_tp1_request.py"):
        dest = source / original.name
        if dest.exists():
            assert dest.read_bytes() == original.read_bytes()
        else:
            shutil.copy2(original, dest)
    with (OUT/"storage_validation.log").open("w") as log:
        validation = subprocess.run([sys.executable,str(BENCH),"--validate-storage"],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    assert validation.returncode == 0
    assert trial(8192,4,0,smoke=True)["status"] == "complete"
    local = json.loads((OUT/"smoke/result.json").read_text())["measurement"]["generated_tokens"]
    baseline = json.loads((OLD/"smoke/request_lrqk_t8192_c0_n4/result.json").read_text())["measurement"]["generated_tokens"]
    assert local == baseline
    rows = []
    for cohort in range(3):
        for length,count in [(32768,128),(65536,128),(130048,128),(130048,32),(130048,512)]:
            row = trial(length,count,cohort)
            rows.append(row)
            report(rows)
            assert row["status"] in ("complete","gpu_oom")
    print(json.dumps(dict(status="finished", trials=len(rows))),flush=True)


if __name__ == "__main__":
    main()
