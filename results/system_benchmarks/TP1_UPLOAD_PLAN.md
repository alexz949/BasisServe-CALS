# TP1 Upload Plan

Status: scope and commit approved by the user. HF raw data uploaded and this manifest accompanies the GitHub publication.

## Destinations

- GitHub: alexz949/BasisServe-CALS, branch main.
- Hugging Face: model repository alexz949/BasisServe-CALS (public).
- Approved commit message: `add TP1 offline and LRQK GPU-local results`.
- Preserve the dirty working tree; prepare the commit in an isolated worktree based on remote main.

## GitHub File Scope

Benchmark code:

```text
benchmarks/system/audit_tp1_request.py
benchmarks/system/bench_tp1_offline_request.py
benchmarks/system/run_tp1_offline_grid.py
benchmarks/system/summarize_tp1_offline.py
benchmarks/system/verify_tp1_offline.py
benchmarks/system/bench_tp1_router_config.py
benchmarks/system/paper_faithful_compat.py
benchmarks/system/bench_tp1_lrqk_local.py
benchmarks/system/run_tp1_lrqk_local.py
benchmarks/system/summarize_tp1_lrqk_local.py
```

Results and upload documentation:

```text
results/system_benchmarks/TP1_UPLOAD_PLAN.md
results/system_benchmarks/tp1_offline/SUMMARY.md
results/system_benchmarks/tp1_offline/aggregate.csv
results/system_benchmarks/tp1_offline/formal_summary.csv
results/system_benchmarks/tp1_offline/formal_outcomes.json
results/system_benchmarks/tp1_offline/smoke_summary.csv
results/system_benchmarks/tp1_offline/smoke_outcomes.json
results/system_benchmarks/tp1_offline/capacity_summary.csv
results/system_benchmarks/tp1_offline/capacity_outcomes.json
results/system_benchmarks/tp1_offline/verification.json
results/system_benchmarks/tp1_offline/inputs/manifest.json
results/system_benchmarks/tp1_offline/freeze/provenance.json
results/system_benchmarks/tp1_offline/tp1_two_panel.png
results/system_benchmarks/tp1_offline/tp1_two_panel.pdf
results/system_benchmarks/tp1_offline/request_phases.png
results/system_benchmarks/tp1_offline/request_phases.pdf
results/system_benchmarks/tp1_lrqk_local/SUMMARY.md
results/system_benchmarks/tp1_lrqk_local/COMPARISON.md
results/system_benchmarks/tp1_lrqk_local/outcomes.json
results/system_benchmarks/tp1_lrqk_local/verification.json
```

On successful HF upload, update the three result-summary Markdown files with artifact links, revision and retrieval/extraction instructions. Preserve reported metrics. Do not include unrelated changes from the working tree.

## HF File Scope

Upload all existing files beneath these two local directories, preserving relative structure and excluding Python bytecode/cache files:

| Local directory | HF destination |
|---|---|
| results/system_benchmarks/tp1_offline/ | system_benchmarks/tp1_offline/ |
| results/system_benchmarks/tp1_lrqk_local/ | system_benchmarks/tp1_lrqk_local/ |

This includes raw per-trial result/measurement/command/progress/outcome JSON, logs (including failure logs), smoke/capacity records, frozen source snapshots, prompts, dependency archives and factor archives. Small summaries may be mirrored on both hosts for readability.

Measured before upload: approximately 20.08 MiB for the original grid and 0.25 MiB for the GPU-local supplement. The largest files are factors.tar.gz (13,933,073 bytes), dependencies.tar.gz (3,501,551 bytes) and prompts.safetensors (1,560,808 bytes). Individual JSON files are small; keeping raw trial files together on HF avoids cluttering GitHub.

The existing frozen dependency archive supplies the original TP1 harness and external implementation sources; these are not replaced by unrelated current-worktree changes. Reproduction still requires the model and the recorded environment.

## Approval and Artifact Revision

The user approved this scope, the commit message and pushing to GitHub main with "提交吧" after reviewing the plan.
HF immutable raw-data revision: `86c03092c663dc6654584143130b5e5abf2fedaa`.
No SHA256 checks are requested or performed.
