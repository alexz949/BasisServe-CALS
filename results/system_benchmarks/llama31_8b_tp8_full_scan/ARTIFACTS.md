# Frozen TP8 Grid Artifacts

Publication status: local results validated; raw artifacts uploaded to HF at immutable data revision `61eda8a41285730eb254a904813666cb38c94725`.

## Local Results

- `SUMMARY.md`: complete three-cohort table, failures, protocol, limitations, and commands.
- `summary.csv`: 48 workload/method rows; 44 complete, 4 prefill-OOM groups.
- `decode_trial_summary.csv`: all 144 trials, including 12 prefill OOMs.
- `validation.json`: 132 successful trials, 1056 validated rank results, 1629 unchanged frozen source files.
- `decode_grid_trials.json`: exact per-trial commands, timestamps, return codes, and log paths.
- `run.log`, `launcher_*.log`, and per-rank logs: progress and failure evidence.
- Per-trial directories: raw rank JSON, generated token IDs, and all 128 measured step times.

## Tested Source

Base commit: `bae46c1409d3fa00030bab88df8b359e20b886bf`, plus the pre-run working-tree source archive in `freeze/source.tar.gz`.

`freeze/manifest.json` lists the 1629 archived files, environment, inputs, and command. `freeze/source.patch` captures tracked changes; the archive also contains the untracked source files present at freezing.

Every archived source file was compared directly with the final working tree and matched. No source SHA256 check was performed. New external-baseline prototype files created during the run are not imported by the frozen runner and are not part of this archive or this publication list. They may appear in later trials' Git status metadata.

## Proposed Publication

GitHub repository: https://github.com/alexz949/BasisServe-CALS

- Target branch: `main`.
- User-provided commit message: `upload new results for figure 3`.
- Exact 32-file list: `GITHUB_FILES.txt`.
- Preserve unrelated existing `main` results and the original worktree's unfinished rebase.

HF repository: https://huggingface.co/alexz949/BasisServe-CALS

Proposed destinations for raw JSON, logs, and source archives, preserving local relative paths:

```text
results/system_benchmarks/tp8_path_opt/
results/system_benchmarks/tp8_full_scan/
results/system_benchmarks/llama31_8b_tp8_full_scan/
```

The upload completed at [HF data revision `61eda8a41285730eb254a904813666cb38c94725`](https://huggingface.co/alexz949/BasisServe-CALS/commit/61eda8a41285730eb254a904813666cb38c94725). The pinned file listing contains [638 path-optimization files](https://huggingface.co/alexz949/BasisServe-CALS/tree/61eda8a41285730eb254a904813666cb38c94725/results/system_benchmarks/tp8_path_opt), [72 full-scan files](https://huggingface.co/alexz949/BasisServe-CALS/tree/61eda8a41285730eb254a904813666cb38c94725/results/system_benchmarks/tp8_full_scan), and [2367 formal-grid files](https://huggingface.co/alexz949/BasisServe-CALS/tree/61eda8a41285730eb254a904813666cb38c94725/results/system_benchmarks/llama31_8b_tp8_full_scan). The formal-grid source archive, trial manifest, and validation JSON were individually confirmed present. The HF data revision identifies the uploaded files even if repository documentation changes later.

The first directory contains historical two-stage pilot measurements. They are not mixed with the final full-scan grid.
