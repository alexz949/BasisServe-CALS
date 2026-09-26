# TP8 Joint Batch Sweep

## Published Data

Raw results are stored at immutable HF revision
`8d96d7b135cd82647727fd5ca0675222adb29c1c`:

- [Complete raw archive](https://huggingface.co/alexz949/BasisServe-CALS/resolve/8d96d7b135cd82647727fd5ca0675222adb29c1c/results/system_benchmarks/tp8_joint_batch_sweep/raw.tar.gz)
- [1,080-file inventory](https://huggingface.co/alexz949/BasisServe-CALS/blob/8d96d7b135cd82647727fd5ca0675222adb29c1c/results/system_benchmarks/tp8_joint_batch_sweep/raw_manifest.json)

The archive retains repository-relative paths and includes formal/smoke rank
JSON, all successful and OOM logs, exact-command manifests, test logs and
`source.tar.gz`. Remote filenames and byte sizes were verified without explicit
SHA256 validation. GitHub contains the runner, summarizer, tests and small
Markdown/CSV tables, not the raw per-rank files.

For reproduction, restore the archived `source.tar.gz` into a separate checkout
before running the recorded commands: it contains the tested model/kernel
implementation, which may differ from the latest main branch. Model weights,
factor banks and prompt files are external inputs identified in the manifests.
Archived Markdown retains its pre-publication status; this README records the
subsequent upload.

## Scope

- Llama-3.1-8B-Instruct and Qwen3-32B, BF16, TP8 on 8 x L40S.
- Dense-local with explicitly selected Flash SDPA versus BasisKV Joint V96,
  full-scan B16R16 routing and pinned-host historical exact K with GPU slots.
- Contexts: 65,536 and 130,048. Batches: 1, 2, 4, 6, 8, 10, 12, 14, 16.
- One fresh-process trial per configuration: 72 attempted points, not repeated
  statistical estimates. No ALS-full, Quest, KV4 or A8 in this sweep.
- Each point uses 16 conditioning forwards then 128 timed decode forwards.
  This is full-model steady decode, not E2E request latency or vLLM scheduling.
- Attempt every requested batch even after an earlier OOM. Record observed OOM
  stages; do not fabricate skipped failures or fall back to Dense offload.
- Stop on a non-OOM failure. Completed/OOM points are retained on resumption.
- No kernel/model arithmetic changes, two-stage routing, new MLP optimization,
  quality evaluation or SHA256 validation.

Llama uses frozen cohort 0 prompts at each context; Qwen uses the first B rows
of its existing calibration windows. Within each model/context, the two arms
use identical prompt prefixes and lengths. Qwen retains static YaRN factor 4.
This measures runtime, not held-out quality. Prompts have sufficient rows for
B16; no repeated single prompt is used to stand in for independent batch rows.

## Metrics

The primary comparison is **peak decode allocated GPU memory versus batch**.
Prefill peak and OOM phases are auxiliary diagnostics. A prefill OOM leaves
decode memory unmeasured; it is not evidence of a decode-cache capacity limit.
The final main tables report the measured decode peaks and paired savings.

`manifest.json` records exact child commands, timestamps, status and metrics;
`summary.csv` is refreshed after each attempt. Every point has its own `run.log`
and per-rank JSON/logs. Successful active batch is checked against all eight
rank records. A failed fixed-batch trial is not partial successful throughput.

- Decode latency: mean of per-step maximum CUDA-event intervals across ranks.
- Aggregate throughput: B * 128 divided by maximum-rank measured wall time.
- Prefill peak, decode peak and decode resident allocated/reserved memory:
  maximum single-rank value in GiB, not total across all GPUs or nvidia-smi usage.
- Host K: sum of explicit persistent exact-K buffer bytes across all ranks.
- `sum_process_max_rss_gib`: sum of process lifetime RSS maxima, not a
  simultaneous host-memory peak. Shared pages may also be counted repeatedly.
- Failure phase comes from the last emitted rank stage. A prefill-complete
  marker only localizes a later failure to conditioning-or-decode, not one of
  those phases individually.

## Commands

Working directory: `/workspace/BasisServe-CALS-opt`. Conda environment: `basis`.
Slurm has no usable configuration on this host; direct execution follows the
existing machine-specific agreement. Formal execution requires user confirmation.

Smoke tests all four model/arm combinations at 4K/B6 with 2 conditioning and 8
measured steps, exercising a non-power-of-two batch before the long grid:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 CUDA_HOME=/usr/local/cuda MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=8.9 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /workspace/miniforge3/bin/conda run --no-capture-output -n basis python benchmarks/system/run_tp8_joint_batch_sweep.py --phase smoke > results/system_benchmarks/tp8_joint_batch_sweep/smoke.log 2>&1
```

Formal grid, after smoke passes and approval:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 CUDA_HOME=/usr/local/cuda MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=8.9 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /workspace/miniforge3/bin/conda run --no-capture-output -n basis python benchmarks/system/run_tp8_joint_batch_sweep.py --phase formal > results/system_benchmarks/tp8_joint_batch_sweep/formal.log 2>&1
```

Harness unit tests: `python -m pytest tests/test_tp8_joint_batch_sweep.py -q`
in `basis`: 3 passed. This covers the 72-point matrix, paired input paths,
non-power-of-two command generation and OOM/non-OOM classification.

Publication was separately approved after the sweep completed.

## Smoke Results

All four 4K/B6 trials completed with 32 complete rank JSON records and finite
logits. Actual batch was six on every rank. No OOM. These eight-step short
measurements are validation, not the formal long-context sweep.

| Model | Arm | Mean ms/step | Max-rank decode peak allocated GiB |
| --- | --- | ---: | ---: |
| Llama-3.1-8B | Dense Flash | 31.7672 | 3.1162 |
| Llama-3.1-8B | Basis Joint V96 | 25.2586 | 4.0318 |
| Qwen3-32B | Dense Flash | 71.7582 | 9.6632 |
| Qwen3-32B | Basis Joint V96 | 61.7850 | 14.1489 |

Raw smoke records are in `smoke/`; aggregate launcher output is `smoke.log`.
Known NUMA binding restrictions and NCCL barrier-device inference warnings
remain. The user approved the exact formal command and the 72-point queue
completed on 2026-09-26: 44 successful trials and 28 GPU OOM, with no non-OOM
failure. All 352 successful rank records and all paired successful prompt
inputs were validated. Status is recorded in `formal/manifest.json`.
The source snapshot is `source.tar.gz`; no experiment hot-path changes were
made after launch. All 301 archived source files matched the final worktree by
direct byte comparison; no SHA256 validation was performed. See
[RESULTS_SUMMARY.md](RESULTS_SUMMARY.md) and [comparison.csv](comparison.csv).
All eight GPUs were released after completion.
Final summary command in the `basis` environment:
`python -m benchmarks.system.summarize_tp8_joint_batch_sweep`.
