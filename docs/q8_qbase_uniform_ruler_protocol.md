# Q-aware Base16 + uniform Fisher R8: RULER 32K

## Scope

User approved proceeding directly with uniform residual R8. No terminal KL
measurement, rank allocation, adaptive schedule, factor refit, or GitHub upload
is part of this run. The previously proposed KL continuation is superseded.

Two paired arms are rerun on the same 88 prompts:

1. Frozen current C1-V80 with full exact K.
2. The same C1-V80 with Q-aware Base16 and uniform Page-Fisher R8, all 36 layers.

Model: Qwen3-8B-Base. Environment: `basis`, BF16, L40S only.
C1 checkpoint: `results/checkpoints/qwen3_8b_c1_v80_32f4h_s32768_als6`.
Bank: `results/checkpoints/q8_qbase_fisher_bank`, completed and verified at all
36 layers. Only the R8 residual factors are installed. The bank's R4/R16
tensors are not evaluated. The bank was fitted on C4 64×32768 windows, with
16×32768 validation windows selecting Base epochs. Base uses eight separate
causal Q positions/window; residual Page-Fisher uses the terminal Q/window.

Dataset: `results/datasets/qwen3_8b_base_ruler_v1_32k_shadowkv11_s8`, seed 42,
official base-completion prompts, 32768 context budget including generation.
Tasks: `niah_single_1`, `niah_single_2`, `niah_single_3`, `niah_multikey_1`,
`niah_multikey_2`, `niah_multiquery`, `niah_multivalue`, `vt`, `fwe`, `qa_1`, `qa_2`.
Eight examples/task; not the complete 13-task RULER suite. `cwe` and
`niah_multikey_3` are not included. This dataset has been used previously and
is not an untouched final test.

The official generation caps are 128 for the seven NIAH tasks, 30 for `vt`,
50 for `fwe`, and 32 for each QA task. Greedy argmax decoding honors EOS.
Scoring follows the existing RULER case-insensitive substring metric: fraction
of reference answers found for NIAH/vt/fwe; any reference answer for QA.
Aggregate accuracy is the mean of the eleven task accuracies.

## Attention and cache settings

- One full-attention C1-V80 Triton prefill per prompt, shared by both arms.
- The first generated token comes from the common prefill. Routing affects
  subsequent decode forwards. Each arm receives an independent prefix-cache fork.
- Page32; B2048 physical tokens per KV group, including pinned page0; no forced
  current page and no adaptive budget.
- Base128+R8 routing coordinates are built for the prefix and appended
  incrementally during decode, using each arm's own trajectory.
- Native BF16 page-LSE selection and sparse exact attention. An explicit
  full-support decode mask disables the alternative fused decode path.
- Exact-K C1 decode uses SDPA with the same explicit support mask.
- Exact K stays on GPU. This is an accuracy oracle, not a PCIe or latency test.
- No old accuracy baseline is reused; no dense-V128 arm is included.

## Checks

Fifteen small CPU tests passed in `basis`, covering prefix isolation, native
one-token decode, EOS/cap handling, repeatability across arm switches, scoring,
causality, residual replay, DP and teacher mass. Running the DP unit test does
not fit or allocate an experimental schedule.

A single-page expanded validity-mask view could not be mutated in place. The
native attention code now materializes the Boolean intersection before any
subsequent in-place operation. The full-budget one-page CPU test and the
existing replay/mass regressions pass; page-selection mathematics is unchanged.

Before formal evaluation, a GPU smoke test runs four generated tokens on the
first prompt, checks 64 unique pages/group including page0, native selector
calls, sidecar lengths and widths, unchanged shared prefix, and bitwise-equal
logits when repeating uniform R8. Its four-token scores are not official
accuracy and are excluded from the 88-sample summary.

## Commands and execution

Submitted jobs: smoke `8300359`; formal array `8300360` (tasks 0–3, after smoke
success); CPU summary `8300361` (after formal array success). GPU workers
request 64 GiB host memory, the CPU summary 8 GiB. Temporary sbatch files were
removed after submission. Formal wall-time limit is two hours; this is not
an estimated duration.

Working directory: `/deac/csc/yangGrp/zhangal/BasisServe-CALS`.
The approved uniform experiment is submitted as a smoke job, a success-dependent
four-worker evaluation array, and a success-dependent CPU summary job.
Each GPU worker requests one L40S and two CPUs; each formal worker gets 22
prompts and runs both arms. All stages use the following program command,
changing `--stage` to `evaluate` or `summarize`, and formal worker
`--shard-index` to 0, 1, 2 or 3:

```bash
/home/zhangal/.conda/envs/basis/bin/python evaluation/eval_qwen3_8b_residual_rank_ruler.py \
  --stage smoke \
  --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 \
  --c1-checkpoint results/checkpoints/qwen3_8b_c1_v80_32f4h_s32768_als6 \
  --bank results/checkpoints/q8_qbase_fisher_bank \
  --data-dir results/datasets/qwen3_8b_base_ruler_v1_32k_shadowkv11_s8 \
  --output-dir results/evaluation/q8_qbase_r8_ruler32k \
  --samples-per-task 8 --sequence-length 32768 \
  --shard-index 0 --num-shards 4 --torch-num-threads 2
```

Each sample is written atomically with predictions, generated IDs, references,
scores, common-prefix checks, protocol, hashes, environment and command.
Summary requires all 88 unique paired samples and recomputes their scores.
Completed sample reuse requires identical protocol and GPU type. Logs are kept
under `logs/qruler-{smoke,eval,summary}-<job>[_<task>].{out,err}`. Temporary
sbatch files are removed immediately after submission.

## Initial execution checks

Smoke job `8300359` completed with exit code `0:0` in 1:24 including startup.
The four-token computation took 8.32 seconds; prefill took 6.40 seconds for
32628 prompt tokens. Peak allocated GPU memory was 25.57 GiB. Uniform R8
exercised 108 native selector calls (three decode forwards × 36 layers),
with 64 unique selected pages/group including page0. Sidecar rank/length
alignment, immutable prefix and exact-repeat logits checks all passed.
The four-token smoke scores are excluded from the formal accuracy report.
The four formal workers started automatically after smoke success.

## Completed formal run

All 88 paired prompts completed. No KL allocation or factor refit was run.

| Job/task | Elapsed | State | Exit code |
|---|---:|---|---|
| 8300360_0 | 06:42 | COMPLETED | 0:0 |
| 8300360_1 | 06:23 | COMPLETED | 0:0 |
| 8300360_2 | 07:07 | COMPLETED | 0:0 |
| 8300360_3 | 06:42 | COMPLETED | 0:0 |
| 8300361 summary | 00:17 | COMPLETED | 0:0 |

Task-balanced accuracy: C1 exact-K **85.2083%**, uniform R8 **69.5076%**,
difference **-15.7008 percentage points**. Paired scores decreased on 22
prompts, were equal on 66, and improved on none. The full per-task table and
predictions are in
[summary.md](../results/evaluation/q8_qbase_r8_ruler32k/summary.md) and
[result.json](../results/evaluation/q8_qbase_r8_ruler32k/result.json).

Independent CPU audit in `basis` verified all 88 sample identities, both arms,
four shard coverages, references, current protocol/source/bank hashes, GPU
identity, common first token, cap/EOS behavior, and unchanged-prefix flags.
Decoding every saved token sequence reproduced its saved prediction. Direct
substring scoring and independent task averaging reproduced all saved scores,
aggregate accuracies and paired counts. Smoke samples are not included.

Formal peak allocated GPU memory was 25.1932 GiB; median paired-sample wall
time was 14.8078 seconds. These are execution diagnostics, not controlled
latency measurements. Twenty exact-K generations and 25 uniform-R8 generations
reached their official task cap without EOS; the caps were not modified.
All finite-logit assertions passed; no CUDA error, OOM or failed job occurred.

The measured difference is relative to the same C1-V80 exact-K reference.
This run does not isolate Base estimation error from the sparse attention
budget, and does not establish whether Q-aware Base is better than MSE Base:
the old MSE-Base arm was not rerun with this protocol.
