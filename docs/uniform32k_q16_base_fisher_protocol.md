# Uniform-32K Q16 Base16 + Page-Fisher R8

## Status

Implementation completed. Twenty small CPU tests and source preflight passed in
the `basis` environment. The user confirmed the commands below. Capture smoke
passed on L40S, including exact equality at all four Q8-shared positions in all
36 layers. All capture, fitting, bank audit, smoke, 88 formal RULER samples and
CPU summary jobs completed successfully. Uniform-32K Q16 Base16+R8 accuracy is
79.14772727%, versus 80.43560606% for terminal-Q16 on the same prompts.

## Controlled comparison

Compare the completed terminal-window configuration against a full-context
position configuration at the same number of queries:

- terminal Q16: positions 25087–32767 at spacing 512; Base is the previously
  fitted terminal-Q8 Base16 and residual R8 is fitted with terminal Q16;
- uniform-32K Q16: positions 2047, 4095, ..., 32767 at spacing 2048; refit both
  Base16 and residual R8 with these same 16 positions.

Both use the same 64 fit and 16 validation C4 windows of length 32768, current
C1-V80 ALS6, Base rank 16, residual rank 8, Page32 and RULER B2048. The Q count
and document population stay fixed; only position coverage and the explicitly
requested Base refit change. This is not all-Q.

The 16 uniform positions are equal-block endpoints across the full context:

`2047, 4095, 6143, 8191, 10239, 12287, 14335, 16383,
18431, 20479, 22527, 24575, 26623, 28671, 30719, 32767`.

These share four positions with the old terminal Q8 capture: 26623, 28671,
30719 and 32767. Query capture must be bitwise equal to the old BF16-origin Q8
values at every shared position, all 80 windows and all 36 layers. No tolerance
or fallback is permitted. This validates the teacher trajectory at shared
positions; it does not newly establish bitwise equality between the older A100
K/V capture and L40S query capture.

## Query capture

Use `scripts/capture_qwen3_8b_q16.py --query-layout uniform32k`. It captures
ordinary dense BF16 model Q after q_proj, q_norm and exact original-position
RoPE, without installing C1. It stores only Q: one BF16 `[36,16,32,128]`
tensor per window, about 360 MiB total for 80 formal windows. K/V rows are
reused from the same existing captures used by all preceding Q1/Q8/Q16 fits.

Run one separate smoke window, then four document shards covering indices
0–79, then a CPU audit. The audit checks all hashes, protocols, shapes, finite
values and shared-position equality before writing a completed manifest.

Output: `results/calibration/uniform32k_q16_queries`.

## Base and residual fitting

Entry: `evaluation/fit_qwen3_8b_uniform_q16_base_fisher.py`.

Initialize Base16 from `results/checkpoints/q8_qbase_fisher_bank`, then optimize
the causal raw-QK squared-error objective at all uniform-Q16 positions. Each Q
sees only keys through its original position and excludes pinned page0. Use the
same Adam protocol as the prior Q-aware Base: 12 maximum epochs, four documents
per step, factor LR 0.002, bias LR 0.0005, gradient clip 1, patience 4, seed 73.
Select the lowest validation raw-QK NMSE including epoch zero.

Freeze the newly selected Base. Recompute post-RoPE residual K and separately
form exact-teacher non-sink Page-Fisher statistics at the same uniform-Q16
positions. Keep every Q/Gram pair and sum losses; do not average Q vectors.
Fit only R8 with 40 BCD sweeps, relative damping 1e-5, iterative tolerance
1e-5 and maximum 100 iterations. Residual validation is diagnostic only; keep
the final fixed sweep. There is no KL allocation or adaptive rank.

Output: `results/checkpoints/uniform32k_q16_base_fisher_r8`, five FP32 tensors
per layer: three new Base tensors and the R8 encoder/query factors. Records
retain input/code hashes, Base training history and both residual diagnostics.

## Evaluation

After a 36-layer bank audit, rerun exact-K C1 and uniform R8 on the same existing
88-prompt RULER subset: 11 tasks × 8 examples. Use the same full C1 prefill,
shared first output token, native BF16 sparse decode, Page32/B2048 including
pinned page0, official caps/EOS and scoring. This is a reused pilot dataset,
not an untouched final set or the full 13-task suite.

Output: `results/evaluation/uniform32k_q16_base_fisher_ruler32k`.

Matched references are terminal-Q16 Base+R8 80.43560606%, terminal-Q8 Base +
terminal-Q8 residual 80.09469697%, terminal-Q8 Base + terminal-Q1 residual
69.50757576%, and C1 exact-K 85.20833333%. These are prior results, not new
uniform-32K measurements. Exact-K generations must be identical in a valid
matched comparison.

## Approved program commands

Working directory: `/deac/csc/yangGrp/zhangal/BasisServe-CALS`. Use
`/home/zhangal/.conda/envs/basis/bin/python`. GPU stages target L40S, with one
GPU/two CPUs per worker and four workers where useful. CPU audits use two CPUs.
Keep logs and remove temporary sbatch files immediately after submission.

### Capture smoke, capture array, capture audit

Run the command below with stages `smoke`, `capture` on shard indices 0–3, and
`summarize`, in success-dependent order:

```bash
/home/zhangal/.conda/envs/basis/bin/python scripts/capture_qwen3_8b_q16.py \
  --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 \
  --output-dir results/calibration/uniform32k_q16_queries \
  --query-layout uniform32k \
  --stage smoke --shard-index 0 --num-shards 4 --torch-num-threads 2
```

### Joint Base16 and residual R8 fitting

Run four layer shards after capture audit:

```bash
/home/zhangal/.conda/envs/basis/bin/python evaluation/fit_qwen3_8b_uniform_q16_base_fisher.py \
  --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 \
  --c1-checkpoint results/checkpoints/qwen3_8b_c1_v80_32f4h_s32768_als6 \
  --initial-bank results/checkpoints/q8_qbase_fisher_bank \
  --query-capture results/calibration/uniform32k_q16_queries \
  --output-dir results/checkpoints/uniform32k_q16_base_fisher_r8 \
  --shard-index 0 --num-shards 4 --torch-num-threads 2
```

### Bank audit and RULER

Audit 36 records, five finite FP32 tensors/layer, current hashes and full
coverage. Then run stages `smoke`, `evaluate` on four shards, and `summarize`:

```bash
/home/zhangal/.conda/envs/basis/bin/python evaluation/eval_qwen3_8b_residual_rank_ruler.py \
  --stage smoke \
  --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 \
  --c1-checkpoint results/checkpoints/qwen3_8b_c1_v80_32f4h_s32768_als6 \
  --bank results/checkpoints/uniform32k_q16_base_fisher_r8 \
  --data-dir results/datasets/qwen3_8b_base_ruler_v1_32k_shadowkv11_s8 \
  --output-dir results/evaluation/uniform32k_q16_base_fisher_ruler32k \
  --samples-per-task 8 --sequence-length 32768 \
  --shard-index 0 --num-shards 4 --torch-num-threads 2
```

## Small tests

Twenty direct test functions passed in `basis` (pytest is absent there). They
cover uniform and terminal position geometry, strict shared-Q overlap, exact
selected-position RoPE equivalence, document/layer/split indexing, completed
capture records, joint-bank shapes and Base-change reporting, causal/pinned
Fisher construction, multi-Q loss pairing/additivity, terminal-Q reproduction,
and existing Base/RULER cache and scoring regressions. Python compilation and
`git diff --check` also pass.

## Submission and initial checks

Submitted on 2026-09-05. Each stage requires the preceding stage to complete
successfully; downstream jobs use kill-on-invalid-dependency.

| Stage | Job | Resources | Time limit |
| --- | --- | --- | --- |
| Capture smoke | 8300495 | one L40S, 2 CPUs, 64 GiB host | 30 min |
| Capture all 80 windows | 8300496, array 0–3 | four L40S workers, 2 CPUs/64 GiB each | 4 h |
| Capture audit | 8300497 | 2 CPUs, 8 GiB | 30 min |
| Joint Base16 + R8 fit | 8300498, array 0–3 | four L40S workers, 2 CPUs/64 GiB each | 6 h |
| Bank audit and RULER smoke | 8300502 | one L40S, 2 CPUs, 64 GiB | 30 min |
| Formal RULER | 8300503, array 0–3 | four L40S workers, 2 CPUs/64 GiB each | 2 h |
| RULER summary | 8300504 | 2 CPUs, 8 GiB | 20 min |

Time limits are reservations, not runtime estimates. All seven temporary sbatch
files were removed after submission. Capture smoke completed and processed
window 0 in 6.30 seconds. It passed all shared-position checks. Formal capture
then started on all four L40S GPUs; no later stage had run at the initial check.

Logs are retained under `logs/u32q16-cap-smoke-8300495.{out,err}`,
`logs/u32q16-cap-8300496_{0,1,2,3}.{out,err}`,
`logs/u32q16-cap-audit-8300497.{out,err}`,
`logs/u32q16-fit-8300498_{0,1,2,3}.{out,err}`,
`logs/u32q16-ruler-smoke-8300502.{out,err}`,
`logs/u32q16-ruler-8300503_{0,1,2,3}.{out,err}` and
`logs/u32q16-summary-8300504.{out,err}`.

## Completed execution

All jobs exited 0:0. Capture smoke took 30 seconds including startup. Formal
capture workers 0–3 took 1:56, 1:57, 1:57 and 1:57; CPU capture audit took
14 seconds and verified all 80 windows, 36 layers and four shared Q positions.

Joint Base16+R8 fit workers 0–3 took 26:46, 26:16, 27:06 and 25:48. All 36
layers and 180 finite FP32 tensors passed the bank audit. Base factors changed
in 36/36 layers. On the uniform-Q16 validation objective, all 36 layers selected
a lower raw-QK NMSE than initialization; median relative reduction was 0.62485%
and mean relative reduction 0.894995%. Selected Base epochs were: epoch 3 for
three layers, 4 for one, 6 for two, 8 for four, 9 for five, 10 for seven, 11
for six and 12 for eight. These are validation-selected local-objective results,
not independent downstream evidence.

Bank audit plus RULER smoke took 1:10. Formal RULER workers took 6:36, 6:15,
7:00 and 6:15; CPU summary took 13 seconds. Formal maximum allocated GPU memory
was 25.1956 GiB. There were no observed NaN, OOM or runtime failures. Fitting
retains the Transformers RotaryEmbedding `device` argument deprecation warning.

## RULER results

| Task | Terminal-Q16 Base+R8 | Uniform-32K-Q16 Base+R8 | Uniform−terminal, pp | C1 exact-K |
| --- | ---: | ---: | ---: | ---: |
| niah_single_1 | 100.0000% | 100.0000% | 0.0000 | 100.0000% |
| niah_single_2 | 100.0000% | 100.0000% | 0.0000 | 100.0000% |
| niah_single_3 | 100.0000% | 100.0000% | 0.0000 | 100.0000% |
| niah_multikey_1 | 87.5000% | 87.5000% | 0.0000 | 87.5000% |
| niah_multikey_2 | 50.0000% | 37.5000% | -12.5000 | 87.5000% |
| niah_multiquery | 96.8750% | 93.7500% | -3.1250 | 96.8750% |
| niah_multivalue | 93.7500% | 96.8750% | +3.1250 | 93.7500% |
| vt | 90.0000% | 92.5000% | +2.5000 | 92.5000% |
| fwe | 79.1667% | 75.0000% | -4.1667 | 91.6667% |
| qa_1 | 50.0000% | 50.0000% | 0.0000 | 50.0000% |
| qa_2 | 37.5000% | 37.5000% | 0.0000 | 37.5000% |
| Task-balanced mean | 80.4356% | 79.1477% | -1.2879 | 85.2083% |

Against terminal-Q16, uniform-32K improved three samples, regressed four and
tied 81. Against the same-run exact-K reference, uniform-32K improved three,
regressed 11 and tied 74; its mean gap is -6.0606 pp. Twenty exact-K and 22
uniform-R8 generations reached their official caps without EOS.

Independent CPU rescoring reproduced all per-sample scores and both arm means.
Prompt/reference/index coverage, caps, first shared token, prefix checks,
evaluation code hashes and backend settings match terminal-Q16. The only
top-level protocol differences are the bank path, bank hashes and factor-bank
fitting protocol. All 88 exact-K generated token sequences are identical.
Tokenizer decode, EOS flags and current evaluation-code hashes also passed.

Result: `results/evaluation/uniform32k_q16_base_fisher_ruler32k/result.json`.
Report: `results/evaluation/uniform32k_q16_base_fisher_ruler32k/summary.md`.
