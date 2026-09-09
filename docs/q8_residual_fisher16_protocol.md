# Q16 Page-Fisher residual R8: nested-query experiment

## Status

Implementation complete; 16 small CPU tests passed in the `basis` environment.
Existing Q8 source checks passed for all 36 layers and all 64 fit plus 16
validation windows. The user confirmed the commands below. Capture smoke passed
on L40S, including bitwise Q8 overlap at all 36 layers. All 80 formal windows
and the independent CPU capture audit completed successfully. All 36 residual
fits, bank audit, RULER smoke, 88 formal paired samples and CPU summary also
completed successfully. Q16 uniform R8 accuracy is 80.43560606%, versus
80.09469697% for Q8 on the same prompts.

## Controlled change

Qwen3-8B-Base, frozen C1-V80 ALS6 and the completed Q-aware Base16 remain
unchanged. Source Base bank: `results/checkpoints/q8_qbase_fisher_bank`.
Only uniform residual R8 is refitted with more causal query positions. Base
continues to be the previously fitted eight-Q Base; it is not refitted with Q16.

Use the existing C4 64×32768 fit and 16×32768 validation windows. Each existing
window packs eight 4096-token chunks. Q16 keeps the same last-8192-token span
as Q8 and halves endpoint spacing from 1024 to 512 tokens. Zero-based positions:

`25087, 25599, 26111, 26623, 27135, 27647, 28159, 28671,
29183, 29695, 30207, 30719, 31231, 31743, 32255, 32767`.

Every other Q16 position, starting at index 1, is an existing Q8 position.
All 32 query heads participate at each position. There are 1024 fit and 256
validation document-position observations per head. This is not all-Q.

## Query-only capture and consistency gate

New entry: `scripts/capture_qwen3_8b_q16.py`.

Capture ordinary dense BF16 SDPA model activations, batch size one, without
installing C1. Project and normalize full Q rows with the original projection
shape, then select and rotate the 16 positions using their original RoPE
positions. No model backward or new K/V storage is required.

For every window and all 36 layers, the eight overlapping Q positions must be
bitwise equal, after FP32 conversion, to the existing Q8 observations. Any
mismatch stops the capture; there is no tolerance fallback, Q replacement, or
automatic continuation with different teacher activations. This controls the
Q8-to-Q16 capture change; it does not newly prove that the older A100 K/V
capture and L40S Q capture were themselves bitwise identical.

The exact same original K/V rows are reused in both the completed Q8 residual
experiment and this Q16 experiment. The initial source check verifies matching
window hashes, split offsets/counts, layer coverage and source manifests.
Existing Q8 values were checked to be exactly BF16-representable.

Formal output is one BF16 tensor `[36,16,32,128]` per window, approximately
360 MiB across 80 windows, excluding metadata and one separate smoke window.
Each file has an atomic completion record, SHA256 and source protocol. The
CPU capture summary independently rechecks all file hashes, shapes, finite
values and all Q8 overlaps before issuing a completed manifest.

Capture smoke processes window 0 through all 36 layers. Formal capture uses
four document shards with global indices `shard, shard+4, ...` across 0–79.
Fit documents are 0–63; validation documents 64–79. Smoke is separate and is
not counted among the 80 formal files.

## Fitting and evaluation

The existing `evaluation/fit_qwen3_8b_q8_fisher_residual.py` now accepts an
explicit `--query-capture` directory. With it, the script reads the audited
Q16 observations; without it, it uses the original Q8 observations. It checks
the completed capture manifest, all 80 artifacts and current source inputs.

The multi-query Page-Fisher algorithm is unchanged: separate causal prefix,
teacher non-sink softmax, page mass and conditional residual-feature Gram for
each query. Keep each Q/Gram pair and sum their losses; do not average queries.
Page32, pinned page0 excluded before non-sink normalization, uniform R8,
40 BCD sweeps, damping 1e-5, iterative tolerance 1e-5, maximum 100 iterations.
Validation is diagnostic only; keep the fixed final sweep. All three Base
tensors per layer are copied and asserted bitwise equal to the source.

Fit output: `results/checkpoints/q8_qbase_fisher16_r8`.
RULER output: `results/evaluation/q8_qbase_fisher16_ruler32k`.
Query output: `results/calibration/q8_q16_queries`.
Previous outputs are not overwritten.

After all fits pass a bank audit, use the same native BF16 RULER evaluator,
C1-V80 full prefill, exact-K reference, Page32/B2048 including pinned page0,
greedy generation and official caps/EOS. The dataset remains 11 tasks × 8
examples = 88 prompts, not an untouched test or the full 13-task suite.
No KL allocation, new Base fit, adaptive budget or offload benchmark.

Comparison references on the same dataset: residual Q1 69.50757576%, residual
Q8 80.09469697%, and C1 exact-K 85.20833333%. These are previous measurements,
not Q16 results. Matched exact-K generation and protocol checks will be used
when comparing completed runs.

## Approved program commands

Working directory: `/deac/csc/yangGrp/zhangal/BasisServe-CALS`.
All stages use `/home/zhangal/.conda/envs/basis/bin/python`.
GPU stages target L40S; four workers where useful, each one GPU and two CPUs.
CPU audit/summary stages use two CPUs. Retain logs and delete temporary sbatch
files immediately after submission. The confirmed L40S pipeline uses lovelace.

### Capture smoke, formal capture, CPU capture audit

Run the command below with `--stage smoke`, then `--stage capture` on four
workers (`--shard-index` 0, 1, 2, 3), then `--stage summarize` on CPU.
Each stage depends on the previous stage succeeding.

```bash
/home/zhangal/.conda/envs/basis/bin/python scripts/capture_qwen3_8b_q16.py \
  --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 \
  --output-dir results/calibration/q8_q16_queries \
  --stage smoke --shard-index 0 --num-shards 4 --torch-num-threads 2
```

### Residual-only fit

After the capture audit succeeds, run four layer shards (indices 0–3).

```bash
/home/zhangal/.conda/envs/basis/bin/python evaluation/fit_qwen3_8b_q8_fisher_residual.py \
  --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 \
  --c1-checkpoint results/checkpoints/qwen3_8b_c1_v80_32f4h_s32768_als6 \
  --initial-bank results/checkpoints/q8_qbase_fisher_bank \
  --query-capture results/calibration/q8_q16_queries \
  --output-dir results/checkpoints/q8_qbase_fisher16_r8 \
  --shard-index 0 --num-shards 4 --torch-num-threads 2
```

### Bank audit, RULER smoke, formal RULER and CPU summary

After verifying all 36 layers, current hashes, 180 finite FP32 tensors and
108 unchanged Base tensors, run the following evaluator with `--stage smoke`,
then `--stage evaluate` on four shards, then `--stage summarize` on CPU.

```bash
/home/zhangal/.conda/envs/basis/bin/python evaluation/eval_qwen3_8b_residual_rank_ruler.py \
  --stage smoke \
  --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 \
  --c1-checkpoint results/checkpoints/qwen3_8b_c1_v80_32f4h_s32768_als6 \
  --bank results/checkpoints/q8_qbase_fisher16_r8 \
  --data-dir results/datasets/qwen3_8b_base_ruler_v1_32k_shadowkv11_s8 \
  --output-dir results/evaluation/q8_qbase_fisher16_ruler32k \
  --samples-per-task 8 --sequence-length 32768 \
  --shard-index 0 --num-shards 4 --torch-num-threads 2
```

## Small-test coverage

Sixteen zero-argument test functions were invoked directly with `runpy` in
`basis` because that environment does not contain pytest. They cover nested
positions, strict overlap failure, selected-versus-full RoPE, fit/validation
document and layer indexing, capture record verification, CLI settings,
multi-Q Fisher loss additivity and pairing, causal/pinned masks, reproduction
of the terminal-Q builder, Base preservation and RULER cache/scoring regressions.

## Submission and initial checks

Submitted on 2026-09-05. Each stage requires the previous stage to exit
successfully; downstream jobs use kill-on-invalid-dependency.

| Stage | Job | Resources | Time limit |
| --- | --- | --- | --- |
| Capture smoke | 8300472 | one L40S, 2 CPUs, 64 GiB host | 30 min |
| Capture all 80 windows | 8300473, array 0–3 | four L40S workers, 2 CPUs/64 GiB each | 4 h |
| Capture audit | 8300474 | 2 CPUs, 8 GiB | 30 min |
| Residual-only fit | 8300475, array 0–3 | four L40S workers, 2 CPUs/64 GiB each | 6 h |
| Bank audit and RULER smoke | 8300479 | one L40S, 2 CPUs, 64 GiB | 30 min |
| Formal RULER | 8300480, array 0–3 | four L40S workers, 2 CPUs/64 GiB each | 2 h |
| RULER summary | 8300481 | 2 CPUs, 8 GiB | 20 min |

Time limits are reservations, not runtime estimates. All seven temporary
sbatch files were removed immediately after submission. Logs are retained as
`logs/q16-cap-smoke-8300472.{out,err}`,
`logs/q16-cap-8300473_{0,1,2,3}.{out,err}`,
`logs/q16-cap-audit-8300474.{out,err}`,
`logs/q16-fit-8300475_{0,1,2,3}.{out,err}`,
`logs/q16-ruler-smoke-8300479.{out,err}`,
`logs/q16-ruler-8300480_{0,1,2,3}.{out,err}` and
`logs/q16-summary-8300481.{out,err}`.

Capture smoke completed with exit 0:0 in 21 seconds including startup. Actual
window computation took 5.15 seconds; peak allocated GPU memory was 18.5304
GiB. All 36 layers passed the exact overlap gate. The first full-capture check
found 24 completed windows; later stages had not started at that check.

Full capture subsequently completed: workers 0–3 took 2:01, 2:02, 2:01 and
2:01, all exit 0:0. CPU capture audit took 18 seconds, exit 0:0, and verified
every Q8 overlap in all 80 windows/all 36 layers, artifact hashes, finite
values and shapes. Residual fit array 8300475 then started on four L40S GPUs.

## Completed execution and results

Fit workers 0–3 completed in 21:23, 19:56, 21:34 and 21:13, respectively,
all exit 0:0. Bank audit plus RULER smoke took 1:07, exit 0:0. All 36 layers,
180 finite FP32 tensors and 108 bitwise unchanged Base tensors passed the
bank audit. Smoke passed its native decode, cache-isolation and repeated-logit
checks; its four-token scores are not formal accuracy.

Formal RULER workers 0–3 completed in 6:37, 6:05, 7:12 and 6:15, all exit
0:0. CPU summary completed in 13 seconds, exit 0:0. These are Slurm elapsed
times including startup. Formal maximum allocated GPU memory was 25.1937 GiB.
There were no observed NaN, OOM or runtime failures. Fitting retains the
Transformers RotaryEmbedding `device` argument deprecation warning.

| Task | Q8 residual R8 | Q16 residual R8 | Q16−Q8, pp | C1 exact-K |
| --- | ---: | ---: | ---: | ---: |
| niah_single_1 | 100.0000% | 100.0000% | 0.0000 | 100.0000% |
| niah_single_2 | 100.0000% | 100.0000% | 0.0000 | 100.0000% |
| niah_single_3 | 100.0000% | 100.0000% | 0.0000 | 100.0000% |
| niah_multikey_1 | 87.5000% | 87.5000% | 0.0000 | 87.5000% |
| niah_multikey_2 | 50.0000% | 50.0000% | 0.0000 | 87.5000% |
| niah_multiquery | 93.7500% | 96.8750% | +3.1250 | 96.8750% |
| niah_multivalue | 78.1250% | 93.7500% | +15.6250 | 93.7500% |
| vt | 92.5000% | 90.0000% | -2.5000 | 92.5000% |
| fwe | 91.6667% | 79.1667% | -12.5000 | 91.6667% |
| qa_1 | 50.0000% | 50.0000% | 0.0000 | 50.0000% |
| qa_2 | 37.5000% | 37.5000% | 0.0000 | 37.5000% |
| Task-balanced mean | 80.0947% | 80.4356% | +0.3409 | 85.2083% |

Against Q8, Q16 improved four samples, regressed five and tied 79. Against the
same-run exact-K reference it improved two, regressed nine and tied 77, with
a mean gap of -4.7727 pp. Twenty exact-K and 25 Q16 residual generations
reached their official task caps without EOS.

Independent CPU rescoring reproduced both arm means and every per-sample score
using `all` fraction-of-reference hits and `part` any-reference hits. The two
runs have the same prompt/reference/index coverage, caps, first shared token,
unchanged-prefix checks, evaluator source hashes and evaluation settings. The
only top-level evaluation-protocol changes are bank path, bank hashes and
the factor-bank fitting protocol. All 88 exact-K generated token sequences
are identical to Q8. The Q16 query capture preserved all original Q8 query
values at every overlapping position, and Base stayed unchanged.

Results: `results/evaluation/q8_qbase_fisher16_ruler32k/result.json` and
`results/evaluation/q8_qbase_fisher16_ruler32k/summary.md`.
