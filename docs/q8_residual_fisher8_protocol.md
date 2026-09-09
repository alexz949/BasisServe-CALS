# Q-aware Base16 with eight-query Page-Fisher R8

## Status

Implementation and eleven small CPU tests completed in `basis`. All 36 source
layers and existing direct/Q8 capture manifests passed preflight. The user
confirmed launching the fitting and RULER pipeline. All 36 fits, the bank audit,
GPU smoke, 88 paired RULER samples and CPU summary completed successfully.
No all-Q capture or all-Q fit is part of this run. Uniform R8 task-balanced
accuracy is 80.09469697%, compared with 69.50757576% for the previous 1-Q residual.

The environment does not contain pytest. The eleven zero-argument test functions
were invoked directly with `runpy` in `basis`, with two Torch CPU threads. Tests
cover causal prefixes, pinned-page exclusion, exact reproduction of the original
terminal-Q builder, Q/Gram pairing, additive multi-Q loss, query-order invariance
within FP32 tolerance, and existing Base and RULER cache/scoring regressions.

## Fixed factors and data

- Model: Qwen3-8B-Base; current C1-V80 ALS6, unchanged.
- Source bank: `results/checkpoints/q8_qbase_fisher_bank`.
- Base: copy the completed Q-aware Base16 factors bitwise; no optimizer or
  epoch selection runs for Base in this experiment.
- Residual: rebuild Page-Fisher statistics and fit uniform R8 at all 36 layers.
  No R4/R16 fit, terminal KL, or adaptive rank allocation.
- C4: 64 fit and 16 validation windows, each 32768 tokens. Existing windows
  pack eight 4096-token chunks; these are not 80 native 32K documents.
- Queries: existing dense-teacher post-RoPE Q captures at zero-based positions
  25599, 26623, 27647, 28671, 29695, 30719, 31743, 32767. All 32 query heads
  at each position; 512 fit and 128 validation document-position examples.
- No new activation capture or large new Q files.

## Objective

For each Q position t, use only K tokens i <= t. Exact K supplies the teacher
token probabilities, page masses and within-page weights. Exclude pinned page0
before non-sink normalization, with page size 32. Residual features remain
`K_post - RoPE(Base16(C1-V80))` at the original token positions.

Each query retains its own residual-feature Fisher Gram and query vector.
Concatenate these paired observations and minimize their summed Page-Fisher
quadratic losses. Do not average query vectors or replace the individual Grams
with a shared covariance. Use equal document-position example weights.

Fit one encoder per KV group and one query factor per query head, as before:
40 BCD sweeps, damping 1e-5, iterative tolerance 1e-5, maximum 100 iterations.
Validation is diagnostic only; save the fixed final sweep. Both the 1-Q and
8-Q evaluations use the same frozen Base. The normalized training losses have
different query populations and are not a matched accuracy comparison.

Output bank: `results/checkpoints/q8_qbase_fisher8_r8`, five tensors per layer:
three unchanged Base tensors plus newly fitted R8 encoder and query factors.
Per-layer records include source hashes, the original frozen-Base protocol,
causal-query settings, diagnostics and a bitwise Base identity assertion.

## Approved program commands

Working directory: `/deac/csc/yangGrp/zhangal/BasisServe-CALS`.
Use four independent GPU workers, one GPU and two CPUs per worker. Workers
use shard indices 0, 1, 2 and 3. The confirmed L40S pipeline uses the four idle
L40S GPUs on lovelace (yangGrp). Each GPU worker requests 64 GiB host memory.
Keep all Slurm logs; temporary sbatch files were deleted after each submission.

```bash
/home/zhangal/.conda/envs/basis/bin/python evaluation/fit_qwen3_8b_q8_fisher_residual.py \
  --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 \
  --c1-checkpoint results/checkpoints/qwen3_8b_c1_v80_32f4h_s32768_als6 \
  --initial-bank results/checkpoints/q8_qbase_fisher_bank \
  --output-dir results/checkpoints/q8_qbase_fisher8_r8 \
  --shard-index 0 --num-shards 4 --torch-num-threads 2
```

After all fits succeed and source/Base identity checks pass, run the existing
RULER smoke, then formal evaluation, then CPU summary. Use the command below
with `--stage smoke`, `--stage evaluate` (four shard indices), and
`--stage summarize`, respectively. No formal accuracy is reported from smoke.

```bash
/home/zhangal/.conda/envs/basis/bin/python evaluation/eval_qwen3_8b_residual_rank_ruler.py \
  --stage smoke \
  --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 \
  --c1-checkpoint results/checkpoints/qwen3_8b_c1_v80_32f4h_s32768_als6 \
  --bank results/checkpoints/q8_qbase_fisher8_r8 \
  --data-dir results/datasets/qwen3_8b_base_ruler_v1_32k_shadowkv11_s8 \
  --output-dir results/evaluation/q8_qbase_fisher8_ruler32k \
  --samples-per-task 8 --sequence-length 32768 \
  --shard-index 0 --num-shards 4 --torch-num-threads 2
```

RULER protocol is unchanged from `q8_qbase_uniform_ruler_protocol.md`: eleven
tasks with eight prompts each, official generation caps and EOS, full C1
prefill with shared first output token, native BF16 sparse decode, Page32,
B2048 per KV group including pinned page0, exact K resident on GPU.
Rerun C1 exact-K and the new uniform R8 arms. Compare the new aggregate and
per-task results against the completed 1-Q residual run in
`results/evaluation/q8_qbase_r8_ruler32k`: 69.5075758% uniform R8 and
85.2083333% C1 exact K. The existing dataset has been used previously.

## Submission and initial runtime checks

Submitted on 2026-09-05:

| Stage | Job | Dependency | Resources / time limit |
| --- | --- | --- | --- |
| Fit all 36 layers | 8300396, array 0–3 | none | four L40S workers; 6 h each |
| Bank audit and GPU smoke | 8300400 | afterok:8300396 | one L40S; 20 min |
| Formal RULER | 8300401, array 0–3 | afterok:8300400 | four L40S workers; 2 h each |
| CPU summary | 8300402 | afterok:8300401 | two CPUs, 8 GiB; 20 min |

Time limits are reservations, not measured runtimes or duration estimates.
All downstream jobs use kill-on-invalid-dependency. The pre-smoke audit checks
36 completed layer records and four shard records against current input/code
hashes, all 180 tensor shapes/dtypes/finiteness, and exact equality of the 108
Base tensors to the frozen source bank.

All four fit workers started on lovelace and entered Page-Fisher statistics
construction. Initial stderr only contained the Transformers deprecation
warning for `Qwen3RotaryEmbedding(..., device=...)`; no fit failure was observed
at this initial check. The bank audit and GPU smoke have not completed yet.

Logs: `logs/q8f8-fit-8300396_{0,1,2,3}.{out,err}`,
`logs/q8f8-smoke-8300400.{out,err}`,
`logs/q8f8-ruler-8300401_{0,1,2,3}.{out,err}`, and
`logs/q8f8-summary-8300402.{out,err}`.

## Completed execution

All jobs exited with code 0:0. Fitting workers 0–3 took 15:19, 14:53, 15:41,
and 15:35, respectively. Audit plus smoke took 1:05. Formal RULER workers
0–3 took 6:35, 6:15, 6:55, and 6:33; CPU summary took 15 seconds.
These are Slurm elapsed times, including startup.

The bank audit passed all 36 layers, 180 finite FP32 tensors, four completed
shards and 108 bitwise unchanged Base tensors. Smoke passed the native-selector,
cache-isolation and repeated-logit checks. Its four-token scores are excluded
from formal accuracy. Smoke peak allocated GPU memory was 25.57 GiB; formal
evaluation maximum was 25.1956 GiB. No NaN/OOM/runtime failure occurred.
The fitting log contains the previously noted RotaryEmbedding deprecation warning.

## RULER comparison: one versus eight residual Q positions

Both banks use the same frozen Q-aware Base16 and C1-V80. The same 88 prompts,
generation settings, Page32/B2048 budget and native decode backend are used.

| Task | Residual 1-Q R8 | Residual 8-Q R8 | Change, pp | C1 exact-K |
| --- | ---: | ---: | ---: | ---: |
| niah_single_1 | 100.0000% | 100.0000% | 0.0000 | 100.0000% |
| niah_single_2 | 100.0000% | 100.0000% | 0.0000 | 100.0000% |
| niah_single_3 | 75.0000% | 100.0000% | +25.0000 | 100.0000% |
| niah_multikey_1 | 87.5000% | 87.5000% | 0.0000 | 87.5000% |
| niah_multikey_2 | 25.0000% | 50.0000% | +25.0000 | 87.5000% |
| niah_multiquery | 84.3750% | 93.7500% | +9.3750 | 96.8750% |
| niah_multivalue | 71.8750% | 78.1250% | +6.2500 | 93.7500% |
| vt | 75.0000% | 92.5000% | +17.5000 | 92.5000% |
| fwe | 58.3333% | 91.6667% | +33.3333 | 91.6667% |
| qa_1 | 50.0000% | 50.0000% | 0.0000 | 50.0000% |
| qa_2 | 37.5000% | 37.5000% | 0.0000 | 37.5000% |
| Task-balanced mean | 69.5076% | 80.0947% | +10.5871 | 85.2083% |

Compared with 1-Q residual: 19 improved samples, two regressed, 67 tied.
Compared with the same-run C1 exact-K: three improved samples, nine regressed,
76 tied; the mean gap is -5.1136 pp. Twenty exact-K generations and 24 8-Q
residual generations reached their official task caps without EOS.

Independent CPU rescoring checked all 88 samples using fraction-of-reference
hits for `match_type=all` and any-reference hit for `match_type=part`. Sample
keys, references, prompt lengths, first shared token and generation caps match
the old experiment. The only evaluation-protocol differences are the bank path,
bank tensor hashes and factor-bank fitting protocol; evaluation source hashes
and all other settings match. All 88 exact-K token sequences match the previous
run exactly. Both reported arm means were independently reproduced.

New result: `results/evaluation/q8_qbase_fisher8_ruler32k/result.json`.
New report: `results/evaluation/q8_qbase_fisher8_ruler32k/summary.md`.
Previous result: `results/evaluation/q8_qbase_r8_ruler32k/result.json`.
