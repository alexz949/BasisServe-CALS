# Closed-form Base16 + Q32 Fisher R8: capture, fitting, RULER

## Scope

The user approved increasing only residual query coverage from Q16 to Q32 in the terminal 8K span. Freeze the same closed-form MSE-RRR Base16 and C1-V80; refit uniform residual R8 with Page-Fisher BCD. No Adam, Base refit, C1 refit, or KL rank allocation.

The complete capture, audit, fitting, and RULER pipeline finished successfully. Seventeen small CPU regression tests passed in the `basis` environment, including Q32/Q16 nesting and value mismatch rejection, selected RoPE versus full-Q RoPE, causal/pinned masking, additive multi-query Fisher loss, and native RULER cache isolation. Final task-balanced RULER accuracy is 81.32575758%.

## Fixed settings

| Item | Setting |
| --- | --- |
| Model | Qwen3-8B-Base, BF16, all 36 layers |
| Payload | Frozen C1-V80, `qwen3_8b_c1_v80_32f4h_s32768_als6` |
| Base | Frozen closed-form affine MSE-RRR Base16 from `q8_residual_kl_bank` |
| Fit/diagnostic windows | Existing C4 64/16 windows, each 32768 tokens |
| Q sampling | 32 positions in terminal 8K, stride 256, all 32 Q heads |
| Fit examples/head | 64 × 32 = 2048 |
| Diagnostic examples/head | 16 × 32 = 512 |
| Residual | Post-RoPE exact K minus rotated Base prediction; uniform R8 |
| Objective | Separate causal non-sink Page-Fisher for each Q; no Q averaging |
| Solver | 40 BCD sweeps plus final query-factor solve; damping/tolerance 1e-5; PCG max 100 |
| Checkpoint choice | Fixed final sweep; diagnostic loss does not select factors |
| Selection | Page32/B2048 including pinned first 32 tokens |
| RULER | Same 11 tasks × 8 prompts, 88 paired samples, 32K |
| Prefill/decode | Full C1 prefill; native BF16 sparse decode in all 36 layers |
| Reference | Same-run full exact K + C1-V80 |
| Storage | GPU exact K and materialized Base128+R8; not an offload benchmark |
| Hardware | Four L40S workers on confirmed `yangGrp`; two CPUs/worker; `basis` |

The windows pack eight 4096-token C4 source windows without inserted separators. They are not native contiguous 32K documents. The RULER pilot is reused, not a new held-out set.

Q positions, zero-based:

\[
J_{32}=\{24576+256(i+1)-1:i=0,\ldots,31\}.
\]

Thus the sequence starts at 24831 and ends at 32767. Every other Q32 position, starting at its second entry, is one of the existing Q16 positions.

## Capture and provenance gates

New dense-teacher capture retains full Q projection and Q normalization and applies RoPE at the selected positions. Original model weights remain unmodified. Each window stores BF16 queries shaped `[36,32,32,128]`.

All existing Q16 values must match bitwise at shared positions, across all 80 windows and all layers. Q8 overlap is independently checked too. These checks run during capture, CPU audit, and fitting preflight. Saved reference manifests and tensor hashes remain intact; current generator source identity does not overwrite historical provenance.

The new capture uses `basisserve.qwen3.q32_queries.v1`. Residual fitting reads the audited position list and count from its manifest. It still copies all Base factors exactly. The evaluator validates the complete bank before RULER smoke.

New artifacts:

- Q capture: `results/calibration/q32_terminal8k`.
- Factors: `results/checkpoints/mse_base_q32_r8`.
- Evaluation: `results/evaluation/mse_base_q32_ruler32k`.

Q1/Q16 factors and results are not overwritten. Q16 residual tensors are not reused as fitted Q32 residual factors.

## Comparison references

| Configuration | RULER task-balanced accuracy |
| --- | ---: |
| Closed-form Base / Q1 R8 | 75.35984848% |
| Closed-form Base / Q16 R8 | 80.56818182% |
| Closed-form Base / Q32 R8 | 81.32575758% |
| Adam Q-aware Base / Q16 R8 | 80.43560606% |
| Exact K + C1-V80 | 85.20833333%; same-run reference |

## Underlying program commands

Working directory: `/deac/csc/yangGrp/zhangal/BasisServe-CALS`. GPU work runs through Slurm, with success dependencies between stages.

Capture smoke uses the following command. Formal capture replaces `--stage smoke` with `--stage capture` and uses shard indices 0–3. CPU audit uses `--stage summarize`.

```bash
/home/zhangal/.conda/envs/basis/bin/python -u scripts/capture_qwen3_8b_q16.py \
 --stage smoke --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 \
 --query-count 32 --query-layout terminal8k \
 --q16-reference results/calibration/q8_q16_queries \
 --output-dir results/calibration/q32_terminal8k \
 --shard-index 0 --num-shards 4 --torch-num-threads 2
```

Residual fitting, shard indices 0–3:

```bash
/home/zhangal/.conda/envs/basis/bin/python -u evaluation/fit_qwen3_8b_q8_fisher_residual.py \
 --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 \
 --c1-checkpoint results/checkpoints/qwen3_8b_c1_v80_32f4h_s32768_als6 \
 --initial-bank results/checkpoints/q8_residual_kl_bank --base-kind closed_form_rrr \
 --query-capture results/calibration/q32_terminal8k \
 --output-dir results/checkpoints/mse_base_q32_r8 \
 --shard-index 0 --num-shards 4 --torch-num-threads 2
```

RULER smoke follows all fitting shards. Formal evaluation uses `--stage evaluate` and shard indices 0–3; CPU summary uses `--stage summarize`.

```bash
/home/zhangal/.conda/envs/basis/bin/python -u evaluation/eval_qwen3_8b_residual_rank_ruler.py \
 --stage smoke --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 \
 --c1-checkpoint results/checkpoints/qwen3_8b_c1_v80_32f4h_s32768_als6 \
 --bank results/checkpoints/mse_base_q32_r8 \
 --data-dir results/datasets/qwen3_8b_base_ruler_v1_32k_shadowkv11_s8 \
 --output-dir results/evaluation/mse_base_q32_ruler32k \
 --samples-per-task 8 --sequence-length 32768 \
 --shard-index 0 --num-shards 4 --torch-num-threads 2
```

Fit workers reserve three hours and 64 GiB host memory each. Capture and RULER workers reserve one hour and 64 GiB each; smoke stages reserve 20 minutes and 64 GiB. CPU audits/summary reserve 20 minutes, two CPUs and 8 GiB on `small`. These are time limits, not runtime estimates.

The pipeline is capture smoke → four capture workers → CPU capture audit → four fit workers → bank checks/RULER smoke → four RULER workers → CPU summary. Downstream jobs cancel if dependencies fail. Submission scripts were removed after successful submission; logs and artifacts are retained. No GitHub upload is authorized.

## Submitted jobs

| Stage | Job | Resources |
| --- | --- | --- |
| Capture smoke | 8300673 | One L40S |
| Capture all 80 windows | 8300674, array 0–3 | Four L40S workers |
| Capture audit | 8300675 | CPU |
| Residual fit | 8300676, array 0–3 | Four L40S workers |
| Bank validation and RULER smoke | 8300677 | One L40S |
| Formal RULER | 8300678, array 0–3 | Four L40S workers |
| RULER summary | 8300679 | CPU |

Each stage depends on successful completion of the preceding stage. Logs use `logs/mse-q32-{stage}-{job}.out` and `.err`; arrays add `_{shard}` after the job number. Stage names are `cap-smoke`, `capture`, `cap-audit`, `fit`, `ruler-smoke`, `evaluate`, and `summary`.

Capture smoke passed: window 0 was captured in 5.18 seconds of window computation, with all 36 layers satisfying Q8 and Q16 overlap equality. The four formal capture workers then started on `lovelace`. These are capture checks, not RULER accuracy results.

## Completed execution

All jobs exited 0:0. Capture smoke elapsed time was 21 seconds. The four capture workers took 1:56, 1:57, 1:56, and 1:56; CPU capture audit took 19 seconds. All 80 windows and 36 layers passed the Q8/Q16 bitwise overlap gates.

Residual fitting workers took 41:08, 40:50, 41:13, and 40:24. Independent bank audit verified all 36 layer records and artifact hashes, 180 finite FP32 tensors, 108 Base tensors bitwise equal to the Q16 bank, and 36 changed residual encoders. Fit/diagnostic example counts were 2048/512 per query head. Base remained frozen and no Adam step occurred.

RULER smoke took 51 seconds. Formal workers took 6:35, 6:15, 6:59, and 6:18. CPU summary took 15 seconds. Formal maximum allocated GPU memory was 25.19556236 GiB. No NaN, OOM, or runtime failure was observed. These elapsed times include startup and are not controlled kernel benchmarks.

## Per-task results

All columns below use the same 88 prompt/reference pairs. Sparse columns retain C1-V80, closed-form Base16, uniform R8, Page32/B2048, and the same native attention path.

| Task | Q1 residual | Q16 residual | Q32 residual | Same-run exact K + C1-V80 |
| --- | ---: | ---: | ---: | ---: |
| niah_single_1 | 100.0000% | 100.0000% | 100.0000% | 100.0000% |
| niah_single_2 | 100.0000% | 100.0000% | 100.0000% | 100.0000% |
| niah_single_3 | 100.0000% | 100.0000% | 100.0000% | 100.0000% |
| niah_multikey_1 | 87.5000% | 87.5000% | 87.5000% | 87.5000% |
| niah_multikey_2 | 25.0000% | 50.0000% | 50.0000% | 87.5000% |
| niah_multiquery | 87.5000% | 100.0000% | 96.8750% | 96.8750% |
| niah_multivalue | 65.6250% | 93.7500% | 96.8750% | 93.7500% |
| vt | 92.5000% | 92.5000% | 92.5000% | 92.5000% |
| fwe | 83.3333% | 75.0000% | 83.3333% | 91.6667% |
| qa_1 | 50.0000% | 50.0000% | 50.0000% | 50.0000% |
| qa_2 | 37.5000% | 37.5000% | 37.5000% | 37.5000% |
| Task-balanced mean | 75.3598% | 80.5682% | 81.3258% | 85.2083% |

## Paired comparisons and checks

The new Q32 arm compared with previous arms:

| Reference | Difference, percentage points | Improved samples | Regressed samples | Ties |
| --- | ---: | ---: | ---: | ---: |
| Closed-form Base/Q16 | +0.75757576 | 3 | 1 | 84 |
| Closed-form Base/Q1 | +5.96590909 | 12 | 2 | 74 |
| Adam Q-aware Base/Q16 | +0.89015152 | 5 | 2 | 81 |
| Same-run exact K + C1-V80 | -3.88257576 | 3 | 7 | 78 |

Independent CPU rescoring computed case-insensitive reference substring hits directly: fractional hits for `all`, any-reference hit for `part`. Every saved sample score and both task-balanced means were reproduced. All samples preserved their immutable prefix and shared first token.

All 88 exact-K reference token sequences were identical to those in the closed-form Q1, closed-form Q16, and Adam/Q16 runs. Against the closed-form Q16 run, dataset identity, physical budget, prefix policy, prompting, generation policy, dense/sparse backends, and all recorded evaluation source hashes matched.

The Q16-to-Q32 mean gain in this reused pilot is 0.76 points: FWE increases by 8.33 task-level points, multivalue increases by 3.125 points, and multiquery decreases by 3.125 points. Multikey-2 is unchanged. This is a small pilot difference with four changed sample scores, not evidence that larger Q counts always improve accuracy. A 3.88-point gap to full exact-K/C1 attention remains.

Formal artifacts: [result JSON](../results/evaluation/mse_base_q32_ruler32k/result.json) and [generated per-task summary](../results/evaluation/mse_base_q32_ruler32k/summary.md). They retain the protocol, predictions, hashes, and executed program commands. No GitHub commit or push was performed.
