# Closed-form Base16 + Q16 Fisher R8: fitting and RULER

## Scope and current status

The user approved freezing the completed closed-form MSE-RRR Base16 and refitting only its uniform R8 residual with 16 causal queries/window. No Adam, Base update, C1 update, rank allocation, or new query capture is included.

The pipeline is complete. Ten small CPU regression tests passed in `basis`. Preflight verified all 36 original Base maps bitwise against their closed-form artifacts, checked the fit inputs, and independently reverified every Q8 overlap in all 80 windows and 36 layers. All fitting shards, smoke, all formal RULER shards, and CPU summary completed with exit 0:0. Task-balanced RULER accuracy is 80.56818182%.

The initial preflight rejected equality between the saved Q16 capture protocol and the current generator protocol. The generator had subsequently added `query_layout`, changed an overlap-description string, and changed its source hash. The immutable model/window/query specifications matched. Capture reuse now checks these semantic inputs explicitly, verifies every saved file against its original record, and rechecks actual Q8 tensor equality. The original capture provenance is retained, not rewritten as current-source output. This was a CPU validation issue; no failed GPU fit was launched.

## Fixed experiment

| Item | Setting |
| --- | --- |
| Model | Qwen3-8B-Base, 36 layers, 8 KV groups, 4 Q heads/group |
| Payload | Frozen C1-V80, `qwen3_8b_c1_v80_32f4h_s32768_als6` |
| Base source | `results/checkpoints/q8_residual_kl_bank` |
| Base objective/solver | Affine rank16 unweighted pre-RoPE K MSE; whitening plus truncated SVD |
| Base update | None; copy original factors bitwise |
| Residual | Exact post-RoPE K minus rotated Base prediction |
| Fit split | Existing 64 C4 windows × 32768 tokens |
| Diagnostic split | Existing 16 C4 windows × 32768 tokens |
| Q observations | 16 positions/window, all 32 heads, separate causal prefixes |
| Fit examples/head | 1024 |
| Diagnostic examples/head | 256 |
| Page objective | Exact-teacher non-sink Page-Fisher; page size 32; first 32 tokens excluded before normalization |
| Solver | 40 BCD sweeps; final query refit; relative damping/tolerance 1e-5; PCG cap 100 |
| Rank/selection | Uniform R8; fixed final sweep; diagnostic loss does not select factors |
| New factors | `results/checkpoints/mse_base_q16_r8` |
| RULER | Same reused 11-task × 8-sample 32K pilot, 88 prompts |
| Attention | Full C1 prefill; all 36 layers sparse during native BF16 decode |
| Budget | Page32/B2048, one pinned prefix page included |
| Accuracy reference | Same-run full exact K + C1-V80 |
| Results | `results/evaluation/mse_base_q16_ruler32k` |
| Environment/hardware | `basis`, four L40S workers on `yangGrp`, two CPUs/worker |

The packed C4 windows are eight 4096-token source windows without inserted separators, not native 32K documents. Query positions are zero-based:

`25087, 25599, 26111, 26623, 27135, 27647, 28159, 28671, 29183, 29695, 30207, 30719, 31231, 31743, 32255, 32767`.

All eligible non-sink causal Key tokens enter each query's statistics. Q vectors are not averaged. Old Q1 residual factors are not reused as fitted Q16 factors. No original checkpoint or Q1 result is overwritten.

## Comparisons

| Base / residual | RULER score |
| --- | ---: |
| Closed-form Base16 / Q1 R8 | 75.35984848% |
| Closed-form Base16 / Q16 R8 | 80.56818182% |
| Adam Q-aware Base16 / Q1 R8 | 69.50757576% |
| Adam Q-aware Base16 / Q16 R8 | 80.43560606% |
| C1-V80 / exact K | 85.20833333%; same-run reference |

The Q1-to-Q16 comparison keeps the closed-form Base fixed. The comparison with 80.44% matches residual Q coverage but changes Base and the residual fitted for that Base. This is a reused pilot, not an untouched final benchmark or evidence of statistical significance by itself.

## Commands

Working directory: `/deac/csc/yangGrp/zhangal/BasisServe-CALS`. Commands execute through Slurm.

Fitting uses shard indices 0–3:

```bash
/home/zhangal/.conda/envs/basis/bin/python -u evaluation/fit_qwen3_8b_q8_fisher_residual.py \
  --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 \
  --c1-checkpoint results/checkpoints/qwen3_8b_c1_v80_32f4h_s32768_als6 \
  --initial-bank results/checkpoints/q8_residual_kl_bank --base-kind closed_form_rrr \
  --query-capture results/calibration/q8_q16_queries \
  --output-dir results/checkpoints/mse_base_q16_r8 \
  --shard-index 0 --num-shards 4 --torch-num-threads 2
```

After all fits pass, run smoke, then `--stage evaluate` with shard indices 0–3, then `--stage summarize --shard-index 0`:

```bash
/home/zhangal/.conda/envs/basis/bin/python -u evaluation/eval_qwen3_8b_residual_rank_ruler.py \
  --stage smoke \
  --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 \
  --c1-checkpoint results/checkpoints/qwen3_8b_c1_v80_32f4h_s32768_als6 \
  --bank results/checkpoints/mse_base_q16_r8 \
  --data-dir results/datasets/qwen3_8b_base_ruler_v1_32k_shadowkv11_s8 \
  --output-dir results/evaluation/mse_base_q16_ruler32k \
  --samples-per-task 8 --sequence-length 32768 \
  --shard-index 0 --num-shards 4 --torch-num-threads 2
```

The evaluator checks all new Base tensors against the original bank before GPU evaluation, validates expected FP32 shapes and finite values, and identifies this bank as closed-form Base rather than Q-aware Base. Exact K remains GPU-resident; the materialized sidecar is Base128+R8. This is not an offload benchmark.

## Submitted pipeline

| Stage | Job | Dependency |
| --- | --- | --- |
| Residual fit, four L40S workers | 8300663, array 0–3 | None |
| Bank checks and RULER smoke, one L40S | 8300667 | All fits successful |
| Formal RULER, four L40S workers | 8300668, array 0–3 | Smoke successful |
| CPU summary | 8300669 | All evaluation shards successful |

Fit workers reserve two hours and 64 GiB host memory each; evaluation workers one hour and 64 GiB each; smoke reserves 20 minutes and 64 GiB. CPU summary uses two CPUs, 8 GiB, 20 minutes on `small`. These are limits, not elapsed-time estimates. Invalid dependencies cancel downstream jobs.

Logs: `logs/mse-q16-fit-8300663_{0,1,2,3}.{out,err}`, `logs/mse-q16-smoke-8300667.{out,err}`, `logs/mse-q16-evaluate-8300668_{0,1,2,3}.{out,err}`, and `logs/mse-q16-summarize-8300669.{out,err}`.

Temporary submission files were removed after submission. Logs and artifacts remain available. No GitHub commit or push was performed.

## Completed execution and verification

Fit worker elapsed times were 23:27, 23:22, 23:33, and 22:35. Smoke took 58 seconds. Formal RULER worker elapsed times were 6:34, 6:04, 6:58, and 6:25. CPU summary took 15 seconds. These are Slurm elapsed times including startup, not isolated kernel benchmarks.

Independent bank audit verified 36 layer records and file hashes, 180 finite FP32 tensors, and 108 Base tensors bitwise equal to the Q1 source bank. All 36 residual encoders changed. No Base update or Adam step occurred. Formal peak allocated GPU memory was 25.19555283 GiB. No NaN, OOM, or runtime failure was observed; fitting logs retain the existing Transformers RotaryEmbedding `device` deprecation warning.

The four-column comparison below uses the same 88 prompt/reference pairs:

| Task | Closed-form Base / Q1 R8 | Closed-form Base / Q16 R8 | Adam Q-aware Base / Q16 R8 | Exact K + C1-V80 |
| --- | ---: | ---: | ---: | ---: |
| niah_single_1 | 100.0000% | 100.0000% | 100.0000% | 100.0000% |
| niah_single_2 | 100.0000% | 100.0000% | 100.0000% | 100.0000% |
| niah_single_3 | 100.0000% | 100.0000% | 100.0000% | 100.0000% |
| niah_multikey_1 | 87.5000% | 87.5000% | 87.5000% | 87.5000% |
| niah_multikey_2 | 25.0000% | 50.0000% | 50.0000% | 87.5000% |
| niah_multiquery | 87.5000% | 100.0000% | 96.8750% | 96.8750% |
| niah_multivalue | 65.6250% | 93.7500% | 93.7500% | 93.7500% |
| vt | 92.5000% | 92.5000% | 90.0000% | 92.5000% |
| fwe | 83.3333% | 75.0000% | 79.1667% | 91.6667% |
| qa_1 | 50.0000% | 50.0000% | 50.0000% | 50.0000% |
| qa_2 | 37.5000% | 37.5000% | 37.5000% | 37.5000% |
| Task-balanced mean | 75.3598% | 80.5682% | 80.4356% | 85.2083% |

Paired comparisons for the new closed-form Base/Q16 arm:

| Reference | Mean difference, pp | Improved samples | Regressed samples | Ties |
| --- | ---: | ---: | ---: | ---: |
| Closed-form Base/Q1 | +5.20833333 | 11 | 3 | 74 |
| Adam Q-aware Base/Q16 | +0.13257576 | 3 | 2 | 83 |
| Same-run exact K + C1-V80 | -4.64015152 | 2 | 8 | 78 |

Independent CPU rescoring directly recomputed case-insensitive substring hits: fraction of references for `all`, any-reference hit for `part`. It reproduced each saved sample score and the macro means. Every sample retained its immutable prefix and common first token. All 88 exact-K generated token sequences were identical to both the closed-form/Q1 and Adam/Q16 historical references.

Dataset identity, physical budget, pinned-prefix size, prompting, generation policy, and sparse/dense backends matched both prior runs. Among tracked evaluation source hashes, only the evaluation entry differed; recorded attention and cache implementation hashes matched. The entry changes validate and label the different factor-bank provenance.

In this pilot, increasing residual query coverage with Base fixed raised accuracy by 5.21 points. At Q16 coverage, the closed-form and Adam Base arms differ by only 0.13 points, with 83 of 88 sample scores tied; this does not establish a meaningful accuracy advantage of one Base fitting method. A 4.64-point gap to full exact-K/C1 attention remains.

Formal artifacts: [result JSON](../results/evaluation/mse_base_q16_ruler32k/result.json) and [generated per-task summary](../results/evaluation/mse_base_q16_ruler32k/summary.md). They contain protocol hashes, predictions, and exact executed program commands.
