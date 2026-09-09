# Terminal-8K Q64/Q128 residual experiment

## Fixed protocol

Qwen3-8B-Base; frozen C1-V80 and closed-form affine MSE-RRR Base16 from `results/checkpoints/q8_residual_kl_bank`; uniform residual R8, Page32, pinned prefix page, B2048. Existing packed C4 windows: 64 fit and 16 diagnostic, each 32768 tokens. No Base or payload update, Adam, KL allocation, or diagnostic checkpoint selection. Residual objective remains per-query causal non-sink Page-Fisher with 40 BCD sweeps and PCG tolerance/damping 1e-5, maximum 100 iterations.

| Sampling | Zero-based positions | Fit/diagnostic Q observations per head | FP32 fit + diagnostic Gram storage |
| --- | --- | --- | --- |
| Terminal Q32 reference | 24831:256:32767 | 2048 / 512 | 5 GiB |
| Terminal Q64 | 24703:128:32767 | 4096 / 1024 | 10 GiB |
| Terminal Q128 | 24639:64:32767 | 8192 / 2048 | 20 GiB |

The notation start:stride:end includes the endpoint. All counts retain 32 query heads. Q64 and Q128 contain every Q32 position. Window-major statistics reuse C1/Base/RoPE/residual token features; per-query statistics and solver work still grow with Q. Gram storage estimates exclude temporary tensors and other live allocations. Full Q128 fitting memory is not validated by capture smoke.

Planned evaluation: same reused 11 RULER32K tasks × 8 prompts, full C1 prefill, native BF16 sparse decode, exact-K/C1 reference rerun, all 36 layers. Exact K remains on GPU. This is not a PCIe benchmark. Terminal Q32 reference accuracy is 81.32575758%.

## Smoke commands and state

Environment `basis`; working directory `/deac/csc/yangGrp/zhangal/BasisServe-CALS`. Twenty-one CPU regression tests passed, including new Q64/Q128 position nesting and overlap mismatch rejection. Capture checks immutable Q16 and Q8 teacher values bitwise. Old captures and checkpoints remain untouched.

```bash
/home/zhangal/.conda/envs/basis/bin/python -u scripts/capture_qwen3_8b_q16.py --stage smoke --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --query-count 64 --query-layout terminal8k --q16-reference results/calibration/q8_q16_queries --output-dir results/calibration/q64_terminal8k --shard-index 0 --num-shards 4 --torch-num-threads 2

/home/zhangal/.conda/envs/basis/bin/python -u scripts/capture_qwen3_8b_q16.py --stage smoke --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --query-count 128 --query-layout terminal8k --q16-reference results/calibration/q8_q16_queries --output-dir results/calibration/q128_terminal8k --shard-index 0 --num-shards 4 --torch-num-threads 2
```

Capture smoke jobs: Q64 8300712, Q128 8300713; one L40S and two CPUs/64 GiB host RAM each. Logs `logs/q64-cap-smoke-8300712.out` / `.err` and `logs/q128-cap-smoke-8300713.out` / `.err`. Temporary submission scripts were removed. At this smoke-only stage, formal fitting and RULER had not been submitted. Completed formal results are recorded below. No GitHub commit or push performed.

Both capture smoke jobs completed successfully, exit 0:0, elapsed 21 seconds each. Window computation was 5.20 seconds for Q64 and 5.25 seconds for Q128. An independent comparison of window 0 verified all 36 layers at every original Q32 position bitwise equal, in addition to the built-in Q16/Q8 checks.

## Formal submission

Both formal pipelines subsequently completed successfully; see the completed results below.

The user confirmed the sequential Q64 then Q128 pipeline. All twelve jobs were submitted; Q128 capture depends on successful Q64 summary. Within each experiment the stages are capture, capture audit, fit, RULER smoke, evaluate, and summary. Failure cancels dependent stages. The old successful capture-smoke jobs had expired from the scheduler's dependency state, so their COMPLETED / 0:0 accounting records were checked before submitting Q64 capture without an old-job dependency. No experiment setting changed.

| Q | Stage | Job ID |
| --- | --- | --- |
| 64 | capture | 8300715 |
| 64 | cap-audit | 8300716 |
| 64 | fit | 8300717 |
| 64 | ruler-smoke | 8300718 |
| 64 | evaluate | 8300719 |
| 64 | summary | 8300720 |
| 128 | capture | 8300721 |
| 128 | cap-audit | 8300722 |
| 128 | fit | 8300726 |
| 128 | ruler-smoke | 8300727 |
| 128 | evaluate | 8300728 |
| 128 | summary | 8300729 |

GPU arrays use four one-L40S workers on the user-confirmed yangGrp partition, two CPUs each. Q64 fit reserves 64 GiB host RAM per worker; Q128 fit reserves 96 GiB. Other GPU jobs reserve 64 GiB. Fit limits are three hours, capture/evaluation one hour, smoke twenty minutes. CPU audit/summary use small, two CPUs, 8 GiB, twenty minutes. These limits are not runtime estimates. Logs are `logs/q{Q}-{stage}-{job}[_shard].out` and `.err`. All temporary submission scripts were removed after successful submission. No commit or push was performed.

## Completed results

All capture, audit, fit, smoke, evaluation and summary jobs completed with exit 0:0. Q64 fit workers took 21:10, 20:44, 22:04 and 21:11; Q128 fit workers took 34:56, 34:12, 36:14 and 34:23. Q64 capture took 1:57–1:59 and audit 19 seconds; Q128 capture took 2:02–2:04 and audit 24 seconds. RULER smoke took 50 / 51 seconds. Q64 evaluation took 6:05–6:59 and summary 14 seconds; Q128 evaluation took 6:05–6:59 and summary 15 seconds. These are job elapsed times, including startup, not isolated kernel measurements.

| Task | Terminal Q32 | Terminal Q64 | Terminal Q128 | Same-run exact K + C1-V80 |
| --- | ---: | ---: | ---: | ---: |
| niah_single_1 | 100.0000% | 100.0000% | 100.0000% | 100.0000% |
| niah_single_2 | 100.0000% | 100.0000% | 100.0000% | 100.0000% |
| niah_single_3 | 100.0000% | 100.0000% | 100.0000% | 100.0000% |
| niah_multikey_1 | 87.5000% | 87.5000% | 87.5000% | 87.5000% |
| niah_multikey_2 | 50.0000% | 50.0000% | 50.0000% | 87.5000% |
| niah_multiquery | 96.8750% | 93.7500% | 93.7500% | 96.8750% |
| niah_multivalue | 96.8750% | 93.7500% | 93.7500% | 93.7500% |
| vt | 92.5000% | 90.0000% | 92.5000% | 92.5000% |
| fwe | 83.3333% | 83.3333% | 70.8333% | 91.6667% |
| qa_1 | 50.0000% | 50.0000% | 50.0000% | 50.0000% |
| qa_2 | 37.5000% | 37.5000% | 37.5000% | 37.5000% |
| Task-balanced mean | 81.3258% | 80.5303% | 79.6212% | 85.2083% |

| Paired comparison | Difference (percentage points) | Improved | Regressed | Tied |
| --- | ---: | ---: | ---: | ---: |
| Q64 minus Q32 | -0.79545455 | 2 | 5 | 81 |
| Q128 minus Q32 | -1.70454545 | 1 | 6 | 81 |
| Q128 minus Q64 | -0.90909091 | 2 | 4 | 82 |

Both new checkpoints passed independent audits of all 36 layer artifact hashes, finite tensors, frozen Base tensors bitwise equal to Q32, and the intended query positions. All 36 residual encoders changed in each new bank versus Q32. No OOM occurred in full fitting. An optional read-only `nvidia-smi` monitoring step was denied execution permission; this did not interrupt the batch jobs, but no fit GPU-memory peak is reported from it.

Independent case-insensitive reference substring rescoring reproduced every sample score. All 88 exact-K generated-token sequences match across Q32, Q64 and Q128. Reference answers, source indices and prefix-immutability checks also match/pass. Maximum allocated evaluation GPU memory was 25.19446564 GiB for Q64 and 25.19283676 GiB for Q128.

On this reused 88-prompt pilot, increasing terminal-8K query density from Q32 to Q64/Q128 did not improve RULER accuracy. These small paired differences do not identify the cause or establish that additional independent calibration windows would be ineffective. The old Q32 fit used the previous query-major statistics implementation; a separate real-capture equivalence smoke had verified unchanged Grams/loss for the new window-major path. No claim of monotonic fitting-error or statistical-significance trends is made here.

Artifacts: [Q64 result](../results/evaluation/mse_base_q64_ruler32k/result.json), [Q64 summary](../results/evaluation/mse_base_q64_ruler32k/summary.md), [Q128 result](../results/evaluation/mse_base_q128_ruler32k/result.json), [Q128 summary](../results/evaluation/mse_base_q128_ruler32k/summary.md). Exact execution commands and environment are retained in the records. No GitHub commit or push performed.

### Executed program commands

Working directory and basis environment are as above. For GPU arrays, `${SLURM_ARRAY_TASK_ID}` is 0–3. Audit uses the capture command with `--stage summarize` and shard 0; RULER smoke/summary use the evaluation command with `--stage smoke` / `--stage summarize` and shard 0.

```bash
/home/zhangal/.conda/envs/basis/bin/python -u scripts/capture_qwen3_8b_q16.py --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --query-count 64 --query-layout terminal8k --q16-reference results/calibration/q8_q16_queries --output-dir results/calibration/q64_terminal8k --stage capture --shard-index ${SLURM_ARRAY_TASK_ID} --num-shards 4 --torch-num-threads 2
```

```bash
/home/zhangal/.conda/envs/basis/bin/python -u evaluation/fit_qwen3_8b_q8_fisher_residual.py --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --c1-checkpoint results/checkpoints/qwen3_8b_c1_v80_32f4h_s32768_als6 --initial-bank results/checkpoints/q8_residual_kl_bank --base-kind closed_form_rrr --query-capture results/calibration/q64_terminal8k --output-dir results/checkpoints/mse_base_q64_r8 --shard-index ${SLURM_ARRAY_TASK_ID} --num-shards 4 --torch-num-threads 2
```

```bash
/home/zhangal/.conda/envs/basis/bin/python -u evaluation/eval_qwen3_8b_residual_rank_ruler.py --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --c1-checkpoint results/checkpoints/qwen3_8b_c1_v80_32f4h_s32768_als6 --bank results/checkpoints/mse_base_q64_r8 --data-dir results/datasets/qwen3_8b_base_ruler_v1_32k_shadowkv11_s8 --output-dir results/evaluation/mse_base_q64_ruler32k --samples-per-task 8 --sequence-length 32768 --stage evaluate --shard-index ${SLURM_ARRAY_TASK_ID} --num-shards 4 --torch-num-threads 2
```

```bash
/home/zhangal/.conda/envs/basis/bin/python -u scripts/capture_qwen3_8b_q16.py --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --query-count 128 --query-layout terminal8k --q16-reference results/calibration/q8_q16_queries --output-dir results/calibration/q128_terminal8k --stage capture --shard-index ${SLURM_ARRAY_TASK_ID} --num-shards 4 --torch-num-threads 2
```

```bash
/home/zhangal/.conda/envs/basis/bin/python -u evaluation/fit_qwen3_8b_q8_fisher_residual.py --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --c1-checkpoint results/checkpoints/qwen3_8b_c1_v80_32f4h_s32768_als6 --initial-bank results/checkpoints/q8_residual_kl_bank --base-kind closed_form_rrr --query-capture results/calibration/q128_terminal8k --output-dir results/checkpoints/mse_base_q128_r8 --shard-index ${SLURM_ARRAY_TASK_ID} --num-shards 4 --torch-num-threads 2
```

```bash
/home/zhangal/.conda/envs/basis/bin/python -u evaluation/eval_qwen3_8b_residual_rank_ruler.py --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --c1-checkpoint results/checkpoints/qwen3_8b_c1_v80_32f4h_s32768_als6 --bank results/checkpoints/mse_base_q128_r8 --data-dir results/datasets/qwen3_8b_base_ruler_v1_32k_shadowkv11_s8 --output-dir results/evaluation/mse_base_q128_ruler32k --samples-per-task 8 --sequence-length 32768 --stage evaluate --shard-index ${SLURM_ARRAY_TASK_ID} --num-shards 4 --torch-num-threads 2
```
