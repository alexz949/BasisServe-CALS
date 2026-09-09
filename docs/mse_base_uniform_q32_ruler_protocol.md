# Frozen closed-form Base16: full-window Q32 residual

## Controlled comparison

Freeze C1-V80 and the closed-form affine MSE-RRR Base16 from `results/checkpoints/q8_residual_kl_bank`. Only residual Q positions change from terminal-8K Q32 to full-window uniform Q32: zero-based `1023, 2047, ..., 32767`. All 32 query heads are retained. Refit uniform R8 with separate causal non-sink Page-Fisher terms, 40 BCD sweeps, PCG tolerance/damping 1e-5 and maximum 100 iterations. No Adam, Base update, payload update or KL allocation.

Use the same packed C4 64 fit / 16 diagnostic windows of 32768 tokens; 2048 / 512 query observations per head. The first two Q prefixes have at most 2048 tokens. They remain in this explicitly full-window sampling experiment. Diagnostics do not select checkpoints. Window-major statistics reuse token features without changing per-query Fisher terms.

Target evaluation: reused RULER32K 11 tasks × 8 prompts, all 36 layers, Page32/B2048 including one pinned prefix page, full C1 prefill, native BF16 sparse decode, same-run exact-K/C1-V80 reference. This is an accuracy pilot with GPU-resident exact K, not an offload benchmark. Previous terminal-Q32 accuracy: 81.32575758%; exact-K/C1-V80 reference: 85.20833333%.

## Checks and status

Twenty CPU regression tests passed in `basis`. New Q32 capture must match all existing uniform-Q16 positions and terminal-Q8 positions bitwise. Historical manifests, factors and evaluation outputs are not overwritten. Capture smoke job: 8300691, one L40S on the previously approved L40S partition; logs `logs/uq32-cap-smoke-8300691.out` and `.err`. Following explicit user confirmation, the formal pipeline was submitted. Temporary submission scripts were removed after submission.

Smoke completed successfully (exit 0:0, 21 seconds including startup). Window 0 capture took 5.17 seconds; all 36 layers passed the existing Q16/Q8 bitwise overlap checks. No new RULER accuracy is available yet.

## Program commands

Working directory: `/deac/csc/yangGrp/zhangal/BasisServe-CALS`. Environment: `basis`. Commands below use the absolute model snapshot. Formal jobs were confirmed after smoke. GPU arrays use shard indices 0–3.

```bash
/home/zhangal/.conda/envs/basis/bin/python -u scripts/capture_qwen3_8b_q16.py --stage smoke --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --query-count 32 --query-layout uniform32k --q16-reference results/calibration/uniform32k_q16_queries --output-dir results/calibration/q32_uniform32k --shard-index 0 --num-shards 4 --torch-num-threads 2
```

Formal capture changes `--stage smoke` to `--stage capture`; CPU audit uses `--stage summarize`.

```bash
/home/zhangal/.conda/envs/basis/bin/python -u evaluation/fit_qwen3_8b_q8_fisher_residual.py --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --c1-checkpoint results/checkpoints/qwen3_8b_c1_v80_32f4h_s32768_als6 --initial-bank results/checkpoints/q8_residual_kl_bank --base-kind closed_form_rrr --query-capture results/calibration/q32_uniform32k --output-dir results/checkpoints/mse_base_uniform_q32_r8 --shard-index 0 --num-shards 4 --torch-num-threads 2

/home/zhangal/.conda/envs/basis/bin/python -u evaluation/eval_qwen3_8b_residual_rank_ruler.py --stage smoke --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --c1-checkpoint results/checkpoints/qwen3_8b_c1_v80_32f4h_s32768_als6 --bank results/checkpoints/mse_base_uniform_q32_r8 --data-dir results/datasets/qwen3_8b_base_ruler_v1_32k_shadowkv11_s8 --output-dir results/evaluation/mse_base_uniform_q32_ruler32k --samples-per-task 8 --sequence-length 32768 --shard-index 0 --num-shards 4 --torch-num-threads 2
```

Formal RULER changes `--stage smoke` to `--stage evaluate`; CPU aggregation uses `--stage summarize`. No commit or push performed.

## Submitted pipeline

| Stage | Job ID | Resources |
| --- | --- | --- |
| Capture | 8300692 | Four L40S workers |
| Capture audit | 8300696 | CPU |
| Frozen Base / residual R8 fit | 8300697 | Four L40S workers |
| RULER smoke and bank validation | 8300698 | One L40S |
| RULER evaluation | 8300699 | Four L40S workers |
| RULER summary | 8300700 | CPU |

Stages use success dependencies, starting from successful capture smoke 8300691; invalid dependencies cancel downstream jobs. GPU workers reserve two CPUs and 64 GiB host memory each. Fit time limit is three hours, other GPU arrays one hour, smoke twenty minutes. CPU stages use two CPUs and 8 GiB on `small`. Logs: `logs/uq32-{stage}-{job}[_shard].out` and `.err`; stage names are `capture`, `cap-audit`, `fit`, `ruler-smoke`, `evaluate`, `summary`.

## Completed results

All stages completed with exit 0:0. Capture workers took 1:57–1:59; capture audit 18 seconds. Fit workers took 13:56, 14:14, 14:52 and 14:27. RULER smoke took 50 seconds; evaluation workers took 6:35, 6:15, 7:00 and 6:18; summary took 14 seconds. These are job elapsed times, not controlled kernel timings. Both Q distribution and statistics implementation differ from the previous terminal-Q32 fit, so the fit-time reduction cannot be attributed solely to either change.

Independent checkpoint audit verified all 36 layer hashes and finite tensors, unchanged Base tensors bitwise against terminal-Q32, and 36 changed residual encoders. All layers record the intended full-window Q32 positions.

| Task | Terminal-8K Q32 | Full-window Q32 | Same-run exact K + C1-V80 |
| --- | ---: | ---: | ---: |
| niah_single_1 | 100.0000% | 100.0000% | 100.0000% |
| niah_single_2 | 100.0000% | 100.0000% | 100.0000% |
| niah_single_3 | 100.0000% | 100.0000% | 100.0000% |
| niah_multikey_1 | 87.5000% | 87.5000% | 87.5000% |
| niah_multikey_2 | 50.0000% | 50.0000% | 87.5000% |
| niah_multiquery | 96.8750% | 96.8750% | 96.8750% |
| niah_multivalue | 96.8750% | 96.8750% | 93.7500% |
| vt | 92.5000% | 95.0000% | 92.5000% |
| fwe | 83.3333% | 70.8333% | 91.6667% |
| qa_1 | 50.0000% | 50.0000% | 50.0000% |
| qa_2 | 37.5000% | 37.5000% | 37.5000% |
| Task-balanced mean | 81.3258% | 80.4167% | 85.2083% |

Full-window Q32 minus terminal-Q32: -0.90909091 percentage points, with 4 improved sample scores, 6 regressed and 78 tied. Multikey-2 and multiquery have compensating per-sample changes despite unchanged task means. VT improves 2.5 task-level points; FWE decreases 12.5 task-level points. The full-window arm is 4.79166667 points below its exact-K/C1 reference.

Independent substring rescoring reproduced every score and both means. All 88 exact-K reference generated-token sequences match the terminal-Q32 run exactly. Prompt reference answers and source indices match; all prefix-immutability checks passed. Maximum allocated evaluation GPU memory was 25.19555283 GiB. No runtime failure occurred.

This reused 88-prompt pilot does not show an improvement from spreading fixed Q32 over the full window; it does not establish a general ordering across query distributions.

Artifacts: [result JSON](../results/evaluation/mse_base_uniform_q32_ruler32k/result.json), [generated summary](../results/evaluation/mse_base_uniform_q32_ruler32k/summary.md). Execution commands are preserved in the result and per-layer records. No GitHub commit or push performed.
