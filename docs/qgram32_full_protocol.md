# Stratified Q32 full-layer residual fitting

## Protocol

Model: Qwen3-8B-Base. Environment: /home/zhangal/.conda/envs/basis.
C4 fit windows 0–63: 64 × 32768 tokens. Diagnostic windows 64–79: 16 × 32768 tokens.
All 36 layers, 8 GQA groups, 32 query heads. Frozen C1-V80 and closed-form affine MSE-RRR Base16 from results/checkpoints/q8_residual_kl_bank.

Per layer, use all fit windows and 512 stride-64 candidate query positions (63 through 32767) to estimate per-head uncentered whitening and four position Grams. Select eight deterministic pivots per 8K bin, 32 positions per layer, shared across windows within that layer. Diagnostic observations do not participate in whitening or selection. FP64 CPU selection; whitening epsilon 1e-6. Repeat selection to check deterministic manifest results.

Only uniform R8 residual factors are refitted, using the original causal non-sink Page-Fisher objective, 40 BCD sweeps and PCG (relative damping 1e-5, tolerance 1e-5, maximum 100 iterations). Each head uses 2048 fit and 512 diagnostic query-window observations. Page size 32; prefix page 0 excluded from fitting and retained by the target router. Target deployment budget B2048. C1 and Base are not refitted. No Adam, autograd optimization, rank allocation, or RULER evaluation is included.

Existing terminal-Q32 and full-window-uniform-Q32 checkpoints remain unchanged as controls. Their results must be compared on common evaluation queries/prompts; diagnostic losses measured at different fitting-policy Q positions are not directly comparable.

## Submitted pipeline

| Stage | Job | Resources | Dependency |
| --- | --- | --- | --- |
| Fit candidate Q capture | 8300828 | 1 L40S, 2 CPUs, 48 GiB host RAM | None |
| Fit-only position selection | 8300829 | CPU partition small, 2 CPUs, 32 GiB RAM | 8300828 succeeds |
| Selected Q capture: reuse fit Q, capture diagnostic Q | 8300830 | 1 L40S, 2 CPUs, 48 GiB host RAM | 8300829 succeeds |
| Uniform R8 fitting | 8300831, array 0–3 | Each task: 1 L40S, 2 CPUs, 48 GiB host RAM; 9 layers | 8300830 succeeds |

At submission handoff, capture is running; the remaining stages are dependency-queued. No formal fit results are available yet. Invalid dependencies cancel downstream tasks. Temporary submission scripts were removed after submission. No unrelated jobs were changed.

Logs: logs/qgram-capture-8300828_4294967294.{out,err}, logs/qgram-select-8300829_4294967294.{out,err}, logs/qgram-diagnostic-8300830_4294967294.{out,err}, logs/qgram-fit-8300831_{0,1,2,3}.{out,err}.

Outputs: results/calibration/qgram_candidates; results/evaluation/qgram32; results/calibration/qgram32; results/checkpoints/mse_base_qgram32_r8.

## Commands

Working directory: /deac/csc/yangGrp/zhangal/BasisServe-CALS.

```bash
/home/zhangal/.conda/envs/basis/bin/python -u scripts/capture_qwen3_8b_q16.py --stage candidates --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --candidate-window-count 64 --candidate-stride 64 --num-shards 1 --output-dir results/calibration/qgram_candidates
```

```bash
/home/zhangal/.conda/envs/basis/bin/python -u evaluation/select_query_positions.py --candidate-capture results/calibration/qgram_candidates --output-dir results/evaluation/qgram32 --policy stratified_query_gram_pivot --device cpu
```

```bash
/home/zhangal/.conda/envs/basis/bin/python -u scripts/capture_qwen3_8b_q16.py --stage selected --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --query-position-manifest results/evaluation/qgram32/positions.json --num-shards 1 --output-dir results/calibration/qgram32
```

```bash
/home/zhangal/.conda/envs/basis/bin/python -u evaluation/fit_qwen3_8b_q8_fisher_residual.py --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --c1-checkpoint results/checkpoints/qwen3_8b_c1_v80_32f4h_s32768_als6 --initial-bank results/checkpoints/q8_residual_kl_bank --base-kind closed_form_rrr --query-capture results/calibration/qgram32 --query-position-manifest results/evaluation/qgram32/positions.json --num-shards 4 --shard-index "$SLURM_ARRAY_TASK_ID" --output-dir results/checkpoints/mse_base_qgram32_r8
```

