# Stratified Q32 residual: RULER 32K pilot

## Fixed protocol

Qwen3-8B-Base, BF16, basis environment. Frozen C1-V80 and closed-form MSE-RRR Base16; new uniform R8 factors from results/checkpoints/mse_base_qgram32_r8. The new sampler used 64 fit C4 windows of 32768 tokens and 16 diagnostic windows. These are existing packed C4 windows, not new native contiguous 32K documents. Per-layer stratified Query-Gram selection chooses 8 positions per 8K bin, 32 Q per window; diagnostic data does not select Q or factors.

Evaluation reuses the existing 11-task × 8-prompt RULER 32K pilot (88 paired prompts), not a new held-out set. Full shared C1 prefill, independent cache forks, greedy task-specific generation caps, all 36 layers sparse for the routing arm, Page32/B2048 including pinned page0, no adaptive budget, no forced current page. The two arms are new uniform R8 routing and a fresh full exact-K + same C1-V80 reference. This is an accuracy oracle with GPU-resident exact K, not a PCIe performance test.

Historical comparison: terminal-Q32 R8 81.32575758%, full-window-uniform Q32 R8 80.41666667%, full exact-K/C1-V80 85.20833333%. New formal scores are not yet available. Compare the full reference token sequences and paired sample scores before attributing differences to sampling.

## Smoke

Job 8300836 completed 0:0 in 50 seconds on one L40S. Peak allocated memory 25.56967497 GiB. Checkpoint validation, repeat logits equality, native sparse selector exercise, rank/cache alignment and immutable prefix checks passed.

Smoke intentionally caps generation at four tokens. Both arms produced the same four-token prefix as the old correct answer; the printed zero scores reflect truncation and are not accuracy measurements. No baseline regression was established by smoke.

## Formal submission

Formal evaluation job 8300837, array 0–3: four L40S workers, each 2 CPUs / 48 GiB host memory and 22 prompts. Dependent CPU summary job 8300841, 2 CPUs / 8 GiB, starts only after all evaluation tasks succeed. Invalid dependencies cancel summary. Temporary submission scripts removed after submission. No unrelated jobs changed. No GitHub upload authorized.

Logs: logs/qgram-ruler-evaluate-8300837_{0,1,2,3}.{out,err}; logs/qgram-ruler-summary-8300841_4294967294.{out,err}.

Results: results/evaluation/qgram32_ruler32k/evaluate/sample_*.json; result.json and summary.md are produced by the summary job.

## Commands

Working directory: /deac/csc/yangGrp/zhangal/BasisServe-CALS.

```bash
/home/zhangal/.conda/envs/basis/bin/python -u evaluation/eval_qwen3_8b_residual_rank_ruler.py --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --c1-checkpoint results/checkpoints/qwen3_8b_c1_v80_32f4h_s32768_als6 --bank results/checkpoints/mse_base_qgram32_r8 --data-dir results/datasets/qwen3_8b_base_ruler_v1_32k_shadowkv11_s8 --output-dir results/evaluation/qgram32_ruler32k --samples-per-task 8 --sequence-length 32768 --num-shards 4 --torch-num-threads 2 --stage evaluate --shard-index "$SLURM_ARRAY_TASK_ID"
```

```bash
/home/zhangal/.conda/envs/basis/bin/python -u evaluation/eval_qwen3_8b_residual_rank_ruler.py --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --c1-checkpoint results/checkpoints/qwen3_8b_c1_v80_32f4h_s32768_als6 --bank results/checkpoints/mse_base_qgram32_r8 --data-dir results/datasets/qwen3_8b_base_ruler_v1_32k_shadowkv11_s8 --output-dir results/evaluation/qgram32_ruler32k --samples-per-task 8 --sequence-length 32768 --num-shards 4 --torch-num-threads 2 --stage summarize --shard-index 0
```

