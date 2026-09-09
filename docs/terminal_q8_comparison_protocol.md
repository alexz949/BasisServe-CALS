# Matched terminal-Q8: uniform versus inherited Query-Gram pivots

## Fixed comparison

Qwen3-8B-Base, frozen C1-V80 checkpoint qwen3_8b_c1_v80_32f4h_s32768_als6 and the same closed-form affine MSE-RRR Base16 from q8_residual_kl_bank. Both arms freshly fit uniform R8; no reuse of old Q-aware-Base Q8 factors. Page32/B2048, pinned page0, all36 layers and8 GQA groups. Environment basis.

Fit: existing64 C4 windows x32768 tokens; diagnostic: existing16 separate windows x32768. These are packed C4 windows, not native contiguous32K documents. Each head sees64x8=512 fit and16x8=128 diagnostic query-window examples. The same causal non-sink Page-Fisher objective,40 BCD sweeps, damping/tolerance1e-5, PCG maximum100 iterations, fixed final factors. No Adam, autograd optimization, Base refit, C1 refit or rank allocation.

Both arms use only positions in the final8K [24576,32768):

- uniform: positions25599,26623,27647,28671,29695,30719,31743,32767 at every layer.
- qgram: reuse the last-bin8 pivots of each layer from results/evaluation/qgram32/positions.json. No new whitening or pivot computation. The inherited whitening was estimated from full-window candidate Q in64 fit windows, not terminal-only Q. Positions are shared across windows within a layer.

Both position manifests are frozen before diagnostic query values are read. All Q values are copied from verified existing BF16 captures: uniform from q128_terminal8k, qgram from qgram32. Source hashes, token hashes and overlapping Q values are checked. No new capture/model forward is required to prepare or fit these factors. Six CPU sampler/manifest tests passed; syntax and whitespace checks passed.

## Evaluation

Each arm runs the same existing RULER32K pilot:11 tasks x8 prompts=88 paired samples, full exact-K+same C1-V80 reference rerun, greedy/task-specific generation caps, full shared C1 prefill and native BF16 sparse decode. Smoke is limited to4 generated tokens and is NOT an accuracy result. The pilot is reused, not untouched held-out data. The fitting objective never uses RULER answers.

Common-query page audit compares historical terminal-Q32 R8, new uniform-terminal-Q8 R8, and new qgram-terminal-Q8 R8. It uses the original diagnostic Q32 terminal8K captures for all three arms:36 layers x8 groups x16 windows x32 Q=147456 conditions/arm. Historical exact/q32 tables are checked bitwise. Saves full ranks, scores, page sets, per-head teacher mass, per-layer overlap CSV and top20 new-qgram8 high-mass misses per layer. Evaluation Q count32 is intentionally independent of fitting Q count8.

## Pipeline

- prepare: job8300848; CPU,2 CPUs/task.
- fit: job8300849, array 0-7%4; L40S,2 CPUs/task.
- smoke: job8300850, array 0-1; L40S,2 CPUs/task.
- ruler: job8300851, array 0-7%4; L40S,2 CPUs/task.
- ruler-summary: job8300852; CPU,2 CPUs/task.
- pages: job8300853, array 0-3; L40S,2 CPUs/task.
- pages-summary: job8300854; CPU,2 CPUs/task.

Stages run sequentially through afterok dependencies; invalid dependencies cancel downstream work. Arrays have at most4 concurrent workers, each one L40S and48GiB host RAM. The two fitting/evaluation arms are scheduled in two waves of4 workers. Pages run only after RULER summary succeeds. Existing jobs/checkpoints/results are not modified. Temporary Slurm submission scripts were removed after submission. No GitHub upload is authorized.

Prepared Q: results/calibration/terminal8/{uniform,qgram}/positions.json and queries/manifest.json.
Factors: results/checkpoints/terminal8_{uniform,qgram}_r8.
RULER: results/evaluation/terminal8_{uniform,qgram}_ruler32k.
Page audit: results/evaluation/terminal8_pages.
Logs: logs/terminal8-{stage}-{job}_{task}.{out,err}.

At submission, preparation is running; downstream work is queued. No Q8 accuracy or overlap result is available yet.

## Exact commands

Working directory: /deac/csc/yangGrp/zhangal/BasisServe-CALS. Shell task variables below are local to each stage. All commands use /home/zhangal/.conda/envs/basis/bin/python.

### prepare

```bash
/home/zhangal/.conda/envs/basis/bin/python -u evaluation/prepare_terminal_q8.py --output-root results/calibration/terminal8
terminal_shard=0
for terminal_policy in uniform qgram; do
/home/zhangal/.conda/envs/basis/bin/python -u evaluation/fit_qwen3_8b_q8_fisher_residual.py --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --c1-checkpoint results/checkpoints/qwen3_8b_c1_v80_32f4h_s32768_als6 --initial-bank results/checkpoints/q8_residual_kl_bank --base-kind closed_form_rrr --query-capture "results/calibration/terminal8/$terminal_policy/queries" --query-position-manifest "results/calibration/terminal8/$terminal_policy/positions.json" --output-dir "results/checkpoints/terminal8_${terminal_policy}_r8" --num-shards 4 --shard-index "$terminal_shard" --preflight-only
done
```

### fit

```bash
terminal_policies=(uniform qgram)
terminal_policy=${terminal_policies[$((SLURM_ARRAY_TASK_ID / 4))]}
terminal_shard=$((SLURM_ARRAY_TASK_ID % 4))
/home/zhangal/.conda/envs/basis/bin/python -u evaluation/fit_qwen3_8b_q8_fisher_residual.py --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --c1-checkpoint results/checkpoints/qwen3_8b_c1_v80_32f4h_s32768_als6 --initial-bank results/checkpoints/q8_residual_kl_bank --base-kind closed_form_rrr --query-capture "results/calibration/terminal8/$terminal_policy/queries" --query-position-manifest "results/calibration/terminal8/$terminal_policy/positions.json" --output-dir "results/checkpoints/terminal8_${terminal_policy}_r8" --num-shards 4 --shard-index "$terminal_shard"
```

### smoke

```bash
terminal_policies=(uniform qgram)
terminal_policy=${terminal_policies[$SLURM_ARRAY_TASK_ID]}
terminal_shard=0
/home/zhangal/.conda/envs/basis/bin/python -u evaluation/eval_qwen3_8b_residual_rank_ruler.py --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --c1-checkpoint results/checkpoints/qwen3_8b_c1_v80_32f4h_s32768_als6 --bank "results/checkpoints/terminal8_${terminal_policy}_r8" --data-dir results/datasets/qwen3_8b_base_ruler_v1_32k_shadowkv11_s8 --output-dir "results/evaluation/terminal8_${terminal_policy}_ruler32k" --samples-per-task 8 --sequence-length 32768 --num-shards 4 --shard-index "$terminal_shard" --stage smoke
```

### ruler

```bash
terminal_policies=(uniform qgram)
terminal_policy=${terminal_policies[$((SLURM_ARRAY_TASK_ID / 4))]}
terminal_shard=$((SLURM_ARRAY_TASK_ID % 4))
/home/zhangal/.conda/envs/basis/bin/python -u evaluation/eval_qwen3_8b_residual_rank_ruler.py --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --c1-checkpoint results/checkpoints/qwen3_8b_c1_v80_32f4h_s32768_als6 --bank "results/checkpoints/terminal8_${terminal_policy}_r8" --data-dir results/datasets/qwen3_8b_base_ruler_v1_32k_shadowkv11_s8 --output-dir "results/evaluation/terminal8_${terminal_policy}_ruler32k" --samples-per-task 8 --sequence-length 32768 --num-shards 4 --shard-index "$terminal_shard" --stage evaluate
```

### ruler-summary

```bash
terminal_shard=0
for terminal_policy in uniform qgram; do
/home/zhangal/.conda/envs/basis/bin/python -u evaluation/eval_qwen3_8b_residual_rank_ruler.py --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --c1-checkpoint results/checkpoints/qwen3_8b_c1_v80_32f4h_s32768_als6 --bank "results/checkpoints/terminal8_${terminal_policy}_r8" --data-dir results/datasets/qwen3_8b_base_ruler_v1_32k_shadowkv11_s8 --output-dir "results/evaluation/terminal8_${terminal_policy}_ruler32k" --samples-per-task 8 --sequence-length 32768 --num-shards 4 --shard-index "$terminal_shard" --stage summarize
done
```

### pages

```bash
for ((comparison_layer=SLURM_ARRAY_TASK_ID; comparison_layer<36; comparison_layer+=4)); do
for ((comparison_group=0; comparison_group<8; comparison_group++)); do
/home/zhangal/.conda/envs/basis/bin/python -u evaluation/compare_residual_selected_pages.py --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --c1-checkpoint results/checkpoints/qwen3_8b_c1_v80_32f4h_s32768_als6 --query-capture results/calibration/q32_terminal8k --comparison-bank q32=results/checkpoints/mse_base_q32_r8 --comparison-bank uniform8=results/checkpoints/terminal8_uniform_r8 --comparison-bank qgram8=results/checkpoints/terminal8_qgram_r8 --output-dir results/evaluation/terminal8_pages --stage evaluate --layer "$comparison_layer" --group "$comparison_group"
done
done
```

### pages-summary

```bash
/home/zhangal/.conda/envs/basis/bin/python -u evaluation/summarize_page_overlap.py --root results/evaluation/terminal8_pages --reference-root results/evaluation/page_overlap_all --ranking-arm qgram8
```

