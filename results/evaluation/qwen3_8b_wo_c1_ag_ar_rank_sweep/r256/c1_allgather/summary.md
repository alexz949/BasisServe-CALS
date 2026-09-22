# Qwen3-8B Wo-only AllGather/AllReduce quality

- Arm: `c1_allgather`
- C1 AllGather source rank: `256`
- Communication-matched AllReduce rank: `512`
- Ideal ring bytes/row/rank: `1536`
- WikiText-2 PPL: `9.40440861`
- MCQ average: `0.60089570`

| Task | Metric | Accuracy |
|---|---|---:|
| arc_easy | acc_norm | 0.57407407 |
| arc_challenge | acc_norm | 0.34385666 |
| hellaswag | acc_norm | 0.71888070 |
| piqa | acc_norm | 0.70729053 |
| winogrande | acc | 0.69218627 |
| boolq | acc | 0.77798165 |
| openbookqa | acc_norm | 0.39200000 |

## Command

```bash
evaluation/eval_qwen3_8b_wo_c1_ag_ar_mcq.py --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --factor-dir results/checkpoints/qwen3_8b_wo_c1_uniform_r256_s20 --arm c1_allgather --output-dir results/evaluation/qwen3_8b_wo_c1_ag_ar_rank_sweep/r256/c1_allgather --tasks arc_easy,arc_challenge,hellaswag,piqa,winogrande,boolq,openbookqa --lm-eval-batch-size 8 --ppl-batch-size 1 --device cuda:0 --torch-num-threads 4
```
