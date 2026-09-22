# Qwen3-8B Wo-only AllGather/AllReduce quality

- Arm: `lr_allreduce`
- C1 AllGather source rank: `256`
- Communication-matched AllReduce rank: `512`
- Ideal ring bytes/row/rank: `1536`
- WikiText-2 PPL: `14.43070772`
- MCQ average: `0.50255890`

| Task | Metric | Accuracy |
|---|---|---:|
| arc_easy | acc_norm | 0.38552189 |
| arc_challenge | acc_norm | 0.25853242 |
| hellaswag | acc_norm | 0.63871739 |
| piqa | acc_norm | 0.61969532 |
| winogrande | acc | 0.63693765 |
| boolq | acc | 0.64250765 |
| openbookqa | acc_norm | 0.33600000 |

## Command

```bash
evaluation/eval_qwen3_8b_wo_c1_ag_ar_mcq.py --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --factor-dir results/checkpoints/qwen3_8b_wo_c1_uniform_r256_s20 --arm lr_allreduce --output-dir results/evaluation/qwen3_8b_wo_c1_ag_ar_rank_sweep/r256/lr_allreduce --tasks arc_easy,arc_challenge,hellaswag,piqa,winogrande,boolq,openbookqa --lm-eval-batch-size 8 --ppl-batch-size 1 --device cuda:0 --torch-num-threads 4
```
