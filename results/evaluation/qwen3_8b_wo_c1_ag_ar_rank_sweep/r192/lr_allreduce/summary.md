# Qwen3-8B Wo-only AllGather/AllReduce quality

- Arm: `lr_allreduce`
- C1 AllGather source rank: `192`
- Communication-matched AllReduce rank: `384`
- Ideal ring bytes/row/rank: `1152`
- WikiText-2 PPL: `31.72678866`
- MCQ average: `0.42135564`

| Task | Metric | Accuracy |
|---|---|---:|
| arc_easy | acc_norm | 0.33880471 |
| arc_challenge | acc_norm | 0.24488055 |
| hellaswag | acc_norm | 0.40440151 |
| piqa | acc_norm | 0.58977149 |
| winogrande | acc | 0.57379637 |
| boolq | acc | 0.50183486 |
| openbookqa | acc_norm | 0.29600000 |

## Command

```bash
evaluation/eval_qwen3_8b_wo_c1_ag_ar_mcq.py --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --factor-dir results/checkpoints/qwen3_8b_wo_c1_uniform_r192_s20 --arm lr_allreduce --output-dir results/evaluation/qwen3_8b_wo_c1_ag_ar_rank_sweep/r192/lr_allreduce --tasks arc_easy,arc_challenge,hellaswag,piqa,winogrande,boolq,openbookqa --lm-eval-batch-size 8 --ppl-batch-size 1 --device cuda:0 --torch-num-threads 4
```
