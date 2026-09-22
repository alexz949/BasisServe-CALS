# Qwen3-8B Wo-only AllGather/AllReduce quality

- Arm: `lr_allreduce`
- C1 AllGather source rank: `1024`
- Communication-matched AllReduce rank: `2048`
- Ideal ring bytes/row/rank: `6144`
- MCQ average: `0.70940984`

| Task | Metric | Accuracy |
|---|---|---:|
| arc_easy | acc_norm | 0.82028620 |
| arc_challenge | acc_norm | 0.57764505 |
| hellaswag | acc_norm | 0.78311093 |
| piqa | acc_norm | 0.79978237 |
| winogrande | acc | 0.73243883 |
| boolq | acc | 0.82660550 |
| openbookqa | acc_norm | 0.42600000 |

## Command

```bash
evaluation/eval_qwen3_8b_wo_c1_ag_ar_mcq.py --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --factor-dir results/checkpoints/qwen3_8b_wo_c1_uniform_r1024_s20 --arm lr_allreduce --output-dir results/evaluation/qwen3_8b_wo_c1_ag_ar_rank_sweep/r1024/lr_allreduce --tasks arc_easy,arc_challenge,hellaswag,piqa,winogrande,boolq,openbookqa --lm-eval-batch-size 8 --ppl-batch-size 1 --device cuda:0 --torch-num-threads 4 --skip-ppl
```
