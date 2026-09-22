# Qwen3-8B Wo-only AllGather/AllReduce quality

- Arm: `lr_allreduce`
- C1 AllGather source rank: `768`
- Communication-matched AllReduce rank: `1536`
- Ideal ring bytes/row/rank: `4608`
- MCQ average: `0.70727842`

| Task | Metric | Accuracy |
|---|---|---:|
| arc_easy | acc_norm | 0.81944444 |
| arc_challenge | acc_norm | 0.58532423 |
| hellaswag | acc_norm | 0.77733519 |
| piqa | acc_norm | 0.80195865 |
| winogrande | acc | 0.73559590 |
| boolq | acc | 0.81529052 |
| openbookqa | acc_norm | 0.41600000 |

## Command

```bash
evaluation/eval_qwen3_8b_wo_c1_ag_ar_mcq.py --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --factor-dir results/checkpoints/qwen3_8b_wo_c1_uniform_r768_s20 --arm lr_allreduce --output-dir results/evaluation/qwen3_8b_wo_c1_ag_ar_rank_sweep/r768/lr_allreduce --tasks arc_easy,arc_challenge,hellaswag,piqa,winogrande,boolq,openbookqa --lm-eval-batch-size 8 --ppl-batch-size 1 --device cuda:0 --torch-num-threads 4 --skip-ppl
```
