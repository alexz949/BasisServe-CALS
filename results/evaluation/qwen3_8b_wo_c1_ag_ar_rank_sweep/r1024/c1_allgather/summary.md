# Qwen3-8B Wo-only AllGather/AllReduce quality

- Arm: `c1_allgather`
- C1 AllGather source rank: `1024`
- Communication-matched AllReduce rank: `2048`
- Ideal ring bytes/row/rank: `6144`
- MCQ average: `0.70429579`

| Task | Metric | Accuracy |
|---|---|---:|
| arc_easy | acc_norm | 0.80176768 |
| arc_challenge | acc_norm | 0.56911263 |
| hellaswag | acc_norm | 0.78649671 |
| piqa | acc_norm | 0.79379761 |
| winogrande | acc | 0.72770324 |
| boolq | acc | 0.83119266 |
| openbookqa | acc_norm | 0.42000000 |

## Command

```bash
evaluation/eval_qwen3_8b_wo_c1_ag_ar_mcq.py --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --factor-dir results/checkpoints/qwen3_8b_wo_c1_uniform_r1024_s20 --arm c1_allgather --output-dir results/evaluation/qwen3_8b_wo_c1_ag_ar_rank_sweep/r1024/c1_allgather --tasks arc_easy,arc_challenge,hellaswag,piqa,winogrande,boolq,openbookqa --lm-eval-batch-size 8 --ppl-batch-size 1 --device cuda:0 --torch-num-threads 4 --skip-ppl
```
