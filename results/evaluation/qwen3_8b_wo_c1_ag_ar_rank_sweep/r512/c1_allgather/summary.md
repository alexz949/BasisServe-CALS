# Qwen3-8B Wo-only AllGather/AllReduce quality

- Arm: `c1_allgather`
- C1 AllGather source rank: `512`
- Communication-matched AllReduce rank: `1024`
- Ideal ring bytes/row/rank: `3072`
- MCQ average: `0.71176931`

| Task | Metric | Accuracy |
|---|---|---:|
| arc_easy | acc_norm | 0.82786195 |
| arc_challenge | acc_norm | 0.60238908 |
| hellaswag | acc_norm | 0.77743477 |
| piqa | acc_norm | 0.79923830 |
| winogrande | acc | 0.73480663 |
| boolq | acc | 0.81865443 |
| openbookqa | acc_norm | 0.42200000 |

## Command

```bash
evaluation/eval_qwen3_8b_wo_c1_ag_ar_mcq.py --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --factor-dir results/checkpoints/qwen3_8b_wo_c1_lr_ar_phase1_tp4_exact_fp64_s20 --arm c1_allgather --output-dir results/evaluation/qwen3_8b_wo_c1_ag_ar_rank_sweep/r512/c1_allgather --tasks arc_easy,arc_challenge,hellaswag,piqa,winogrande,boolq,openbookqa --lm-eval-batch-size 8 --ppl-batch-size 1 --device cuda:0 --torch-num-threads 4 --skip-ppl
```
