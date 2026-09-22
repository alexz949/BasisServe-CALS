# Qwen3-8B Wo-only AllGather/AllReduce quality

- Arm: `c1_allgather`
- C1 AllGather source rank: `192`
- Communication-matched AllReduce rank: `384`
- Ideal ring bytes/row/rank: `1152`
- WikiText-2 PPL: `13.68877247`
- MCQ average: `0.52374394`

| Task | Metric | Accuracy |
|---|---|---:|
| arc_easy | acc_norm | 0.45370370 |
| arc_challenge | acc_norm | 0.27047782 |
| hellaswag | acc_norm | 0.62597092 |
| piqa | acc_norm | 0.66158868 |
| winogrande | acc | 0.63614838 |
| boolq | acc | 0.64831804 |
| openbookqa | acc_norm | 0.37000000 |

## Command

```bash
evaluation/eval_qwen3_8b_wo_c1_ag_ar_mcq.py --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --factor-dir results/checkpoints/qwen3_8b_wo_c1_uniform_r192_s20 --arm c1_allgather --output-dir results/evaluation/qwen3_8b_wo_c1_ag_ar_rank_sweep/r192/c1_allgather --tasks arc_easy,arc_challenge,hellaswag,piqa,winogrande,boolq,openbookqa --lm-eval-batch-size 8 --ppl-batch-size 1 --device cuda:0 --torch-num-threads 4
```
