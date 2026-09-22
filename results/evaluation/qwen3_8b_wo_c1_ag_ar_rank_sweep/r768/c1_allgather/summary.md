# Qwen3-8B Wo-only AllGather/AllReduce quality

- Arm: `c1_allgather`
- C1 AllGather source rank: `768`
- Communication-matched AllReduce rank: `1536`
- Ideal ring bytes/row/rank: `4608`
- MCQ average: `0.70398437`

| Task | Metric | Accuracy |
|---|---|---:|
| arc_easy | acc_norm | 0.80092593 |
| arc_challenge | acc_norm | 0.56484642 |
| hellaswag | acc_norm | 0.78440550 |
| piqa | acc_norm | 0.79597388 |
| winogrande | acc | 0.72454617 |
| boolq | acc | 0.83119266 |
| openbookqa | acc_norm | 0.42600000 |

## Command

```bash
evaluation/eval_qwen3_8b_wo_c1_ag_ar_mcq.py --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --factor-dir results/checkpoints/qwen3_8b_wo_c1_uniform_r768_s20 --arm c1_allgather --output-dir results/evaluation/qwen3_8b_wo_c1_ag_ar_rank_sweep/r768/c1_allgather --tasks arc_easy,arc_challenge,hellaswag,piqa,winogrande,boolq,openbookqa --lm-eval-batch-size 8 --ppl-batch-size 1 --device cuda:0 --torch-num-threads 4 --skip-ppl
```
