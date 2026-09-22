# Qwen3-8B Wo-only AllGather/AllReduce quality

- Arm: `c1_allgather`
- C1 AllGather source rank: `384`
- Communication-matched AllReduce rank: `768`
- Ideal ring bytes/row/rank: `2304`
- WikiText-2 PPL: `7.67803249`
- MCQ average: `0.70973535`

| Task | Metric | Accuracy |
|---|---|---:|
| arc_easy | acc_norm | 0.83543771 |
| arc_challenge | acc_norm | 0.59982935 |
| hellaswag | acc_norm | 0.76568413 |
| piqa | acc_norm | 0.80195865 |
| winogrande | acc | 0.72533544 |
| boolq | acc | 0.81590214 |
| openbookqa | acc_norm | 0.42400000 |

## Command

```bash
evaluation/eval_qwen3_8b_wo_c1_ag_ar_mcq.py --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --factor-dir results/checkpoints/qwen3_8b_wo_c1_uniform_r384_s20 --arm c1_allgather --output-dir results/evaluation/qwen3_8b_wo_c1_ag_ar_rank_sweep/r384/c1_allgather --tasks arc_easy,arc_challenge,hellaswag,piqa,winogrande,boolq,openbookqa --lm-eval-batch-size 8 --ppl-batch-size 1 --device cuda:0 --torch-num-threads 4
```
