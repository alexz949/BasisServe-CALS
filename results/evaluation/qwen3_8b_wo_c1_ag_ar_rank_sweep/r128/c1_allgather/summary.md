# Qwen3-8B Wo-only AllGather/AllReduce quality

- Arm: `c1_allgather`
- C1 AllGather source rank: `128`
- Communication-matched AllReduce rank: `256`
- Ideal ring bytes/row/rank: `768`
- WikiText-2 PPL: `44.18062608`
- MCQ average: `0.40683328`

| Task | Metric | Accuracy |
|---|---|---:|
| arc_easy | acc_norm | 0.30387205 |
| arc_challenge | acc_norm | 0.25597270 |
| hellaswag | acc_norm | 0.36705835 |
| piqa | acc_norm | 0.56147987 |
| winogrande | acc | 0.56274665 |
| boolq | acc | 0.51070336 |
| openbookqa | acc_norm | 0.28600000 |

## Command

```bash
evaluation/eval_qwen3_8b_wo_c1_ag_ar_mcq.py --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --factor-dir results/checkpoints/qwen3_8b_wo_c1_uniform_r128_s20 --arm c1_allgather --output-dir results/evaluation/qwen3_8b_wo_c1_ag_ar_rank_sweep/r128/c1_allgather --tasks arc_easy,arc_challenge,hellaswag,piqa,winogrande,boolq,openbookqa --lm-eval-batch-size 8 --ppl-batch-size 1 --device cuda:0 --torch-num-threads 4
```
