# Qwen3-8B Wo-only AllGather/AllReduce quality

- Arm: `lr_allreduce`
- C1 AllGather source rank: `384`
- Communication-matched AllReduce rank: `768`
- Ideal ring bytes/row/rank: `2304`
- WikiText-2 PPL: `8.39663739`
- MCQ average: `0.58041576`

| Task | Metric | Accuracy |
|---|---|---:|
| arc_easy | acc_norm | 0.46422559 |
| arc_challenge | acc_norm | 0.30972696 |
| hellaswag | acc_norm | 0.74556861 |
| piqa | acc_norm | 0.65016322 |
| winogrande | acc | 0.70955012 |
| boolq | acc | 0.77767584 |
| openbookqa | acc_norm | 0.40600000 |

## Command

```bash
evaluation/eval_qwen3_8b_wo_c1_ag_ar_mcq.py --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --factor-dir results/checkpoints/qwen3_8b_wo_c1_uniform_r384_s20 --arm lr_allreduce --output-dir results/evaluation/qwen3_8b_wo_c1_ag_ar_rank_sweep/r384/lr_allreduce --tasks arc_easy,arc_challenge,hellaswag,piqa,winogrande,boolq,openbookqa --lm-eval-batch-size 8 --ppl-batch-size 1 --device cuda:0 --torch-num-threads 4
```
