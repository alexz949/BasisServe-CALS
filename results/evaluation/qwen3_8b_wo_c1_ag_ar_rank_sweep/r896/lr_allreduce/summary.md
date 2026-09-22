# Qwen3-8B Wo-only AllGather/AllReduce quality

- Arm: `lr_allreduce`
- C1 AllGather source rank: `896`
- Communication-matched AllReduce rank: `1792`
- Ideal ring bytes/row/rank: `5376`
- MCQ average: `0.70577180`

| Task | Metric | Accuracy |
|---|---|---:|
| arc_easy | acc_norm | 0.81313131 |
| arc_challenge | acc_norm | 0.56911263 |
| hellaswag | acc_norm | 0.78062139 |
| piqa | acc_norm | 0.79978237 |
| winogrande | acc | 0.73559590 |
| boolq | acc | 0.82415902 |
| openbookqa | acc_norm | 0.41800000 |

## Command

```bash
evaluation/eval_qwen3_8b_wo_c1_ag_ar_mcq.py --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --factor-dir results/checkpoints/qwen3_8b_wo_c1_uniform_r896_s20 --arm lr_allreduce --output-dir results/evaluation/qwen3_8b_wo_c1_ag_ar_rank_sweep/r896/lr_allreduce --tasks arc_easy,arc_challenge,hellaswag,piqa,winogrande,boolq,openbookqa --lm-eval-batch-size 8 --ppl-batch-size 1 --device cuda:0 --torch-num-threads 4 --skip-ppl
```
