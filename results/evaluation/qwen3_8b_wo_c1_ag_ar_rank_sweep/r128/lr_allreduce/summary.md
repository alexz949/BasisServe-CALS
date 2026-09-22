# Qwen3-8B Wo-only AllGather/AllReduce quality

- Arm: `lr_allreduce`
- C1 AllGather source rank: `128`
- Communication-matched AllReduce rank: `256`
- Ideal ring bytes/row/rank: `768`
- WikiText-2 PPL: `107.06951366`
- MCQ average: `0.38363266`

| Task | Metric | Accuracy |
|---|---|---:|
| arc_easy | acc_norm | 0.28998316 |
| arc_challenge | acc_norm | 0.24744027 |
| hellaswag | acc_norm | 0.30312687 |
| piqa | acc_norm | 0.54080522 |
| winogrande | acc | 0.54301500 |
| boolq | acc | 0.50305810 |
| openbookqa | acc_norm | 0.25800000 |

## Command

```bash
evaluation/eval_qwen3_8b_wo_c1_ag_ar_mcq.py --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --factor-dir results/checkpoints/qwen3_8b_wo_c1_uniform_r128_s20 --arm lr_allreduce --output-dir results/evaluation/qwen3_8b_wo_c1_ag_ar_rank_sweep/r128/lr_allreduce --tasks arc_easy,arc_challenge,hellaswag,piqa,winogrande,boolq,openbookqa --lm-eval-batch-size 8 --ppl-batch-size 1 --device cuda:0 --torch-num-threads 4
```
