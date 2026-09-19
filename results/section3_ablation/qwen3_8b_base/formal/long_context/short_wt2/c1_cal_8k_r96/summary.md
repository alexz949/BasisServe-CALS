# Qwen3-8B Section-3 short-context quality

- Method: `c1`
- Rank: `96`
- WikiText-2 PPL: `7.22952499`

## Command

```bash
evaluation/eval_qwen3_8b_section3_short.py --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --method c1 --checkpoint-dir results/section3_ablation/qwen3_8b_base/formal/long_context/calibration/8k/c1/r96 --output-dir results/section3_ablation/qwen3_8b_base/formal/long_context/short_wt2/c1_cal_8k_r96 --skip-mcq --ppl-batch-size 2 --device cuda:0 --torch-num-threads 4
```
