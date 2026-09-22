# STAR-KV V-only adaptive MCQ

- Average accuracy: `0.69226854`
- Actual V-cache compression: `45.9527%`

| Task | Metric | Accuracy |
|---|---|---:|
| arc_easy | acc_norm | 0.78914141 |
| arc_challenge | acc_norm | 0.53754266 |
| hellaswag | acc_norm | 0.76807409 |
| piqa | acc_norm | 0.79053319 |
| winogrande | acc | 0.72296764 |
| boolq | acc | 0.81162080 |
| openbookqa | acc_norm | 0.42600000 |

## Command

```bash
evaluation/eval_starkv_v50_adaptive_mcq.py --checkpoint ICLR-results/qwen3-8b/star-v50-adaptive/full --output-dir ICLR-results/qwen3-8b/star-v50-adaptive/mcq/full --lm-eval-batch-size 8
```
