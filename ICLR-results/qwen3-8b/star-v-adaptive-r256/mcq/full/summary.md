# STAR-KV V-only adaptive MCQ

- Target compressed-layer mean rank: `256`
- Actual all-layer V-cache compression: `68.7500%`
- Seven-task average accuracy: `0.64128856`
- Paper six-task average accuracy: `0.62630454`

| Task | Metric | Accuracy |
|---|---|---:|
| arc_easy | acc_norm | 0.76094276 |
| arc_challenge | acc_norm | 0.50170648 |
| hellaswag | acc_norm | 0.69856602 |
| piqa | acc_norm | 0.75788901 |
| winogrande | acc | 0.65272297 |
| boolq | acc | 0.73119266 |
| openbookqa | acc_norm | 0.38600000 |

## Command

```bash
evaluation/eval_starkv_v_adaptive_mcq.py --checkpoint ICLR-results/qwen3-8b/star-v-adaptive-r256/full --output-dir ICLR-results/qwen3-8b/star-v-adaptive-r256/mcq/full --lm-eval-batch-size 8
```
