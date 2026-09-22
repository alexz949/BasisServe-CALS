# STAR-KV V-only adaptive MCQ

- Target compressed-layer mean rank: `128`
- Actual all-layer V-cache compression: `80.2083%`
- Seven-task average accuracy: `0.49468262`
- Paper six-task average accuracy: `0.47947427`

| Task | Metric | Accuracy |
|---|---|---:|
| arc_easy | acc_norm | 0.53956229 |
| arc_challenge | acc_norm | 0.31569966 |
| hellaswag | acc_norm | 0.48068114 |
| piqa | acc_norm | 0.66430903 |
| winogrande | acc | 0.54459353 |
| boolq | acc | 0.58593272 |
| openbookqa | acc_norm | 0.33200000 |

## Command

```bash
evaluation/eval_starkv_v_adaptive_mcq.py --checkpoint ICLR-results/qwen3-8b/star-v-adaptive-r128/full --output-dir ICLR-results/qwen3-8b/star-v-adaptive-r128/mcq/full --lm-eval-batch-size 8
```
