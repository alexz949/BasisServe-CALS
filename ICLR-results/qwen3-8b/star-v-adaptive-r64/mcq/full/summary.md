# STAR-KV V-only adaptive MCQ

- Target compressed-layer mean rank: `64`
- Actual all-layer V-cache compression: `85.9375%`
- Seven-task average accuracy: `0.39123655`
- Paper six-task average accuracy: `0.39155987`

| Task | Metric | Accuracy |
|---|---|---:|
| arc_easy | acc_norm | 0.38930976 |
| arc_challenge | acc_norm | 0.23890785 |
| hellaswag | acc_norm | 0.34146584 |
| piqa | acc_norm | 0.59412405 |
| winogrande | acc | 0.50355170 |
| boolq | acc | 0.38929664 |
| openbookqa | acc_norm | 0.28200000 |

## Command

```bash
evaluation/eval_starkv_v_adaptive_mcq.py --checkpoint ICLR-results/qwen3-8b/star-v-adaptive-r64/full --output-dir ICLR-results/qwen3-8b/star-v-adaptive-r64/mcq/full --lm-eval-batch-size 8
```
