# Section 4 Dense-V LongBench-32K

Six tasks × 100 paired seed-43 prompts. Llama-3.1-8B-Instruct; Dense V128; B2048; Page32; sink32/recent64.

| Task | Page-Fisher | QGram Score-only |
|---|---:|---:|
| qasper | 43.4822 | 43.7719 |
| multifieldqa_en | 55.5713 | 55.3030 |
| hotpotqa | 52.7292 | 52.7292 |
| 2wikimqa | 47.9271 | 50.4668 |
| gov_report | 34.1523 | 33.8206 |
| qmsum | 24.2990 | 24.2880 |
| Mean | 43.0269 | 43.3966 |

## Paired comparisons

```json
{
  "qgram_score_only_vs_page_fisher": {
    "wins": 100,
    "losses": 94,
    "ties": 406,
    "macro_mean_delta": 0.3697328735205474,
    "stratified_paired_bootstrap_95_ci": [
      -0.08047650269454083,
      0.8750361417176421
    ],
    "bootstrap_replicates": 10000,
    "bootstrap_seed": 20260918
  }
}
```
