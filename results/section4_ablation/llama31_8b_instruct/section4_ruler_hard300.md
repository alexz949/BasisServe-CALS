# Section 4 Hard-Task RULER-64K Objective Selection

Six hard tasks × 50 independent seed-43 prompts. Dense V128; B16R16; B2048; Page32; sink32/recent64.

| Task | Page-Fisher | Score-MSE | Fixed Score-only | QGram Score-only |
|---|---:|---:|---:|---:|
| niah_multiquery | 90.0000 | 90.5000 | 90.0000 | 89.5000 |
| vt | 71.6000 | 70.0000 | 70.8000 | 70.8000 |
| cwe | 0.8000 | 0.6000 | 0.6000 | 0.6000 |
| fwe | 72.6667 | 70.6667 | 72.0000 | 72.6667 |
| qa_1 | 68.0000 | 68.0000 | 68.0000 | 70.0000 |
| qa_2 | 54.0000 | 54.0000 | 54.0000 | 54.0000 |
| Mean | 59.5111 | 58.9611 | 59.2333 | 59.5944 |

## Paired comparisons against Page-Fisher

```json
{
  "score_mse_vs_page_fisher": {
    "wins": 9,
    "losses": 17,
    "ties": 274,
    "delta_candidate_minus_page_fisher": -0.5500000000000114,
    "stratified_paired_bootstrap_95_ci": [
      -1.5166666666666668,
      0.4388888888888889
    ],
    "bootstrap_replicates": 10000,
    "bootstrap_seed": 20260918,
    "significantly_worse_than_page_fisher": false
  },
  "score_only_vs_page_fisher": {
    "wins": 7,
    "losses": 11,
    "ties": 282,
    "delta_candidate_minus_page_fisher": -0.2777777777777857,
    "stratified_paired_bootstrap_95_ci": [
      -1.0333333333333334,
      0.4777777777777779
    ],
    "bootstrap_replicates": 10000,
    "bootstrap_seed": 20260918,
    "significantly_worse_than_page_fisher": false
  },
  "qgram_score_only_vs_page_fisher": {
    "wins": 10,
    "losses": 12,
    "ties": 278,
    "delta_candidate_minus_page_fisher": 0.08333333333332149,
    "stratified_paired_bootstrap_95_ci": [
      -1.0444444444444443,
      1.2611111111111113
    ],
    "bootstrap_replicates": 10000,
    "bootstrap_seed": 20260918,
    "significantly_worse_than_page_fisher": false
  }
}
```
