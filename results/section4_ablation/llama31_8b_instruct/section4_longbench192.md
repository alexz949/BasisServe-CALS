# Section 4 Dense-V LongBench-32K

Six tasks × 32 paired seed-43 prompts. Llama-3.1-8B-Instruct; Dense V128; B2048; Page32; sink32/recent64.

| Task | Dense Full | Exact-K | Page-Fisher | Score-MSE | Fixed Score-only | QGram Score-only |
|---|---:|---:|---:|---:|---:|---:|
| qasper | 36.4999 | 35.3527 | 35.7259 | 35.6452 | 36.3239 | 36.3239 |
| multifieldqa_en | 54.3941 | 55.2151 | 56.3970 | 56.8175 | 56.7685 | 56.8175 |
| hotpotqa | 39.8602 | 39.8602 | 40.3147 | 40.7612 | 40.3147 | 40.3147 |
| 2wikimqa | 34.6433 | 36.9792 | 34.9905 | 36.9792 | 34.9905 | 36.9792 |
| gov_report | 34.9192 | 34.7354 | 34.5348 | 34.5523 | 34.6514 | 34.6521 |
| qmsum | 25.1378 | 25.0725 | 25.2411 | 25.6140 | 25.2653 | 25.3970 |
| Mean | 37.5758 | 37.8692 | 37.8673 | 38.3949 | 38.0524 | 38.4141 |

## Paired comparisons

```json
{
  "exact_k_vs_dense_full": {
    "wins": 33,
    "losses": 35,
    "ties": 124,
    "macro_mean_delta": 0.2934049877510745,
    "stratified_paired_bootstrap_95_ci": [
      -0.37564629429045954,
      1.155629471423032
    ],
    "bootstrap_replicates": 10000,
    "bootstrap_seed": 20260918
  },
  "page_fisher_vs_dense_full": {
    "wins": 37,
    "losses": 30,
    "ties": 125,
    "macro_mean_delta": 0.2915787086068846,
    "stratified_paired_bootstrap_95_ci": [
      -0.2989556493373077,
      1.0127614851868239
    ],
    "bootstrap_replicates": 10000,
    "bootstrap_seed": 20260918
  },
  "score_mse_vs_dense_full": {
    "wins": 40,
    "losses": 35,
    "ties": 117,
    "macro_mean_delta": 0.8191282612866999,
    "stratified_paired_bootstrap_95_ci": [
      -0.032004330346238646,
      1.825955800761146
    ],
    "bootstrap_replicates": 10000,
    "bootstrap_seed": 20260918
  },
  "score_only_vs_dense_full": {
    "wins": 36,
    "losses": 32,
    "ties": 124,
    "macro_mean_delta": 0.47662703643909765,
    "stratified_paired_bootstrap_95_ci": [
      -0.12410271091195411,
      1.2184283244849938
    ],
    "bootstrap_replicates": 10000,
    "bootstrap_seed": 20260918
  },
  "qgram_score_only_vs_dense_full": {
    "wins": 39,
    "losses": 35,
    "ties": 118,
    "macro_mean_delta": 0.838308905870548,
    "stratified_paired_bootstrap_95_ci": [
      0.05421689418063736,
      1.8284077381922803
    ],
    "bootstrap_replicates": 10000,
    "bootstrap_seed": 20260918
  },
  "score_mse_vs_page_fisher": {
    "wins": 34,
    "losses": 29,
    "ties": 129,
    "macro_mean_delta": 0.5275495526798153,
    "stratified_paired_bootstrap_95_ci": [
      -0.15263301773816557,
      1.3963176164067856
    ],
    "bootstrap_replicates": 10000,
    "bootstrap_seed": 20260918
  },
  "score_only_vs_page_fisher": {
    "wins": 33,
    "losses": 28,
    "ties": 131,
    "macro_mean_delta": 0.18504832783221303,
    "stratified_paired_bootstrap_95_ci": [
      -0.19339482446998368,
      0.5823653685946989
    ],
    "bootstrap_replicates": 10000,
    "bootstrap_seed": 20260918
  },
  "qgram_score_only_vs_page_fisher": {
    "wins": 36,
    "losses": 26,
    "ties": 130,
    "macro_mean_delta": 0.5467301972636633,
    "stratified_paired_bootstrap_95_ci": [
      -0.07455528642843544,
      1.3764594440677465
    ],
    "bootstrap_replicates": 10000,
    "bootstrap_seed": 20260918
  }
}
```
