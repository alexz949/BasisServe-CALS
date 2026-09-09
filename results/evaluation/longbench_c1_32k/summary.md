# C1-V80 LongBench-v1: four-arm32K-cap pilot

192 fixed prompts, six tasks x32; all four arms share full-C1 prefill. Only decode attention differs. Frozen Base16/R8 checkpoints; no benchmark fitting.

| Task | Full exact K | Sparse exact K | Query-Gram Q32 | Terminal Q32 |
|---|---:|---:|---:|---:|
| qasper | 19.6094 | 18.8967 | 19.5324 | 19.7245 |
| multifieldqa_en | 30.7387 | 29.1173 | 29.8343 | 29.3452 |
| hotpotqa | 29.7403 | 32.8628 | 36.3228 | 30.6941 |
| 2wikimqa | 31.2642 | 37.0228 | 31.5359 | 33.3671 |
| gov_report | 27.2987 | 30.0715 | 29.4190 | 27.8504 |
| qmsum | 26.3270 | 26.8727 | 26.1977 | 25.4020 |
| Mean | 27.4964 | 29.1406 | 28.8070 | 27.7305 |

Scores are0–100; QA uses official F1, summaries use official ROUGE-L, best alternative reference. These are not all accuracies.

Actual input lengths: {'min': 1192, 'max': 30431, 'mean': 9244.588541666666, 'truncated': 0, 'at_most_2048': 6}.32K is the total input+reserved-output cap, not a fixed prompt length. Token-level middle truncation is recorded per sample; no padding.

Page32/B2048 with pinned page0; exact sparse uses FP32 exact-QK selection and the same BF16 exact-K/C1-V payload path. Full exact K uses SDPA decode. All36 layers included.
Query-Gram means32 calibration Q across four8K bins; terminal means32 calibration Q in the final8K. Both use Base16+R8. These are not terminal-Q8 checkpoints.
Qwen3-8B-Base, official completion prompts, greedy/task-specific caps. This is a6-task pilot, not full LongBench or LongBench-E. The four-arm mean is task-arithmetic, not an official all-task leaderboard score.

## Paired score changes

{
  "sparse_exact_k_vs_full_exact_k": {
    "improvements": 41,
    "regressions": 37,
    "ties": 114,
    "mean_delta_pp": 1.6442569818405843
  },
  "qgram32_vs_sparse_exact_k": {
    "improvements": 41,
    "regressions": 44,
    "ties": 107,
    "mean_delta_pp": -0.33362031299081707
  },
  "terminal32_vs_sparse_exact_k": {
    "improvements": 36,
    "regressions": 50,
    "ties": 106,
    "mean_delta_pp": -1.4100916162128705
  },
  "qgram32_vs_terminal32": {
    "improvements": 50,
    "regressions": 31,
    "ties": 111,
    "mean_delta_pp": 1.0764713032220534
  }
}

Environment: basis; four L40S workers. GPU-resident accuracy oracle, not a PCIe performance benchmark. Each sample JSON preserves exact command, provenance, tokens and scores.
