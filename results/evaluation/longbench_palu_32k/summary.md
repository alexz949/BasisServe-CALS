# LongBench: matched-calibration PaLU M/G4, dense, and C1

Qwen3-8B-Base; 192 fixed prompts, six tasks x 32; BF16, basis, L40S.

| Task | Dense K/V | PaLU M | PaLU G4 | C1 full K | C1 exact sparse | C1 Query-Gram | C1 terminal |
|---|---:|---:|---:|---:|---:|---:|---:|
| qasper | 39.3018 | 27.2131 | 16.9565 | 19.6094 | 18.8967 | 19.5324 | 19.7245 |
| multifieldqa_en | 52.8498 | 32.3183 | 36.6639 | 30.7387 | 29.1173 | 29.8343 | 29.3452 |
| hotpotqa | 60.8872 | 16.1260 | 29.4210 | 29.7403 | 32.8628 | 36.3228 | 30.6941 |
| 2wikimqa | 50.1190 | 28.8711 | 22.7083 | 31.2642 | 37.0228 | 31.5359 | 33.3671 |
| gov_report | 29.1896 | 21.5618 | 25.3024 | 27.2987 | 30.0715 | 29.4190 | 27.8504 |
| qmsum | 26.1460 | 25.2504 | 16.0173 | 26.3270 | 26.8727 | 26.1977 | 25.4020 |
| Mean | 43.0822 | 25.2234 | 24.5116 | 27.4964 | 29.1406 | 28.8070 | 27.7305 |

Scores are official QA F1 or summary ROUGE-L on a 0–100 scale, not all accuracies. Six-task arithmetic mean, not full LongBench.
Exact same saved inputs, references, completion prompts, greedy/EOS policy and task generation caps. Actual prompt lengths 1,192–30,431; 32K is the total cap.
Both PaLU arms use full exact K and their own approximate V throughout prefill/decode. Factors are executed as a BF16 latent writer plus per-group reconstruction, using standard V128 cache and dense SDPA. This is not a compact-cache or speed benchmark.
PaLU uses exactly the C1 32 x 32K fit tokens, matched whitening and the same newly measured Fisher statistics. M actual average rank is 81.7778; G4 is 80.0000. Nominal R80 does not mean identical realized budgets.
C1 uses its previously evaluated full-C1 Triton prefill; its four arms differ only during decode. PaLU and dense use SDPA prefill. Differences against C1 cannot be attributed solely to factor fitting.
No factors or configurations were selected or refit using LongBench results.

## Paired changes

```json
{
  "palu_m_vs_dense": {
    "improvements": 36,
    "regressions": 118,
    "ties": 38,
    "mean_delta_pp": -17.858785650729654
  },
  "palu_m_vs_c1_full": {
    "improvements": 64,
    "regressions": 93,
    "ties": 35,
    "mean_delta_pp": -2.2729336631196095
  },
  "palu_g4_vs_dense": {
    "improvements": 31,
    "regressions": 108,
    "ties": 53,
    "mean_delta_pp": -18.57067214189599
  },
  "palu_g4_vs_c1_full": {
    "improvements": 62,
    "regressions": 85,
    "ties": 45,
    "mean_delta_pp": -2.984820154285952
  }
}
```

All 384 new predictions were decoded and rescored; source identities, eight shard manifests, EOS/caps, and official per-task scoring were checked.
See docs/longbench_palu_protocol.md for settings and exact commands.
