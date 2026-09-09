# LongBench-v1 dense K/V and C1-V80 comparison

Qwen3-8B-Base; 192 frozen prompts, six tasks with 32 prompts each. BF16, basis, four L40S workers.

| Task | Dense K + dense V | Full K + V80 | Exact sparse + V80 | Query-Gram Q32 + V80 | Terminal Q32 + V80 |
|---|---:|---:|---:|---:|---:|
| qasper | 39.3018 | 19.6094 | 18.8967 | 19.5324 | 19.7245 |
| multifieldqa_en | 52.8498 | 30.7387 | 29.1173 | 29.8343 | 29.3452 |
| hotpotqa | 60.8872 | 29.7403 | 32.8628 | 36.3228 | 30.6941 |
| 2wikimqa | 50.1190 | 31.2642 | 37.0228 | 31.5359 | 33.3671 |
| gov_report | 29.1896 | 27.2987 | 30.0715 | 29.4190 | 27.8504 |
| qmsum | 26.1460 | 26.3270 | 26.8727 | 26.1977 | 25.4020 |
| Mean | 43.0822 | 27.4964 | 29.1406 | 28.8070 | 27.7305 |

QA: official F1; summaries: official ROUGE-L; scores on a 0–100 scale, not all accuracies.

32K is the input plus reserved generation cap. Actual inputs: 1,192–30,431 tokens; mean 9,244.59; none truncated.
The new baseline uses unmodified dense K128/V128 in BOTH prefill and decode. Its first token is not forced to match C1.
The existing four C1 arms share full-C1 prefill; their attention differences apply during decode. Sparse arms use Page32/B2048 with pinned page 0.
Base16/R8 banks are frozen: Query-Gram uses 8 calibration Q in each of four 8K bins; terminal uses 32 Q in the last 8K.
Same saved input tokens, references, greedy decoding, EOS policy and generation caps (128/64/32/32/512/512). No benchmark fitting or tuning.
This is a six-task pilot, not full LongBench. The dense versus C1 comparison also differs in prefill backend (SDPA versus C1 Triton); it is not a pure matched-kernel V-compression ablation.

## Paired score changes relative to dense K/V

```json
{
  "full_exact_k_vs_dense_k_dense_v": {
    "improvements": 48,
    "regressions": 104,
    "ties": 40,
    "mean_delta_pp": -15.585851987610049
  },
  "sparse_exact_k_vs_dense_k_dense_v": {
    "improvements": 52,
    "regressions": 98,
    "ties": 42,
    "mean_delta_pp": -13.941595005769464
  },
  "qgram32_vs_dense_k_dense_v": {
    "improvements": 51,
    "regressions": 99,
    "ties": 42,
    "mean_delta_pp": -14.275215318760281
  },
  "terminal32_vs_dense_k_dense_v": {
    "improvements": 44,
    "regressions": 106,
    "ties": 42,
    "mean_delta_pp": -15.351686621982335
  }
}
```

All 192 dense predictions were re-decoded and rescored; task means checked against the official scorer. Inputs match the previously independently audited dataset.
Commands and full execution settings are recorded in docs/longbench_dense_protocol.md and each sample JSON.
