# LongBench: two-sided KL C1 with matched Base16 + residual R8

Qwen3-8B-Base, basis, BF16, four L40S workers. All arms use original dense prefill. Frozen 192 prompts, six tasks with32 prompts each; official QA F1 / summary ROUGE-L on0–100 scale. Six-task arithmetic mean. Not full LongBench.

| Task | Dense | Uniform80 full K | KL avg80 full K | KL avg80 Base16/R8 sparse |
|---|---:|---:|---:|---:|
| qasper | 39.3018 | 34.9041 | 35.2802 | 34.9063 |
| multifieldqa_en | 52.8498 | 47.9694 | 48.6936 | 49.2924 |
| hotpotqa | 60.8872 | 59.3363 | 62.1941 | 62.6062 |
| 2wikimqa | 50.1190 | 47.2545 | 46.9940 | 46.9940 |
| gov_report | 29.1896 | 29.3136 | 28.6770 | 28.8272 |
| qmsum | 26.1460 | 27.3302 | 26.9735 | 26.2185 |
| Mean | 43.0822 | 41.0180 | 41.4687 | 41.4741 |

C1 is the unchanged alpha1 two-sided-KL allocation, average rank80. Dense V128 cache is projected into each layer's actual C1 coordinates after prefill. First token is dense argmax.
Base16 was refitted by closed-form affine MSE reduced-rank regression in the selected C1 coordinates. Residual R8 was refitted with non-sink Page-Fisher,40 BCD sweeps and PCG. No Adam or model-weight training.
Router calibration: existing C4 captures,64 x32K fit and16 x32K diagnostic. Existing Query-Gram Q32 positions,8 per8K stratum; positions and C1 factors are not reselected. Diagnostic windows do not select the final factors.
Sparse decode on all36 layers: Page32, physical token budget2048 (64 pages), pinned prefix page0, no adaptive budget or forced current page. Exact K and resident C1 values are used within selected support.
Same prompts, greedy decoding, EOS and task caps as full-K controls. Input-plus-generation cap32K; actual prompts1,192–30,431 tokens.
Accuracy oracle: GPU-resident exact K and materialized Base128+R8 sidecar, not actual CPU offload or a latency benchmark. Old uniform80 router weights were NOT attached to incompatible allocated coordinates.
The old uniform versus KL checkpoint qualification still applies: KL export includes encoder gauge canonicalization and decoder closure. The KL sparse/full comparison uses exactly the same exported payload.

First-token agreement: 192/192.
Cap exits without EOS: 37/192.
Peak allocated memory: 26.060 GiB.

All 36 router layers passed factor-hash, provenance, shape and finite-value checks. Full-fit workers took 14:27, 16:01, 16:20 and 14:27, reusing the two prior full-protocol layer checks. Decode smoke passed. Formal LongBench workers took 6:12, 5:15, 7:12 and 9:12; CPU scoring/audit took 18 seconds. All jobs exited 0 without retries. No non-finite logits or assertion failures occurred. Commands and complete settings are recorded in `docs/longbench_c1_kl_base_residual_protocol.md`.

The near-equal aggregate sparse/full-K score does not imply identical predictions: 34 samples improved, 36 regressed and 122 tied relative to KL full-K. This remains a 192-prompt pilot, not a general accuracy-equivalence claim.

## Paired score changes

```json
{
  "dense": {
    "improvements": 43,
    "regressions": 54,
    "ties": 95,
    "mean_delta_pp": -1.6081259428179475
  },
  "uniform_full": {
    "improvements": 44,
    "regressions": 39,
    "ties": 109,
    "mean_delta_pp": 0.45609670986377776
  },
  "kl_full": {
    "improvements": 34,
    "regressions": 36,
    "ties": 122,
    "mean_delta_pp": 0.005358040538188139
  }
}
```
