# LongBench: C1-V96 with Base16/R16 sparse decode

Qwen3-8B-Base, BF16, basis, four independent L40S workers. Same192 frozen prompts, six tasks x32.
Both V96 arms use the same full causal C1-V96 Triton prefill. Only decode changes; no dense-prefill V96 arm.

| Task | Dense | V96 full exact K | V96 offline R8 | V96 offline R16 |
|---|---:|---:|---:|---:|
| qasper | 39.3018 | 32.4074 | 31.0890 | 31.6381 |
| multifieldqa_en | 52.8498 | 42.8418 | 38.2836 | 41.5424 |
| hotpotqa | 60.8872 | 57.9676 | 56.4416 | 56.4416 |
| 2wikimqa | 50.1190 | 40.2530 | 41.9792 | 40.2530 |
| gov_report | 29.1896 | 30.5532 | 28.6572 | 31.0985 |
| qmsum | 26.1460 | 27.6797 | 25.8794 | 26.7642 |
| Mean | 43.0822 | 38.6171 | 37.0550 | 37.9563 |

Sparse: all36 layers, Page32/B2048, pinned page0, no adaptive budget or forced current page.
Selected exact K and resident C1-V96 payload. GPU-resident Base128+R16 sidecars: accuracy oracle, not offload/latency benchmark.
Matched Base16 fitted by closed-form affine MSE RRR. Matched R16 fitted with non-sink Page-Fisher, Q32 across four8K bins,40 BCD sweeps. No Adam.
Router uses C4 64x32K fit and16x32K diagnostic; frozen C1 uses32x32K fit and4x32K diagnostic. No benchmark fitting.
QA F1 and summary ROUGE-L, scores0–100, six-task arithmetic mean. Not full LongBench; actual inputs1192–30431, total cap32K.
Greedy sampling, original EOS and task caps. Old baselines reused without modification.

First-token agreement with full-K V96: 192/192.
Generation-cap exits without EOS: 45/192.
Peak allocated GPU memory: 25.454 GiB.

## Paired score changes

```json
{
  "dense": {
    "improvements": 48,
    "regressions": 74,
    "ties": 70,
    "mean_delta_pp": -5.125941032995648
  },
  "v96_full": {
    "improvements": 36,
    "regressions": 41,
    "ties": 115,
    "mean_delta_pp": -0.6608328672891446
  },
  "v96_r8": {
    "improvements": 41,
    "regressions": 33,
    "ties": 118,
    "mean_delta_pp": 0.901289918125272
  }
}
```

Commands and protocol: docs/longbench_c1_v96_r16_protocol.md. Exact commands preserved per sample.
