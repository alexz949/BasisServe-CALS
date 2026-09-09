# LongBench: C1-V96 with Base16/R8 sparse decode

Qwen3-8B-Base, BF16, basis, four independent L40S workers. Same192 frozen prompts, six tasks x32.
Both V96 arms use the same full causal C1-V96 Triton prefill. Only decode changes; no dense-prefill V96 arm.

| Task | Dense | V96 full exact K | V96 Base16/R8 sparse |
|---|---:|---:|---:|
| qasper | 39.3018 | 32.4074 | 31.0890 |
| multifieldqa_en | 52.8498 | 42.8418 | 38.2836 |
| hotpotqa | 60.8872 | 57.9676 | 56.4416 |
| 2wikimqa | 50.1190 | 40.2530 | 41.9792 |
| gov_report | 29.1896 | 30.5532 | 28.6572 |
| qmsum | 26.1460 | 27.6797 | 25.8794 |
| Mean | 43.0822 | 38.6171 | 37.0550 |

Sparse: all36 layers, Page32/B2048, pinned page0, no adaptive budget or forced current page.
Selected exact K and resident C1-V96 payload. GPU-resident Base128+R8 sidecars: accuracy oracle, not offload/latency benchmark.
Matched Base16 fitted by closed-form affine MSE RRR. Matched R8 fitted with non-sink Page-Fisher, Q32 across four8K bins,40 BCD sweeps. No Adam.
Router uses C4 64x32K fit and16x32K diagnostic; frozen C1 uses32x32K fit and4x32K diagnostic. No benchmark fitting.
QA F1 and summary ROUGE-L, scores0–100, six-task arithmetic mean. Not full LongBench; actual inputs1192–30431, total cap32K.
Greedy sampling, original EOS and task caps. Old baselines reused without modification.

First-token agreement with full-K V96: 192/192.
Generation-cap exits without EOS: 38/192.
Peak allocated GPU memory: 25.309 GiB.

## Paired score changes

```json
{
  "dense": {
    "improvements": 44,
    "regressions": 81,
    "ties": 67,
    "mean_delta_pp": -6.02723095112092
  },
  "v96_full": {
    "improvements": 27,
    "regressions": 54,
    "ties": 111,
    "mean_delta_pp": -1.5621227854144166
  }
}
```

Commands and protocol: docs/longbench_c1_v96_router_protocol.md. Exact commands preserved per sample.
