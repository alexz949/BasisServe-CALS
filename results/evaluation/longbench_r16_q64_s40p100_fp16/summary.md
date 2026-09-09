# LongBench: C1-V96 with Base16/R16 sparse decode

Qwen3-8B-Base, FP16, basis, four independent V100 workers. Same192 frozen prompts, six tasks x32.
Full-K and R16 use FP16 memory-efficient prefill. Dense and offline R8 columns are BF16 context only.

| Task | Dense BF16 context | Full-K FP16 | R8 BF16 context | R16 FP16 |
|---|---:|---:|---:|---:|
| qasper | 39.3018 | 32.4596 | 31.0890 | 31.7077 |
| multifieldqa_en | 52.8498 | 40.8995 | 38.2836 | 42.3154 |
| hotpotqa | 60.8872 | 56.2213 | 56.4416 | 56.4416 |
| 2wikimqa | 50.1190 | 43.8690 | 41.9792 | 43.8690 |
| gov_report | 29.1896 | 30.3007 | 28.6572 | 30.9495 |
| qmsum | 26.1460 | 27.2612 | 25.8794 | 26.3186 |
| Mean | 43.0822 | 38.5019 | 37.0550 | 38.6003 |

Sparse: all36 layers, Page32/B2048, pinned page0, no adaptive budget or forced current page.
Selected exact K and resident C1-V96 payload. GPU-resident Base128+R16 sidecars: accuracy oracle, not offload/latency benchmark.
Matched Base16 fitted by closed-form affine MSE RRR. Matched R16 fitted with non-sink Page-Fisher, Q64 across four8K bins,BCD/PCG settings are recorded in bank_protocol. No Adam.
Router uses C4 64x32K fit and16x32K diagnostic; frozen C1 uses32x32K fit and4x32K diagnostic. No benchmark fitting.
QA F1 and summary ROUGE-L, scores0–100, six-task arithmetic mean. Not full LongBench; actual inputs1192–30431, total cap32K.
Greedy sampling, original EOS and task caps. Old baselines reused without modification.

First-token agreement with full-K V96: 192/192.
Generation-cap exits without EOS: 42/192.
Peak allocated GPU memory: 25.455 GiB.

## Paired score changes

```json
{
  "dense": {
    "improvements": 50,
    "regressions": 71,
    "ties": 71,
    "mean_delta_pp": -4.481933308594414
  },
  "v96_full": {
    "improvements": 34,
    "regressions": 32,
    "ties": 126,
    "mean_delta_pp": 0.09841876273080175
  },
  "v96_r8": {
    "improvements": 48,
    "regressions": 29,
    "ties": 115,
    "mean_delta_pp": 1.5452976425265064
  }
}
```

Commands and protocol: docs/longbench_c1_v96_r16_protocol.md. Exact commands preserved per sample.
