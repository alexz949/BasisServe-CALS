# LongBench: C1-V96 with Base16/R8 sparse decode

Qwen3-8B-Base, FP16, basis, four independent V100 workers. Same192 frozen prompts, six tasks x32.
V96 arms use FP16 memory-efficient C1 prefill. Dense BF16 is contextual, not precision-matched.

| Task | Dense BF16 (context) | FP16 V96 full exact K | FP16 V96 Base16/prompt-R8 sparse |
|---|---:|---:|---:|
| qasper | 39.3018 | 32.4596 | 31.1859 |
| multifieldqa_en | 52.8498 | 40.8995 | 44.4284 |
| hotpotqa | 60.8872 | 56.2213 | 57.2629 |
| 2wikimqa | 50.1190 | 43.8690 | 43.8690 |
| gov_report | 29.1896 | 30.3007 | 29.0060 |
| qmsum | 26.1460 | 27.2612 | 26.4968 |
| Mean | 43.0822 | 38.5019 | 38.7082 |

Sparse: all36 layers, Page32/B2048, pinned page0, no adaptive budget or forced current page.
Selected exact K and resident C1-V96 payload. GPU-resident Base128+R8 sidecars: accuracy oracle, not offload/latency benchmark.
Matched Base16 fitted by closed-form affine MSE RRR. R8 is fitted per prompt using FP32 shared-query covariance and residual covariance eigensolves. No BCD or Adam; E/U frozen during decode.
Base uses the existing C4 bank; frozen C1 uses C4 32x32K fit and4x32K diagnostic. Residual uses current prompt activations only, not benchmark answers.
QA F1 and summary ROUGE-L, scores0–100, six-task arithmetic mean. Not full LongBench; actual inputs1192–30431, total cap32K.
Greedy sampling, original EOS and task caps. Old baselines reused without modification.

First-token agreement with full-K V96: 192/192.
Generation-cap exits without EOS: 36/192.
Peak allocated GPU memory: 25.246 GiB.

## Paired score changes

```json
{
  "dense": {
    "improvements": 49,
    "regressions": 73,
    "ties": 70,
    "mean_delta_pp": -4.374041920029541
  },
  "v96_full": {
    "improvements": 30,
    "regressions": 37,
    "ties": 125,
    "mean_delta_pp": 0.20631015129567487
  }
}
```

Commands and protocol: docs/longbench_c1_v96_spectral_protocol.md. Exact commands preserved per sample.
