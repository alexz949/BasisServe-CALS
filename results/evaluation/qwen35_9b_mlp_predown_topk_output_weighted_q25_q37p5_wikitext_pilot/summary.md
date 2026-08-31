# Qwen3.5 MLP pre-down Top-K WikiText-2 pilot

Top-K is applied to the exact SwiGLU product immediately before every MLP down projection. The oracle zero-fills and executes the original dense projection. Potential MAC reduction requires a sparse kernel; standard row-parallel output AllReduce traffic is unchanged.

| Variant | K/source | PPL | PPL delta | Delta NLL | Paired SE | Top-1 agreement | Retained energy | Potential down MAC reduction | AllReduce reduction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 1536 | 12.3135 | 0 | 0 | 0 | 1 | 1 | 0% | 0% |
| mlp_topk_25_output_weighted_source_local | 384 | 13.03516 | +5.861% | +0.05695438 | 0.00706145 | 0.846257 | 0.922910 | 75.0% | 0% |
| mlp_topk_37p5_output_weighted_source_local | 576 | 12.51729 | +1.655% | +0.01641464 | 0.005131453 | 0.906311 | 0.965667 | 62.5% | 0% |
