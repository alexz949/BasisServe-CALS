# Qwen3.5 MLP pre-down Top-K WikiText-2 pilot

Top-K is applied to the exact SwiGLU product immediately before every MLP down projection. The oracle zero-fills and executes the original dense projection. Potential MAC reduction requires a sparse kernel; standard row-parallel output AllReduce traffic is unchanged.

| Variant | K/source | PPL | PPL delta | Delta NLL | Paired SE | Top-1 agreement | Retained energy | Potential down MAC reduction | AllReduce reduction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 1536 | 12.3135 | 0 | 0 | 0 | 1 | 1 | 0% | 0% |
| mlp_topk_25_source_local | 384 | 12.95007 | +5.170% | +0.05040503 | 0.006744567 | 0.845523 | 0.923648 | 75.0% | 0% |
| mlp_topk_37p5_source_local | 576 | 12.51042 | +1.599% | +0.01586604 | 0.004756581 | 0.905333 | 0.966023 | 62.5% | 0% |
