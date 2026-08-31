# Qwen3.5 MLP pre-down Top-K WikiText-2 pilot

Top-K is applied to the exact SwiGLU product immediately before every MLP down projection. The oracle zero-fills and executes the original dense projection. Potential MAC reduction requires a sparse kernel; standard row-parallel output AllReduce traffic is unchanged.

| Variant | K/source | PPL | PPL delta | Delta NLL | Paired SE | Top-1 agreement | Retained energy | Potential down MAC reduction | AllReduce reduction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 1536 | 12.3135 | 0 | 0 | 0 | 1 | 1 | 0% | 0% |
| mlp_topk_50_source_local | 768 | 12.34026 | +0.217% | +0.002170801 | 0.003061053 | 0.940558 | 0.986177 | 50.0% | 0% |
| mlp_topk_75_source_local | 1152 | 12.33347 | +0.162% | +0.001620531 | 0.00109686 | 0.979574 | 0.998910 | 25.0% | 0% |
