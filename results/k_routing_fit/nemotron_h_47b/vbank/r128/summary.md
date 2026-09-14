# Nemotron-H-47B-Reasoning-128K GQA C1 V128 joint fit

- K remains dense; every one of 8 physical V heads retains rank 128/128.
- Total KV-cache retention is 100% (0% reduction).
- Encoder initialization: `activation-weighted-svd`.
- Checkpoint policy: `fixed decoder-refitted endpoint after encoder sweep 6`.
- V encoders and head output decoders use a full-layer joint solve with cross-head covariance.
- Fit contexts: 256; held-out diagnostic contexts: 64.
- Mean held-out factor-dtype relative MSE: `0`.

| Layer | Exported boundary | Sweep | Held-out relative MSE |
|---:|:---|---:|---:|
| 17 | exact_dense_endpoint | 0 | 0 |
| 38 | exact_dense_endpoint | 0 | 0 |
| 49 | exact_dense_endpoint | 0 | 0 |
| 60 | exact_dense_endpoint | 0 | 0 |
| 86 | exact_dense_endpoint | 0 | 0 |
