# Llama-3.1-8B GQA C1 V128 joint fit

- K remains dense; every one of 8 physical V heads retains rank 128/128.
- Total KV-cache retention is 100% (0% reduction).
- Encoder initialization: `activation-weighted-svd`.
- Checkpoint policy: `fixed decoder-refitted endpoint after encoder sweep 6`.
- V encoders and head output decoders use a full-layer joint solve with cross-head covariance.
- Fit contexts: 64; held-out diagnostic contexts: 16.
- Mean held-out factor-dtype relative MSE: `0`.

| Layer | Exported boundary | Sweep | Held-out relative MSE |
|---:|:---|---:|---:|
| 0 | exact_dense_endpoint | 0 | 0 |
| 1 | exact_dense_endpoint | 0 | 0 |
| 2 | exact_dense_endpoint | 0 | 0 |
| 3 | exact_dense_endpoint | 0 | 0 |
| 4 | exact_dense_endpoint | 0 | 0 |
| 5 | exact_dense_endpoint | 0 | 0 |
| 6 | exact_dense_endpoint | 0 | 0 |
| 7 | exact_dense_endpoint | 0 | 0 |
| 8 | exact_dense_endpoint | 0 | 0 |
| 9 | exact_dense_endpoint | 0 | 0 |
| 10 | exact_dense_endpoint | 0 | 0 |
| 11 | exact_dense_endpoint | 0 | 0 |
| 12 | exact_dense_endpoint | 0 | 0 |
| 13 | exact_dense_endpoint | 0 | 0 |
| 14 | exact_dense_endpoint | 0 | 0 |
| 15 | exact_dense_endpoint | 0 | 0 |
| 16 | exact_dense_endpoint | 0 | 0 |
| 17 | exact_dense_endpoint | 0 | 0 |
| 18 | exact_dense_endpoint | 0 | 0 |
| 19 | exact_dense_endpoint | 0 | 0 |
| 20 | exact_dense_endpoint | 0 | 0 |
| 21 | exact_dense_endpoint | 0 | 0 |
| 22 | exact_dense_endpoint | 0 | 0 |
| 23 | exact_dense_endpoint | 0 | 0 |
| 24 | exact_dense_endpoint | 0 | 0 |
| 25 | exact_dense_endpoint | 0 | 0 |
| 26 | exact_dense_endpoint | 0 | 0 |
| 27 | exact_dense_endpoint | 0 | 0 |
| 28 | exact_dense_endpoint | 0 | 0 |
| 29 | exact_dense_endpoint | 0 | 0 |
| 30 | exact_dense_endpoint | 0 | 0 |
| 31 | exact_dense_endpoint | 0 | 0 |
