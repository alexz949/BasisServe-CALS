# Nemotron-H-47B-Reasoning-128K GQA C1 V112 joint fit

- K remains dense; every one of 8 physical V heads retains rank 112/128.
- Total KV-cache retention is 93.75% (6.25% reduction).
- Encoder initialization: `activation-weighted-svd`.
- Checkpoint policy: `fixed decoder-refitted endpoint after encoder sweep 6`.
- V encoders and head output decoders use a full-layer joint solve with cross-head covariance.
- Fit contexts: 256; held-out diagnostic contexts: 64.
- Mean held-out factor-dtype relative MSE: `0.0191280646`.

| Layer | Exported boundary | Sweep | Held-out relative MSE |
|---:|:---|---:|---:|
| 17 | after_redecoder | 12 | 0.0214979621 |
| 38 | after_redecoder | 12 | 0.0166496623 |
| 49 | after_redecoder | 12 | 0.0206663715 |
| 60 | after_redecoder | 12 | 0.0178802309 |
| 86 | after_redecoder | 12 | 0.0189460961 |
