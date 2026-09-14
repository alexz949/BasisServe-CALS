# Nemotron-H-47B-Reasoning-128K GQA C1 V64 joint fit

- K remains dense; every one of 8 physical V heads retains rank 64/128.
- Total KV-cache retention is 75% (25% reduction).
- Encoder initialization: `activation-weighted-svd`.
- Checkpoint policy: `fixed decoder-refitted endpoint after encoder sweep 6`.
- V encoders and head output decoders use a full-layer joint solve with cross-head covariance.
- Fit contexts: 256; held-out diagnostic contexts: 64.
- Mean held-out factor-dtype relative MSE: `0.107235436`.

| Layer | Exported boundary | Sweep | Held-out relative MSE |
|---:|:---|---:|---:|
| 17 | after_redecoder | 12 | 0.121468942 |
| 38 | after_redecoder | 12 | 0.0863153368 |
| 49 | after_redecoder | 12 | 0.109633919 |
| 60 | after_redecoder | 12 | 0.103229858 |
| 86 | after_redecoder | 12 | 0.115529124 |
