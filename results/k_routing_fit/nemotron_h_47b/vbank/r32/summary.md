# Nemotron-H-47B-Reasoning-128K GQA C1 V32 joint fit

- K remains dense; every one of 8 physical V heads retains rank 32/128.
- Total KV-cache retention is 62.5% (37.5% reduction).
- Encoder initialization: `activation-weighted-svd`.
- Checkpoint policy: `fixed decoder-refitted endpoint after encoder sweep 6`.
- V encoders and head output decoders use a full-layer joint solve with cross-head covariance.
- Fit contexts: 256; held-out diagnostic contexts: 64.
- Mean held-out factor-dtype relative MSE: `0.205890862`.

| Layer | Exported boundary | Sweep | Held-out relative MSE |
|---:|:---|---:|---:|
| 17 | after_redecoder | 12 | 0.238270506 |
| 38 | after_redecoder | 12 | 0.164886757 |
| 49 | after_redecoder | 12 | 0.207763013 |
| 60 | after_redecoder | 12 | 0.205842402 |
| 86 | after_redecoder | 12 | 0.212691632 |
