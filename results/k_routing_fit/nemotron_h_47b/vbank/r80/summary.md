# Nemotron-H-47B-Reasoning-128K GQA C1 V80 joint fit

- K remains dense; every one of 8 physical V heads retains rank 80/128.
- Total KV-cache retention is 81.25% (18.75% reduction).
- Encoder initialization: `activation-weighted-svd`.
- Checkpoint policy: `fixed decoder-refitted endpoint after encoder sweep 6`.
- V encoders and head output decoders use a full-layer joint solve with cross-head covariance.
- Fit contexts: 256; held-out diagnostic contexts: 64.
- Mean held-out factor-dtype relative MSE: `0.0722608994`.

| Layer | Exported boundary | Sweep | Held-out relative MSE |
|---:|:---|---:|---:|
| 17 | after_redecoder | 12 | 0.0809630444 |
| 38 | after_redecoder | 12 | 0.0591029797 |
| 49 | after_redecoder | 12 | 0.0749319474 |
| 60 | after_redecoder | 12 | 0.0688445269 |
| 86 | after_redecoder | 12 | 0.0774619985 |
