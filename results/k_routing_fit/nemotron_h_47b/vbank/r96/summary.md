# Nemotron-H-47B-Reasoning-128K GQA C1 V96 joint fit

- K remains dense; every one of 8 physical V heads retains rank 96/128.
- Total KV-cache retention is 87.5% (12.5% reduction).
- Encoder initialization: `activation-weighted-svd`.
- Checkpoint policy: `fixed decoder-refitted endpoint after encoder sweep 6`.
- V encoders and head output decoders use a full-layer joint solve with cross-head covariance.
- Fit contexts: 256; held-out diagnostic contexts: 64.
- Mean held-out factor-dtype relative MSE: `0.0433055514`.

| Layer | Exported boundary | Sweep | Held-out relative MSE |
|---:|:---|---:|---:|
| 17 | after_redecoder | 12 | 0.0482991721 |
| 38 | after_redecoder | 12 | 0.03620465 |
| 49 | after_redecoder | 12 | 0.0456593037 |
| 60 | after_redecoder | 12 | 0.0409269026 |
| 86 | after_redecoder | 12 | 0.0454377284 |
