# Nemotron-H-47B-Reasoning-128K GQA C1 V48 joint fit

- K remains dense; every one of 8 physical V heads retains rank 48/128.
- Total KV-cache retention is 68.75% (31.25% reduction).
- Encoder initialization: `activation-weighted-svd`.
- Checkpoint policy: `fixed decoder-refitted endpoint after encoder sweep 6`.
- V encoders and head output decoders use a full-layer joint solve with cross-head covariance.
- Fit contexts: 256; held-out diagnostic contexts: 64.
- Mean held-out factor-dtype relative MSE: `0.150131423`.

| Layer | Exported boundary | Sweep | Held-out relative MSE |
|---:|:---|---:|---:|
| 17 | after_redecoder | 12 | 0.171996074 |
| 38 | after_redecoder | 12 | 0.119775016 |
| 49 | after_redecoder | 12 | 0.152178967 |
| 60 | after_redecoder | 12 | 0.146905113 |
| 86 | after_redecoder | 12 | 0.159801946 |
