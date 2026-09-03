# Llama-2-7B MHA C1 V112 joint fit

- K remains dense; every one of 32 physical V heads retains rank 112/128.
- Total KV-cache retention is 93.75% (6.25% reduction).
- Encoder initialization: `activation-weighted-svd`.
- Checkpoint policy: `fixed decoder-refitted endpoint after encoder sweep 6`.
- V encoders and head output decoders use a full-layer joint solve with cross-head covariance.
- Fit contexts: 256; held-out diagnostic contexts: 64.
- Mean held-out factor-dtype relative MSE: `0.0331577547`.

| Layer | Exported boundary | Sweep | Held-out relative MSE |
|---:|:---|---:|---:|
| 0 | after_redecoder | 6 | 1.21001871e-05 |
| 1 | after_redecoder | 6 | 0.00630302031 |
| 2 | after_redecoder | 6 | 0.0160482705 |
| 3 | after_redecoder | 6 | 0.0127168937 |
| 4 | after_redecoder | 6 | 0.0183635822 |
| 5 | after_redecoder | 6 | 0.0174618839 |
| 6 | after_redecoder | 6 | 0.0213895086 |
| 7 | after_redecoder | 6 | 0.0253090118 |
| 8 | after_redecoder | 6 | 0.0291578397 |
| 9 | after_redecoder | 6 | 0.0325540585 |
| 10 | after_redecoder | 6 | 0.0328357429 |
| 11 | after_redecoder | 6 | 0.0351683248 |
| 12 | after_redecoder | 6 | 0.0363006272 |
| 13 | after_redecoder | 6 | 0.0361798248 |
| 14 | after_redecoder | 6 | 0.0397841654 |
| 15 | after_redecoder | 6 | 0.0331866592 |
| 16 | after_redecoder | 6 | 0.0316543987 |
| 17 | after_redecoder | 6 | 0.0419000339 |
| 18 | after_redecoder | 6 | 0.0397304657 |
| 19 | after_redecoder | 6 | 0.0417292218 |
| 20 | after_redecoder | 6 | 0.0340535928 |
| 21 | after_redecoder | 6 | 0.0506342765 |
| 22 | after_redecoder | 6 | 0.0369723218 |
| 23 | after_redecoder | 6 | 0.0533808006 |
| 24 | after_redecoder | 6 | 0.0404059248 |
| 25 | after_redecoder | 6 | 0.0599274507 |
| 26 | after_redecoder | 6 | 0.0364384052 |
| 27 | after_redecoder | 6 | 0.047653123 |
| 28 | after_redecoder | 6 | 0.0485471929 |
| 29 | after_redecoder | 6 | 0.0471666011 |
| 30 | after_redecoder | 6 | 0.0399912002 |
| 31 | after_redecoder | 6 | 0.0180916268 |
