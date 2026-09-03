# Llama-2-7B MHA C1 V32 joint fit

- K remains dense; every one of 32 physical V heads retains rank 32/128.
- Total KV-cache retention is 62.5% (37.5% reduction).
- Encoder initialization: `activation-weighted-svd`.
- Checkpoint policy: `fixed decoder-refitted endpoint after encoder sweep 6`.
- V encoders and head output decoders use a full-layer joint solve with cross-head covariance.
- Fit contexts: 256; held-out diagnostic contexts: 64.
- Mean held-out factor-dtype relative MSE: `0.3235352`.

| Layer | Exported boundary | Sweep | Held-out relative MSE |
|---:|:---|---:|---:|
| 0 | after_redecoder | 6 | 0.00278865651 |
| 1 | after_redecoder | 6 | 0.111374626 |
| 2 | after_redecoder | 6 | 0.230743048 |
| 3 | after_redecoder | 6 | 0.155713758 |
| 4 | after_redecoder | 6 | 0.22281588 |
| 5 | after_redecoder | 6 | 0.218101917 |
| 6 | after_redecoder | 6 | 0.240047284 |
| 7 | after_redecoder | 6 | 0.266238651 |
| 8 | after_redecoder | 6 | 0.300436761 |
| 9 | after_redecoder | 6 | 0.329230455 |
| 10 | after_redecoder | 6 | 0.337017656 |
| 11 | after_redecoder | 6 | 0.345993908 |
| 12 | after_redecoder | 6 | 0.359029391 |
| 13 | after_redecoder | 6 | 0.358294655 |
| 14 | after_redecoder | 6 | 0.387359167 |
| 15 | after_redecoder | 6 | 0.349929647 |
| 16 | after_redecoder | 6 | 0.358699594 |
| 17 | after_redecoder | 6 | 0.40983602 |
| 18 | after_redecoder | 6 | 0.401198839 |
| 19 | after_redecoder | 6 | 0.412792088 |
| 20 | after_redecoder | 6 | 0.358329195 |
| 21 | after_redecoder | 6 | 0.465360923 |
| 22 | after_redecoder | 6 | 0.34769772 |
| 23 | after_redecoder | 6 | 0.464121876 |
| 24 | after_redecoder | 6 | 0.361706072 |
| 25 | after_redecoder | 6 | 0.510838893 |
| 26 | after_redecoder | 6 | 0.321934967 |
| 27 | after_redecoder | 6 | 0.411947314 |
| 28 | after_redecoder | 6 | 0.406029549 |
| 29 | after_redecoder | 6 | 0.390458322 |
| 30 | after_redecoder | 6 | 0.337399413 |
| 31 | after_redecoder | 6 | 0.179660164 |
