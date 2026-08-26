# Qwen3-8B-Base GQA C1 V32 joint fit

- K remains dense; every one of 8 physical V heads retains rank 32/128.
- Total KV-cache retention is 62.5% (37.5% reduction).
- Encoder initialization: `activation-weighted-svd`.
- Held-out selection boundaries: `decoder-closed`.
- V encoders and head output decoders use a full-layer joint solve with cross-head covariance.
- Fit contexts: 256; held-out selection contexts: 64.
- Mean held-out factor-dtype relative MSE: `0.267109482`.

| Layer | Selected boundary | Sweep | Held-out relative MSE |
|---:|:---|---:|---:|
| 0 | after_redecoder | 5 | 0.0419048675 |
| 1 | after_redecoder | 5 | 0.138171498 |
| 2 | after_redecoder | 5 | 0.158872779 |
| 3 | after_redecoder | 5 | 0.153208034 |
| 4 | after_redecoder | 5 | 0.205346518 |
| 5 | after_redecoder | 5 | 0.193811864 |
| 6 | after_redecoder | 5 | 0.262139752 |
| 7 | after_redecoder | 5 | 0.289999143 |
| 8 | after_redecoder | 5 | 0.331243464 |
| 9 | after_redecoder | 5 | 0.367169078 |
| 10 | after_redecoder | 5 | 0.338606897 |
| 11 | after_redecoder | 5 | 0.322110052 |
| 12 | after_redecoder | 5 | 0.237478651 |
| 13 | after_redecoder | 5 | 0.263121963 |
| 14 | after_redecoder | 5 | 0.30479907 |
| 15 | after_redecoder | 5 | 0.288846279 |
| 16 | after_redecoder | 5 | 0.302158938 |
| 17 | after_redecoder | 5 | 0.220260705 |
| 18 | after_redecoder | 5 | 0.267262586 |
| 19 | after_redecoder | 5 | 0.213220733 |
| 20 | after_redecoder | 5 | 0.286971583 |
| 21 | after_redecoder | 5 | 0.3072126 |
| 22 | after_redecoder | 5 | 0.275285685 |
| 23 | after_redecoder | 5 | 0.307101588 |
| 24 | after_redecoder | 5 | 0.240632761 |
| 25 | after_redecoder | 5 | 0.297303455 |
| 26 | after_redecoder | 5 | 0.374299262 |
| 27 | after_redecoder | 5 | 0.335720473 |
| 28 | after_redecoder | 5 | 0.324580982 |
| 29 | after_redecoder | 5 | 0.385390941 |
| 30 | after_redecoder | 5 | 0.279414371 |
| 31 | after_redecoder | 5 | 0.372996907 |
| 32 | after_redecoder | 5 | 0.285771773 |
| 33 | after_redecoder | 5 | 0.366711674 |
| 34 | after_redecoder | 5 | 0.169046327 |
| 35 | after_redecoder | 5 | 0.107768091 |
