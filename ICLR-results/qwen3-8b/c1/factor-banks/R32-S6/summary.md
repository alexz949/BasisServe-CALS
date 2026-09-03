# Qwen3-8B-Base GQA C1 V32 joint fit

- K remains dense; every one of 8 physical V heads retains rank 32/128.
- Total KV-cache retention is 62.5% (37.5% reduction).
- Encoder initialization: `activation-weighted-svd`.
- Checkpoint policy: `fixed decoder-refitted endpoint after encoder sweep 6`.
- V encoders and head output decoders use a full-layer joint solve with cross-head covariance.
- Fit contexts: 256; held-out diagnostic contexts: 64.
- Mean held-out factor-dtype relative MSE: `0.266824514`.

| Layer | Exported boundary | Sweep | Held-out relative MSE |
|---:|:---|---:|---:|
| 0 | after_redecoder | 6 | 0.0418194431 |
| 1 | after_redecoder | 6 | 0.137980168 |
| 2 | after_redecoder | 6 | 0.158701535 |
| 3 | after_redecoder | 6 | 0.153057706 |
| 4 | after_redecoder | 6 | 0.205025137 |
| 5 | after_redecoder | 6 | 0.193494286 |
| 6 | after_redecoder | 6 | 0.261801909 |
| 7 | after_redecoder | 6 | 0.289549673 |
| 8 | after_redecoder | 6 | 0.330987002 |
| 9 | after_redecoder | 6 | 0.366883885 |
| 10 | after_redecoder | 6 | 0.338382828 |
| 11 | after_redecoder | 6 | 0.321875833 |
| 12 | after_redecoder | 6 | 0.237064314 |
| 13 | after_redecoder | 6 | 0.262758075 |
| 14 | after_redecoder | 6 | 0.304458305 |
| 15 | after_redecoder | 6 | 0.288315553 |
| 16 | after_redecoder | 6 | 0.301893049 |
| 17 | after_redecoder | 6 | 0.219937442 |
| 18 | after_redecoder | 6 | 0.266932066 |
| 19 | after_redecoder | 6 | 0.212902401 |
| 20 | after_redecoder | 6 | 0.286556794 |
| 21 | after_redecoder | 6 | 0.306786796 |
| 22 | after_redecoder | 6 | 0.27489791 |
| 23 | after_redecoder | 6 | 0.306743628 |
| 24 | after_redecoder | 6 | 0.240357158 |
| 25 | after_redecoder | 6 | 0.297041923 |
| 26 | after_redecoder | 6 | 0.374125926 |
| 27 | after_redecoder | 6 | 0.33543294 |
| 28 | after_redecoder | 6 | 0.324304032 |
| 29 | after_redecoder | 6 | 0.385043374 |
| 30 | after_redecoder | 6 | 0.27918354 |
| 31 | after_redecoder | 6 | 0.372742263 |
| 32 | after_redecoder | 6 | 0.285571151 |
| 33 | after_redecoder | 6 | 0.366493377 |
| 34 | after_redecoder | 6 | 0.168885496 |
| 35 | after_redecoder | 6 | 0.107695589 |
