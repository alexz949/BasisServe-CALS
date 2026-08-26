# Qwen3-32B GQA C1 V80 joint fit

- K remains dense; every one of 8 physical V heads retains rank 80/128.
- Total KV-cache retention is 81.25% (18.75% reduction).
- Encoder initialization: `activation-weighted-svd`.
- Held-out selection boundaries: `decoder-closed`.
- V encoders and head output decoders use a full-layer joint solve with cross-head covariance.
- Fit contexts: 256; held-out selection contexts: 64.
- Mean held-out factor-dtype relative MSE: `0.0943515463`.

| Layer | Selected boundary | Sweep | Held-out relative MSE |
|---:|:---|---:|---:|
| 0 | decoder_only | 0 | 0.012487575 |
| 1 | decoder_only | 0 | 0.00874058562 |
| 2 | decoder_only | 0 | 0.0222841472 |
| 3 | decoder_only | 0 | 0.0287280126 |
| 4 | decoder_only | 0 | 0.0381112304 |
| 5 | decoder_only | 0 | 0.0270620919 |
| 6 | decoder_only | 0 | 0.0237442647 |
| 7 | decoder_only | 0 | 0.0876500516 |
| 8 | decoder_only | 0 | 0.0597312781 |
| 9 | decoder_only | 0 | 0.0338555963 |
| 10 | decoder_only | 0 | 0.0483386765 |
| 11 | decoder_only | 0 | 0.0977635953 |
| 12 | decoder_only | 0 | 0.0538585544 |
| 13 | decoder_only | 0 | 0.0771018315 |
| 14 | decoder_only | 0 | 0.0739679469 |
| 15 | decoder_only | 0 | 0.049263075 |
| 16 | decoder_only | 0 | 0.0419710907 |
| 17 | decoder_only | 0 | 0.0289061232 |
| 18 | decoder_only | 0 | 0.0366710229 |
| 19 | decoder_only | 0 | 0.0489622006 |
| 20 | decoder_only | 0 | 0.0506060101 |
| 21 | decoder_only | 0 | 0.0693133657 |
| 22 | decoder_only | 0 | 0.0774801672 |
| 23 | decoder_only | 0 | 0.0803456317 |
| 24 | decoder_only | 0 | 0.126318903 |
| 25 | decoder_only | 0 | 0.100094068 |
| 26 | decoder_only | 0 | 0.0992181085 |
| 27 | decoder_only | 0 | 0.103768495 |
| 28 | decoder_only | 0 | 0.149321033 |
| 29 | decoder_only | 0 | 0.111335557 |
| 30 | decoder_only | 0 | 0.176271567 |
| 31 | decoder_only | 0 | 0.148913702 |
| 32 | decoder_only | 0 | 0.0833518846 |
| 33 | decoder_only | 0 | 0.0688214808 |
| 34 | decoder_only | 0 | 0.0899164063 |
| 35 | decoder_only | 0 | 0.137194861 |
| 36 | decoder_only | 0 | 0.105880046 |
| 37 | decoder_only | 0 | 0.129902649 |
| 38 | decoder_only | 0 | 0.131354501 |
| 39 | decoder_only | 0 | 0.108161031 |
| 40 | decoder_only | 0 | 0.0999874831 |
| 41 | decoder_only | 0 | 0.075285204 |
| 42 | decoder_only | 0 | 0.107005393 |
| 43 | decoder_only | 0 | 0.114576024 |
| 44 | decoder_only | 0 | 0.12561576 |
| 45 | decoder_only | 0 | 0.112233898 |
| 46 | decoder_only | 0 | 0.117072623 |
| 47 | decoder_only | 0 | 0.136324984 |
| 48 | decoder_only | 0 | 0.111157161 |
| 49 | decoder_only | 0 | 0.119996826 |
| 50 | decoder_only | 0 | 0.121163152 |
| 51 | decoder_only | 0 | 0.105278846 |
| 52 | decoder_only | 0 | 0.133208138 |
| 53 | decoder_only | 0 | 0.104492206 |
| 54 | decoder_only | 0 | 0.161436327 |
| 55 | decoder_only | 0 | 0.140040705 |
| 56 | decoder_only | 0 | 0.140236734 |
| 57 | decoder_only | 0 | 0.162908427 |
| 58 | decoder_only | 0 | 0.156002477 |
| 59 | decoder_only | 0 | 0.146775024 |
| 60 | decoder_only | 0 | 0.167390513 |
| 61 | decoder_only | 0 | 0.112348107 |
| 62 | decoder_only | 0 | 0.174705378 |
| 63 | decoder_only | 0 | 0.046489154 |
