# Qwen3-32B GQA C1 V128 joint fit

- K remains dense; every one of 8 physical V heads retains rank 128/128.
- Total KV-cache retention is 100% (0% reduction).
- Encoder initialization: `activation-weighted-svd`.
- Held-out selection boundaries: `decoder-closed`.
- V encoders and head output decoders use a full-layer joint solve with cross-head covariance.
- Fit contexts: 128; held-out selection contexts: 64.
- Mean held-out factor-dtype relative MSE: `0`.

| Layer | Selected boundary | Sweep | Held-out relative MSE |
|---:|:---|---:|---:|
| 0 | decoder_only | 0 | 0 |
| 1 | decoder_only | 0 | 0 |
| 2 | decoder_only | 0 | 0 |
| 3 | decoder_only | 0 | 0 |
| 4 | decoder_only | 0 | 0 |
| 5 | decoder_only | 0 | 0 |
| 6 | decoder_only | 0 | 0 |
| 7 | decoder_only | 0 | 0 |
| 8 | decoder_only | 0 | 0 |
| 9 | decoder_only | 0 | 0 |
| 10 | decoder_only | 0 | 0 |
| 11 | decoder_only | 0 | 0 |
| 12 | decoder_only | 0 | 0 |
| 13 | decoder_only | 0 | 0 |
| 14 | decoder_only | 0 | 0 |
| 15 | decoder_only | 0 | 0 |
| 16 | decoder_only | 0 | 0 |
| 17 | decoder_only | 0 | 0 |
| 18 | decoder_only | 0 | 0 |
| 19 | decoder_only | 0 | 0 |
| 20 | decoder_only | 0 | 0 |
| 21 | decoder_only | 0 | 0 |
| 22 | decoder_only | 0 | 0 |
| 23 | decoder_only | 0 | 0 |
| 24 | decoder_only | 0 | 0 |
| 25 | decoder_only | 0 | 0 |
| 26 | decoder_only | 0 | 0 |
| 27 | decoder_only | 0 | 0 |
| 28 | decoder_only | 0 | 0 |
| 29 | decoder_only | 0 | 0 |
| 30 | decoder_only | 0 | 0 |
| 31 | decoder_only | 0 | 0 |
| 32 | decoder_only | 0 | 0 |
| 33 | decoder_only | 0 | 0 |
| 34 | decoder_only | 0 | 0 |
| 35 | decoder_only | 0 | 0 |
| 36 | decoder_only | 0 | 0 |
| 37 | decoder_only | 0 | 0 |
| 38 | decoder_only | 0 | 0 |
| 39 | decoder_only | 0 | 0 |
| 40 | decoder_only | 0 | 0 |
| 41 | decoder_only | 0 | 0 |
| 42 | decoder_only | 0 | 0 |
| 43 | decoder_only | 0 | 0 |
| 44 | decoder_only | 0 | 0 |
| 45 | decoder_only | 0 | 0 |
| 46 | decoder_only | 0 | 0 |
| 47 | decoder_only | 0 | 0 |
| 48 | decoder_only | 0 | 0 |
| 49 | decoder_only | 0 | 0 |
| 50 | decoder_only | 0 | 0 |
| 51 | decoder_only | 0 | 0 |
| 52 | decoder_only | 0 | 0 |
| 53 | decoder_only | 0 | 0 |
| 54 | decoder_only | 0 | 0 |
| 55 | decoder_only | 0 | 0 |
| 56 | decoder_only | 0 | 0 |
| 57 | decoder_only | 0 | 0 |
| 58 | decoder_only | 0 | 0 |
| 59 | decoder_only | 0 | 0 |
| 60 | decoder_only | 0 | 0 |
| 61 | decoder_only | 0 | 0 |
| 62 | decoder_only | 0 | 0 |
| 63 | decoder_only | 0 | 0 |
