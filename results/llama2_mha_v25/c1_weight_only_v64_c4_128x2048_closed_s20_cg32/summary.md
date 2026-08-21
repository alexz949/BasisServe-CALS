# Llama-2-7B MHA C1 V64 joint fit

- K remains dense; every one of 32 V heads retains rank 64/128.
- Total KV-cache retention is 75% (25% reduction).
- Encoder initialization: `weight-only-svd`.
- Held-out selection boundaries: `decoder-closed`.
- V encoders and head output decoders use a full-layer joint solve with cross-head covariance.
- Fit contexts: 128; held-out selection contexts: 64.
- Mean held-out factor-dtype relative MSE: `0.178689454`.

| Layer | Selected boundary | Sweep | Held-out relative MSE |
|---:|:---|---:|---:|
| 0 | after_redecoder | 18 | 0.000476592063 |
| 1 | after_redecoder | 20 | 0.0480016043 |
| 2 | after_redecoder | 20 | 0.109325505 |
| 3 | after_redecoder | 20 | 0.07852609 |
| 4 | after_redecoder | 20 | 0.113882257 |
| 5 | after_redecoder | 13 | 0.115104007 |
| 6 | after_redecoder | 14 | 0.130030321 |
| 7 | after_redecoder | 20 | 0.150380271 |
| 8 | after_redecoder | 20 | 0.162520098 |
| 9 | after_redecoder | 20 | 0.184643645 |
| 10 | after_redecoder | 20 | 0.187310761 |
| 11 | after_redecoder | 16 | 0.195624114 |
| 12 | after_redecoder | 20 | 0.203868826 |
| 13 | after_redecoder | 14 | 0.210011103 |
| 14 | after_redecoder | 20 | 0.221642311 |
| 15 | after_redecoder | 15 | 0.191848777 |
| 16 | after_redecoder | 20 | 0.19421575 |
| 17 | after_redecoder | 20 | 0.232154036 |
| 18 | after_redecoder | 20 | 0.221906065 |
| 19 | after_redecoder | 20 | 0.230388414 |
| 20 | after_redecoder | 13 | 0.19258359 |
| 21 | after_redecoder | 7 | 0.255313806 |
| 22 | after_redecoder | 20 | 0.19027483 |
| 23 | after_redecoder | 9 | 0.259189402 |
| 24 | after_redecoder | 6 | 0.207413501 |
| 25 | after_redecoder | 7 | 0.284251251 |
| 26 | after_redecoder | 9 | 0.188171429 |
| 27 | after_redecoder | 8 | 0.235621819 |
| 28 | after_redecoder | 4 | 0.227907024 |
| 29 | after_redecoder | 7 | 0.2217517 |
| 30 | after_redecoder | 9 | 0.18642605 |
| 31 | after_redecoder | 15 | 0.0872975974 |
