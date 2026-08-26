# Qwen3-8B-Base GQA C1 V64 joint fit

- K remains dense; every one of 8 physical V heads retains rank 64/128.
- Total KV-cache retention is 75% (25% reduction).
- Encoder initialization: `activation-weighted-svd`.
- Held-out selection boundaries: `decoder-closed`.
- V encoders and head output decoders use a full-layer joint solve with cross-head covariance.
- Fit contexts: 256; held-out selection contexts: 64.
- Mean held-out factor-dtype relative MSE: `0.137045686`.

| Layer | Selected boundary | Sweep | Held-out relative MSE |
|---:|:---|---:|---:|
| 0 | after_redecoder | 5 | 0.0202174723 |
| 1 | after_redecoder | 5 | 0.0539904254 |
| 2 | after_redecoder | 5 | 0.0692472503 |
| 3 | after_redecoder | 5 | 0.06740099 |
| 4 | after_redecoder | 5 | 0.0938695665 |
| 5 | after_redecoder | 5 | 0.086994147 |
| 6 | after_redecoder | 5 | 0.121037054 |
| 7 | after_redecoder | 5 | 0.14081579 |
| 8 | after_redecoder | 5 | 0.169349821 |
| 9 | after_redecoder | 5 | 0.19388865 |
| 10 | after_redecoder | 5 | 0.17892811 |
| 11 | after_redecoder | 5 | 0.169043877 |
| 12 | after_redecoder | 5 | 0.119021969 |
| 13 | after_redecoder | 5 | 0.134792086 |
| 14 | after_redecoder | 5 | 0.158944821 |
| 15 | after_redecoder | 5 | 0.150266281 |
| 16 | after_redecoder | 5 | 0.161603134 |
| 17 | after_redecoder | 5 | 0.114151545 |
| 18 | after_redecoder | 5 | 0.141989271 |
| 19 | after_redecoder | 5 | 0.101504774 |
| 20 | after_redecoder | 5 | 0.152807163 |
| 21 | after_redecoder | 5 | 0.161885009 |
| 22 | after_redecoder | 5 | 0.143898565 |
| 23 | after_redecoder | 5 | 0.155084903 |
| 24 | after_redecoder | 5 | 0.108442142 |
| 25 | after_redecoder | 5 | 0.153418449 |
| 26 | after_redecoder | 5 | 0.201277711 |
| 27 | after_redecoder | 5 | 0.179036086 |
| 28 | after_redecoder | 5 | 0.170109408 |
| 29 | after_redecoder | 5 | 0.203108313 |
| 30 | after_redecoder | 5 | 0.147307745 |
| 31 | after_redecoder | 5 | 0.200536916 |
| 32 | after_redecoder | 5 | 0.151882395 |
| 33 | after_redecoder | 5 | 0.201804948 |
| 34 | after_redecoder | 5 | 0.0935312134 |
| 35 | after_redecoder | 5 | 0.0624566924 |
