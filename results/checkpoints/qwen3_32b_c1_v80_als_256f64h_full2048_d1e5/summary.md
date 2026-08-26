# Qwen3-32B GQA C1 V80 joint fit

- K remains dense; every one of 8 physical V heads retains rank 80/128.
- Total KV-cache retention is 81.25% (18.75% reduction).
- Encoder initialization: `activation-weighted-svd`.
- Held-out selection boundaries: `decoder-closed`.
- V encoders and head output decoders use a full-layer joint solve with cross-head covariance.
- Fit contexts: 256; held-out selection contexts: 64.
- Mean held-out factor-dtype relative MSE: `0.0908279146`.

| Layer | Selected boundary | Sweep | Held-out relative MSE |
|---:|:---|---:|---:|
| 0 | after_redecoder | 5 | 0.00963971301 |
| 1 | after_redecoder | 5 | 0.00777472867 |
| 2 | after_redecoder | 5 | 0.0209229109 |
| 3 | after_redecoder | 5 | 0.0270771971 |
| 4 | after_redecoder | 5 | 0.0355289124 |
| 5 | after_redecoder | 5 | 0.0259205721 |
| 6 | after_redecoder | 5 | 0.0225772754 |
| 7 | after_redecoder | 5 | 0.0838250533 |
| 8 | after_redecoder | 5 | 0.0565702717 |
| 9 | after_redecoder | 5 | 0.0328834271 |
| 10 | after_redecoder | 5 | 0.0468456617 |
| 11 | after_redecoder | 5 | 0.0930622606 |
| 12 | after_redecoder | 5 | 0.0525903936 |
| 13 | after_redecoder | 5 | 0.0748009073 |
| 14 | after_redecoder | 5 | 0.0719472808 |
| 15 | after_redecoder | 5 | 0.0471255947 |
| 16 | after_redecoder | 5 | 0.0405021069 |
| 17 | after_redecoder | 5 | 0.0276815866 |
| 18 | after_redecoder | 5 | 0.0355818349 |
| 19 | after_redecoder | 5 | 0.0478389326 |
| 20 | after_redecoder | 5 | 0.0495812406 |
| 21 | after_redecoder | 5 | 0.066848877 |
| 22 | after_redecoder | 5 | 0.0752027193 |
| 23 | after_redecoder | 5 | 0.0775094592 |
| 24 | after_redecoder | 5 | 0.123507966 |
| 25 | after_redecoder | 5 | 0.0970381623 |
| 26 | after_redecoder | 5 | 0.0953136101 |
| 27 | after_redecoder | 5 | 0.100721744 |
| 28 | after_redecoder | 5 | 0.145592475 |
| 29 | after_redecoder | 5 | 0.108504154 |
| 30 | after_redecoder | 5 | 0.172016247 |
| 31 | after_redecoder | 5 | 0.14380743 |
| 32 | after_redecoder | 5 | 0.0799383326 |
| 33 | after_redecoder | 5 | 0.0660989585 |
| 34 | after_redecoder | 5 | 0.0869510236 |
| 35 | after_redecoder | 5 | 0.133316322 |
| 36 | after_redecoder | 5 | 0.102588664 |
| 37 | after_redecoder | 5 | 0.125618059 |
| 38 | after_redecoder | 5 | 0.126872596 |
| 39 | after_redecoder | 5 | 0.102788473 |
| 40 | after_redecoder | 5 | 0.0944820908 |
| 41 | after_redecoder | 5 | 0.0715326831 |
| 42 | after_redecoder | 5 | 0.1023421 |
| 43 | after_redecoder | 5 | 0.109569137 |
| 44 | after_redecoder | 5 | 0.120984165 |
| 45 | after_redecoder | 5 | 0.107211704 |
| 46 | after_redecoder | 5 | 0.112129466 |
| 47 | after_redecoder | 5 | 0.131096115 |
| 48 | after_redecoder | 5 | 0.106944035 |
| 49 | after_redecoder | 5 | 0.115167642 |
| 50 | after_redecoder | 5 | 0.116002159 |
| 51 | after_redecoder | 5 | 0.100084504 |
| 52 | after_redecoder | 5 | 0.128415451 |
| 53 | after_redecoder | 5 | 0.100697878 |
| 54 | after_redecoder | 5 | 0.155163649 |
| 55 | after_redecoder | 5 | 0.132656873 |
| 56 | after_redecoder | 5 | 0.134251835 |
| 57 | after_redecoder | 5 | 0.157293669 |
| 58 | after_redecoder | 5 | 0.152224385 |
| 59 | after_redecoder | 5 | 0.141242614 |
| 60 | after_redecoder | 5 | 0.163402961 |
| 61 | after_redecoder | 5 | 0.109444783 |
| 62 | after_redecoder | 5 | 0.166401512 |
| 63 | after_redecoder | 5 | 0.0437319919 |
