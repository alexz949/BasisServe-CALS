# Qwen3-32B GQA C1 V32 joint fit

- K remains dense; every one of 8 physical V heads retains rank 32/128.
- Total KV-cache retention is 62.5% (37.5% reduction).
- Encoder initialization: `activation-weighted-svd`.
- Held-out selection boundaries: `decoder-closed`.
- V encoders and head output decoders use a full-layer joint solve with cross-head covariance.
- Fit contexts: 256; held-out selection contexts: 64.
- Mean held-out factor-dtype relative MSE: `0.281189735`.

| Layer | Selected boundary | Sweep | Held-out relative MSE |
|---:|:---|---:|---:|
| 0 | after_redecoder | 5 | 0.0621546412 |
| 1 | after_redecoder | 5 | 0.0624990216 |
| 2 | after_redecoder | 5 | 0.124311447 |
| 3 | after_redecoder | 5 | 0.133972023 |
| 4 | after_redecoder | 5 | 0.147747395 |
| 5 | after_redecoder | 5 | 0.115427535 |
| 6 | after_redecoder | 5 | 0.111701273 |
| 7 | after_redecoder | 5 | 0.293992983 |
| 8 | after_redecoder | 5 | 0.206419493 |
| 9 | after_redecoder | 5 | 0.166081548 |
| 10 | after_redecoder | 5 | 0.209884595 |
| 11 | after_redecoder | 5 | 0.326573435 |
| 12 | after_redecoder | 5 | 0.217277258 |
| 13 | after_redecoder | 5 | 0.275601124 |
| 14 | after_redecoder | 5 | 0.270173872 |
| 15 | after_redecoder | 5 | 0.204049833 |
| 16 | after_redecoder | 5 | 0.187373765 |
| 17 | after_redecoder | 5 | 0.138321952 |
| 18 | after_redecoder | 5 | 0.183232822 |
| 19 | after_redecoder | 5 | 0.217205516 |
| 20 | after_redecoder | 5 | 0.223862776 |
| 21 | after_redecoder | 5 | 0.26096394 |
| 22 | after_redecoder | 5 | 0.274579765 |
| 23 | after_redecoder | 5 | 0.292650289 |
| 24 | after_redecoder | 5 | 0.3863115 |
| 25 | after_redecoder | 5 | 0.323305418 |
| 26 | after_redecoder | 5 | 0.320823601 |
| 27 | after_redecoder | 5 | 0.331252098 |
| 28 | after_redecoder | 5 | 0.429813871 |
| 29 | after_redecoder | 5 | 0.325887165 |
| 30 | after_redecoder | 5 | 0.479754209 |
| 31 | after_redecoder | 5 | 0.402283647 |
| 32 | after_redecoder | 5 | 0.251977358 |
| 33 | after_redecoder | 5 | 0.22444636 |
| 34 | after_redecoder | 5 | 0.279686296 |
| 35 | after_redecoder | 5 | 0.385691132 |
| 36 | after_redecoder | 5 | 0.302231662 |
| 37 | after_redecoder | 5 | 0.35690232 |
| 38 | after_redecoder | 5 | 0.34237056 |
| 39 | after_redecoder | 5 | 0.29348251 |
| 40 | after_redecoder | 5 | 0.272795707 |
| 41 | after_redecoder | 5 | 0.216512152 |
| 42 | after_redecoder | 5 | 0.292903587 |
| 43 | after_redecoder | 5 | 0.307924463 |
| 44 | after_redecoder | 5 | 0.326858548 |
| 45 | after_redecoder | 5 | 0.290662141 |
| 46 | after_redecoder | 5 | 0.312060074 |
| 47 | after_redecoder | 5 | 0.359467619 |
| 48 | after_redecoder | 5 | 0.291242337 |
| 49 | after_redecoder | 5 | 0.319809321 |
| 50 | after_redecoder | 5 | 0.32712814 |
| 51 | after_redecoder | 5 | 0.314437469 |
| 52 | after_redecoder | 5 | 0.38955039 |
| 53 | after_redecoder | 5 | 0.295007932 |
| 54 | after_redecoder | 5 | 0.440486789 |
| 55 | after_redecoder | 5 | 0.369763045 |
| 56 | after_redecoder | 5 | 0.35842612 |
| 57 | after_redecoder | 5 | 0.410496083 |
| 58 | after_redecoder | 5 | 0.397572284 |
| 59 | after_redecoder | 5 | 0.382472056 |
| 60 | after_redecoder | 5 | 0.379385585 |
| 61 | after_redecoder | 5 | 0.296014334 |
| 62 | after_redecoder | 5 | 0.390299131 |
| 63 | after_redecoder | 5 | 0.112589717 |
