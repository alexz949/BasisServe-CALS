# Qwen3-8B-Base GQA C1 V80 joint fit

- K remains dense; every one of 8 physical V heads retains rank 80/128.
- Total KV-cache retention is 81.25% (18.75% reduction).
- Encoder initialization: `activation-weighted-svd`.
- Held-out selection boundaries: `decoder-closed`.
- V encoders and head output decoders use a full-layer joint solve with cross-head covariance.
- Fit contexts: 256; held-out selection contexts: 64.
- Mean held-out factor-dtype relative MSE: `0.0923011825`.

| Layer | Selected boundary | Sweep | Held-out relative MSE |
|---:|:---|---:|---:|
| 0 | after_redecoder | 5 | 0.0128070488 |
| 1 | after_redecoder | 5 | 0.03198267 |
| 2 | after_redecoder | 5 | 0.0434639825 |
| 3 | after_redecoder | 5 | 0.0422757179 |
| 4 | after_redecoder | 5 | 0.0598542826 |
| 5 | after_redecoder | 5 | 0.0551420716 |
| 6 | after_redecoder | 5 | 0.0770085252 |
| 7 | after_redecoder | 5 | 0.0925648953 |
| 8 | after_redecoder | 5 | 0.113623856 |
| 9 | after_redecoder | 5 | 0.130918988 |
| 10 | after_redecoder | 5 | 0.121306361 |
| 11 | after_redecoder | 5 | 0.115013115 |
| 12 | after_redecoder | 5 | 0.0799546303 |
| 13 | after_redecoder | 5 | 0.0909816147 |
| 14 | after_redecoder | 5 | 0.108456586 |
| 15 | after_redecoder | 5 | 0.102353735 |
| 16 | after_redecoder | 5 | 0.110876368 |
| 17 | after_redecoder | 5 | 0.0778859002 |
| 18 | after_redecoder | 5 | 0.097516554 |
| 19 | after_redecoder | 5 | 0.0676443561 |
| 20 | after_redecoder | 5 | 0.10488579 |
| 21 | after_redecoder | 5 | 0.110501002 |
| 22 | after_redecoder | 5 | 0.0978940072 |
| 23 | after_redecoder | 5 | 0.104100335 |
| 24 | after_redecoder | 5 | 0.0692895198 |
| 25 | after_redecoder | 5 | 0.103951292 |
| 26 | after_redecoder | 5 | 0.137982826 |
| 27 | after_redecoder | 5 | 0.122738228 |
| 28 | after_redecoder | 5 | 0.116675418 |
| 29 | after_redecoder | 5 | 0.138063769 |
| 30 | after_redecoder | 5 | 0.100618295 |
| 31 | after_redecoder | 5 | 0.135881932 |
| 32 | after_redecoder | 5 | 0.102503775 |
| 33 | after_redecoder | 5 | 0.137487792 |
| 34 | after_redecoder | 5 | 0.0647335407 |
| 35 | after_redecoder | 5 | 0.0439037903 |
