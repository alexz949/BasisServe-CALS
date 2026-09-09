# Qwen3-32B GQA C1 V112 joint fit

- K remains dense; every one of 8 physical V heads retains rank 112/128.
- Total KV-cache retention is 93.75% (6.25% reduction).
- Encoder initialization: `activation-weighted-svd`.
- Checkpoint policy: `fixed decoder-refitted endpoint after encoder sweep 6`.
- V encoders and head output decoders use a full-layer joint solve with cross-head covariance.
- Fit contexts: 256; held-out diagnostic contexts: 64.
- Mean held-out factor-dtype relative MSE: `0.0230567048`.

| Layer | Exported boundary | Sweep | Held-out relative MSE |
|---:|:---|---:|---:|
| 0 | after_redecoder | 6 | 0.00121007298 |
| 1 | after_redecoder | 6 | 0.000831766869 |
| 2 | after_redecoder | 6 | 0.00316221549 |
| 3 | after_redecoder | 6 | 0.00444567848 |
| 4 | after_redecoder | 6 | 0.00694040867 |
| 5 | after_redecoder | 6 | 0.00469177177 |
| 6 | after_redecoder | 6 | 0.00392649509 |
| 7 | after_redecoder | 6 | 0.0183621928 |
| 8 | after_redecoder | 6 | 0.0125179317 |
| 9 | after_redecoder | 6 | 0.0059822034 |
| 10 | after_redecoder | 6 | 0.00906882574 |
| 11 | after_redecoder | 6 | 0.0212948874 |
| 12 | after_redecoder | 6 | 0.0108443994 |
| 13 | after_redecoder | 6 | 0.0159237628 |
| 14 | after_redecoder | 6 | 0.0154848551 |
| 15 | after_redecoder | 6 | 0.00940476791 |
| 16 | after_redecoder | 6 | 0.00769974686 |
| 17 | after_redecoder | 6 | 0.00504638378 |
| 18 | after_redecoder | 6 | 0.00603769154 |
| 19 | after_redecoder | 6 | 0.00916431416 |
| 20 | after_redecoder | 6 | 0.00982711539 |
| 21 | after_redecoder | 6 | 0.0145248771 |
| 22 | after_redecoder | 6 | 0.0166370454 |
| 23 | after_redecoder | 6 | 0.0166361712 |
| 24 | after_redecoder | 6 | 0.0299332126 |
| 25 | after_redecoder | 6 | 0.0226048817 |
| 26 | after_redecoder | 6 | 0.0226530934 |
| 27 | after_redecoder | 6 | 0.0236306159 |
| 28 | after_redecoder | 6 | 0.037248804 |
| 29 | after_redecoder | 6 | 0.0278805838 |
| 30 | after_redecoder | 6 | 0.0455188688 |
| 31 | after_redecoder | 6 | 0.0377552577 |
| 32 | after_redecoder | 6 | 0.0192809143 |
| 33 | after_redecoder | 6 | 0.0154209291 |
| 34 | after_redecoder | 6 | 0.0214050942 |
| 35 | after_redecoder | 6 | 0.0356266378 |
| 36 | after_redecoder | 6 | 0.0268127985 |
| 37 | after_redecoder | 6 | 0.0322342139 |
| 38 | after_redecoder | 6 | 0.034764955 |
| 39 | after_redecoder | 6 | 0.0277539858 |
| 40 | after_redecoder | 6 | 0.0254838262 |
| 41 | after_redecoder | 6 | 0.0191515881 |
| 42 | after_redecoder | 6 | 0.0280279309 |
| 43 | after_redecoder | 6 | 0.030536449 |
| 44 | after_redecoder | 6 | 0.0336184438 |
| 45 | after_redecoder | 6 | 0.0298374864 |
| 46 | after_redecoder | 6 | 0.0305554463 |
| 47 | after_redecoder | 6 | 0.0367085163 |
| 48 | after_redecoder | 6 | 0.0292663665 |
| 49 | after_redecoder | 6 | 0.031239729 |
| 50 | after_redecoder | 6 | 0.0318749914 |
| 51 | after_redecoder | 6 | 0.0261354898 |
| 52 | after_redecoder | 6 | 0.0337527097 |
| 53 | after_redecoder | 6 | 0.0269232749 |
| 54 | after_redecoder | 6 | 0.0403566387 |
| 55 | after_redecoder | 6 | 0.0351389631 |
| 56 | after_redecoder | 6 | 0.0367789642 |
| 57 | after_redecoder | 6 | 0.0442838712 |
| 58 | after_redecoder | 6 | 0.0416237322 |
| 59 | after_redecoder | 6 | 0.0377837196 |
| 60 | after_redecoder | 6 | 0.0484406635 |
| 61 | after_redecoder | 6 | 0.0282506517 |
| 62 | after_redecoder | 6 | 0.0479151925 |
| 63 | after_redecoder | 6 | 0.0117540325 |
