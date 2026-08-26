# Qwen3-32B GQA C1 V64 joint fit

- K remains dense; every one of 8 physical V heads retains rank 64/128.
- Total KV-cache retention is 75% (25% reduction).
- Encoder initialization: `activation-weighted-svd`.
- Held-out selection boundaries: `decoder-closed`.
- V encoders and head output decoders use a full-layer joint solve with cross-head covariance.
- Fit contexts: 256; held-out selection contexts: 64.
- Mean held-out factor-dtype relative MSE: `0.137814457`.

| Layer | Selected boundary | Sweep | Held-out relative MSE |
|---:|:---|---:|---:|
| 0 | after_redecoder | 5 | 0.0204955065 |
| 1 | after_redecoder | 5 | 0.0164904702 |
| 2 | after_redecoder | 5 | 0.0394151911 |
| 3 | after_redecoder | 5 | 0.0482561516 |
| 4 | after_redecoder | 5 | 0.0600397395 |
| 5 | after_redecoder | 5 | 0.0446282147 |
| 6 | after_redecoder | 5 | 0.0397950762 |
| 7 | after_redecoder | 5 | 0.13491432 |
| 8 | after_redecoder | 5 | 0.0907788566 |
| 9 | after_redecoder | 5 | 0.0584190978 |
| 10 | after_redecoder | 5 | 0.0798470961 |
| 11 | after_redecoder | 5 | 0.147498463 |
| 12 | after_redecoder | 5 | 0.0875572496 |
| 13 | after_redecoder | 5 | 0.121169676 |
| 14 | after_redecoder | 5 | 0.116848487 |
| 15 | after_redecoder | 5 | 0.0798089205 |
| 16 | after_redecoder | 5 | 0.0701400965 |
| 17 | after_redecoder | 5 | 0.0490443264 |
| 18 | after_redecoder | 5 | 0.0638729565 |
| 19 | after_redecoder | 5 | 0.0823994223 |
| 20 | after_redecoder | 5 | 0.0848893181 |
| 21 | after_redecoder | 5 | 0.109636313 |
| 22 | after_redecoder | 5 | 0.120802822 |
| 23 | after_redecoder | 5 | 0.125858169 |
| 24 | after_redecoder | 5 | 0.189466605 |
| 25 | after_redecoder | 5 | 0.15160689 |
| 26 | after_redecoder | 5 | 0.149183982 |
| 27 | after_redecoder | 5 | 0.156655491 |
| 28 | after_redecoder | 5 | 0.218512814 |
| 29 | after_redecoder | 5 | 0.163757109 |
| 30 | after_redecoder | 5 | 0.255399281 |
| 31 | after_redecoder | 5 | 0.213278083 |
| 32 | after_redecoder | 5 | 0.122512741 |
| 33 | after_redecoder | 5 | 0.103608521 |
| 34 | after_redecoder | 5 | 0.13374129 |
| 35 | after_redecoder | 5 | 0.198226266 |
| 36 | after_redecoder | 5 | 0.153662208 |
| 37 | after_redecoder | 5 | 0.187273277 |
| 38 | after_redecoder | 5 | 0.185072943 |
| 39 | after_redecoder | 5 | 0.152117682 |
| 40 | after_redecoder | 5 | 0.139924295 |
| 41 | after_redecoder | 5 | 0.106755393 |
| 42 | after_redecoder | 5 | 0.150964304 |
| 43 | after_redecoder | 5 | 0.160344922 |
| 44 | after_redecoder | 5 | 0.175848612 |
| 45 | after_redecoder | 5 | 0.156102787 |
| 46 | after_redecoder | 5 | 0.16459722 |
| 47 | after_redecoder | 5 | 0.19109193 |
| 48 | after_redecoder | 5 | 0.156444736 |
| 49 | after_redecoder | 5 | 0.168933604 |
| 50 | after_redecoder | 5 | 0.17090186 |
| 51 | after_redecoder | 5 | 0.151397893 |
| 52 | after_redecoder | 5 | 0.19220706 |
| 53 | after_redecoder | 5 | 0.148777378 |
| 54 | after_redecoder | 5 | 0.231225941 |
| 55 | after_redecoder | 5 | 0.194632416 |
| 56 | after_redecoder | 5 | 0.194452458 |
| 57 | after_redecoder | 5 | 0.227081522 |
| 58 | after_redecoder | 5 | 0.221420474 |
| 59 | after_redecoder | 5 | 0.206956177 |
| 60 | after_redecoder | 5 | 0.226805758 |
| 61 | after_redecoder | 5 | 0.159345442 |
| 62 | after_redecoder | 5 | 0.23441653 |
| 63 | after_redecoder | 5 | 0.0628154402 |
