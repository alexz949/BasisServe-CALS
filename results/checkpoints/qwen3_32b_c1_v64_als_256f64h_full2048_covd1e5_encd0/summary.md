# Qwen3-32B GQA C1 V64 joint fit

- K remains dense; every one of 8 physical V heads retains rank 64/128.
- Total KV-cache retention is 75% (25% reduction).
- Encoder initialization: `activation-weighted-svd`.
- Held-out selection boundaries: `decoder-closed`.
- V encoders and head output decoders use a full-layer joint solve with cross-head covariance.
- Fit contexts: 256; held-out selection contexts: 64.
- Mean held-out factor-dtype relative MSE: `0.137814583`.

| Layer | Selected boundary | Sweep | Held-out relative MSE |
|---:|:---|---:|---:|
| 0 | after_redecoder | 5 | 0.0204963949 |
| 1 | after_redecoder | 5 | 0.0164906971 |
| 2 | after_redecoder | 5 | 0.0394151911 |
| 3 | after_redecoder | 5 | 0.0482566749 |
| 4 | after_redecoder | 5 | 0.0600397505 |
| 5 | after_redecoder | 5 | 0.0446270241 |
| 6 | after_redecoder | 5 | 0.0397952218 |
| 7 | after_redecoder | 5 | 0.134915054 |
| 8 | after_redecoder | 5 | 0.0907792647 |
| 9 | after_redecoder | 5 | 0.0584190978 |
| 10 | after_redecoder | 5 | 0.0798464828 |
| 11 | after_redecoder | 5 | 0.147498211 |
| 12 | after_redecoder | 5 | 0.0875566942 |
| 13 | after_redecoder | 5 | 0.121169172 |
| 14 | after_redecoder | 5 | 0.116848353 |
| 15 | after_redecoder | 5 | 0.0798093358 |
| 16 | after_redecoder | 5 | 0.0701401034 |
| 17 | after_redecoder | 5 | 0.0490437284 |
| 18 | after_redecoder | 5 | 0.0638728505 |
| 19 | after_redecoder | 5 | 0.0823996108 |
| 20 | after_redecoder | 5 | 0.0848893181 |
| 21 | after_redecoder | 5 | 0.109636097 |
| 22 | after_redecoder | 5 | 0.120803074 |
| 23 | after_redecoder | 5 | 0.125859038 |
| 24 | after_redecoder | 5 | 0.189466769 |
| 25 | after_redecoder | 5 | 0.151606977 |
| 26 | after_redecoder | 5 | 0.149184959 |
| 27 | after_redecoder | 5 | 0.156654934 |
| 28 | after_redecoder | 5 | 0.218512735 |
| 29 | after_redecoder | 5 | 0.1637569 |
| 30 | after_redecoder | 5 | 0.255399067 |
| 31 | after_redecoder | 5 | 0.213278022 |
| 32 | after_redecoder | 5 | 0.122512883 |
| 33 | after_redecoder | 5 | 0.103608531 |
| 34 | after_redecoder | 5 | 0.133740453 |
| 35 | after_redecoder | 5 | 0.19822612 |
| 36 | after_redecoder | 5 | 0.153662146 |
| 37 | after_redecoder | 5 | 0.187273372 |
| 38 | after_redecoder | 5 | 0.185073151 |
| 39 | after_redecoder | 5 | 0.15211844 |
| 40 | after_redecoder | 5 | 0.139924557 |
| 41 | after_redecoder | 5 | 0.106755236 |
| 42 | after_redecoder | 5 | 0.150965049 |
| 43 | after_redecoder | 5 | 0.160344641 |
| 44 | after_redecoder | 5 | 0.175848355 |
| 45 | after_redecoder | 5 | 0.156101929 |
| 46 | after_redecoder | 5 | 0.164597807 |
| 47 | after_redecoder | 5 | 0.191091786 |
| 48 | after_redecoder | 5 | 0.156445377 |
| 49 | after_redecoder | 5 | 0.168932951 |
| 50 | after_redecoder | 5 | 0.170901262 |
| 51 | after_redecoder | 5 | 0.151397277 |
| 52 | after_redecoder | 5 | 0.192207077 |
| 53 | after_redecoder | 5 | 0.148777061 |
| 54 | after_redecoder | 5 | 0.231226589 |
| 55 | after_redecoder | 5 | 0.194632523 |
| 56 | after_redecoder | 5 | 0.194452784 |
| 57 | after_redecoder | 5 | 0.227082258 |
| 58 | after_redecoder | 5 | 0.221420755 |
| 59 | after_redecoder | 5 | 0.206955607 |
| 60 | after_redecoder | 5 | 0.22680701 |
| 61 | after_redecoder | 5 | 0.159346693 |
| 62 | after_redecoder | 5 | 0.234419669 |
| 63 | after_redecoder | 5 | 0.0628171633 |
