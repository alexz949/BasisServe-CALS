# Qwen3-32B GQA C1 V80 joint fit

- K remains dense; every one of 8 physical V heads retains rank 80/128.
- Total KV-cache retention is 81.25% (18.75% reduction).
- Encoder initialization: `activation-weighted-svd`.
- Held-out selection boundaries: `decoder-closed`.
- V encoders and head output decoders use a full-layer joint solve with cross-head covariance.
- Fit contexts: 128; held-out selection contexts: 64.
- Mean held-out factor-dtype relative MSE: `0.153144108`.

| Layer | Selected boundary | Sweep | Held-out relative MSE |
|---:|:---|---:|---:|
| 0 | after_redecoder | 5 | 0.0157671075 |
| 1 | after_redecoder | 5 | 0.0132567349 |
| 2 | after_redecoder | 5 | 0.0348924884 |
| 3 | after_redecoder | 5 | 0.0438677284 |
| 4 | after_redecoder | 5 | 0.0620044369 |
| 5 | after_redecoder | 5 | 0.0457912849 |
| 6 | after_redecoder | 5 | 0.0376490297 |
| 7 | after_redecoder | 5 | 0.144718862 |
| 8 | after_redecoder | 5 | 0.101349032 |
| 9 | after_redecoder | 5 | 0.0675789946 |
| 10 | after_redecoder | 5 | 0.0785870959 |
| 11 | after_redecoder | 5 | 0.154699408 |
| 12 | after_redecoder | 5 | 0.0983747048 |
| 13 | after_redecoder | 5 | 0.115405207 |
| 14 | after_redecoder | 5 | 0.132698253 |
| 15 | after_redecoder | 5 | 0.0730745858 |
| 16 | after_redecoder | 5 | 0.076215221 |
| 17 | after_redecoder | 5 | 0.0504850689 |
| 18 | after_redecoder | 5 | 0.0683245364 |
| 19 | after_redecoder | 5 | 0.0985896958 |
| 20 | after_redecoder | 5 | 0.116987233 |
| 21 | after_redecoder | 5 | 0.11122593 |
| 22 | after_redecoder | 5 | 0.121004387 |
| 23 | after_redecoder | 5 | 0.22652086 |
| 24 | after_redecoder | 4 | 0.199237093 |
| 25 | after_redecoder | 5 | 0.170396378 |
| 26 | after_redecoder | 5 | 0.150439429 |
| 27 | after_redecoder | 5 | 0.180384144 |
| 28 | after_redecoder | 5 | 0.281643089 |
| 29 | after_redecoder | 5 | 0.174643482 |
| 30 | after_redecoder | 5 | 0.306966572 |
| 31 | after_redecoder | 5 | 0.236250908 |
| 32 | after_redecoder | 5 | 0.125082861 |
| 33 | after_redecoder | 5 | 0.118588811 |
| 34 | after_redecoder | 5 | 0.141754732 |
| 35 | after_redecoder | 5 | 0.215168116 |
| 36 | after_redecoder | 5 | 0.1581026 |
| 37 | after_redecoder | 5 | 0.187270418 |
| 38 | after_redecoder | 5 | 0.190711855 |
| 39 | after_redecoder | 5 | 0.155290682 |
| 40 | after_redecoder | 5 | 0.147856046 |
| 41 | after_redecoder | 5 | 0.108293119 |
| 42 | after_redecoder | 5 | 0.15842663 |
| 43 | after_redecoder | 5 | 0.16762208 |
| 44 | after_redecoder | 5 | 0.188846659 |
| 45 | after_redecoder | 5 | 0.1615981 |
| 46 | after_redecoder | 5 | 0.170540844 |
| 47 | after_redecoder | 5 | 0.209059544 |
| 48 | after_redecoder | 5 | 0.167388158 |
| 49 | after_redecoder | 5 | 0.172543833 |
| 50 | after_redecoder | 5 | 0.182823614 |
| 51 | after_redecoder | 5 | 0.154672263 |
| 52 | after_redecoder | 5 | 0.200845378 |
| 53 | after_redecoder | 5 | 0.159336071 |
| 54 | after_redecoder | 5 | 0.250723582 |
| 55 | after_redecoder | 5 | 0.204387274 |
| 56 | after_redecoder | 5 | 0.219202379 |
| 57 | after_redecoder | 5 | 0.257996092 |
| 58 | after_redecoder | 5 | 0.286360867 |
| 59 | after_redecoder | 5 | 0.233902406 |
| 60 | after_redecoder | 5 | 0.274034787 |
| 61 | after_redecoder | 5 | 0.175582873 |
| 62 | after_redecoder | 2 | 0.402374554 |
| 63 | after_redecoder | 5 | 0.065806696 |
