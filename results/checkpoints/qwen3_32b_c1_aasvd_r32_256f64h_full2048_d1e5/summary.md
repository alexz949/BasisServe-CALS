# Qwen3-32B GQA C1 V32 joint fit

- K remains dense; every one of 8 physical V heads retains rank 32/128.
- Total KV-cache retention is 62.5% (37.5% reduction).
- Encoder initialization: `activation-weighted-svd`.
- Held-out selection boundaries: `decoder-closed`.
- V encoders and head output decoders use a full-layer joint solve with cross-head covariance.
- Fit contexts: 256; held-out selection contexts: 64.
- Mean held-out factor-dtype relative MSE: `0.289529875`.

| Layer | Selected boundary | Sweep | Held-out relative MSE |
|---:|:---|---:|---:|
| 0 | decoder_only | 0 | 0.0692798544 |
| 1 | decoder_only | 0 | 0.0706008642 |
| 2 | decoder_only | 0 | 0.132409679 |
| 3 | decoder_only | 0 | 0.141002356 |
| 4 | decoder_only | 0 | 0.154699037 |
| 5 | decoder_only | 0 | 0.120293899 |
| 6 | decoder_only | 0 | 0.118739803 |
| 7 | decoder_only | 0 | 0.304026703 |
| 8 | decoder_only | 0 | 0.216366327 |
| 9 | decoder_only | 0 | 0.172488047 |
| 10 | decoder_only | 0 | 0.216151756 |
| 11 | decoder_only | 0 | 0.341489153 |
| 12 | decoder_only | 0 | 0.223193799 |
| 13 | decoder_only | 0 | 0.282823729 |
| 14 | decoder_only | 0 | 0.275948256 |
| 15 | decoder_only | 0 | 0.212959531 |
| 16 | decoder_only | 0 | 0.195220557 |
| 17 | decoder_only | 0 | 0.144210587 |
| 18 | decoder_only | 0 | 0.187265668 |
| 19 | decoder_only | 0 | 0.221980851 |
| 20 | decoder_only | 0 | 0.227708437 |
| 21 | decoder_only | 0 | 0.269953147 |
| 22 | decoder_only | 0 | 0.282286614 |
| 23 | decoder_only | 0 | 0.300160184 |
| 24 | decoder_only | 0 | 0.391036947 |
| 25 | decoder_only | 0 | 0.330664708 |
| 26 | decoder_only | 0 | 0.330720241 |
| 27 | decoder_only | 0 | 0.337737338 |
| 28 | decoder_only | 0 | 0.437618751 |
| 29 | decoder_only | 0 | 0.332372524 |
| 30 | decoder_only | 0 | 0.486282841 |
| 31 | decoder_only | 0 | 0.409903734 |
| 32 | decoder_only | 0 | 0.262144684 |
| 33 | decoder_only | 0 | 0.231321459 |
| 34 | decoder_only | 0 | 0.286610237 |
| 35 | decoder_only | 0 | 0.393853513 |
| 36 | decoder_only | 0 | 0.308924524 |
| 37 | decoder_only | 0 | 0.364974989 |
| 38 | decoder_only | 0 | 0.349986677 |
| 39 | decoder_only | 0 | 0.304514327 |
| 40 | decoder_only | 0 | 0.283531134 |
| 41 | decoder_only | 0 | 0.227525112 |
| 42 | decoder_only | 0 | 0.302302696 |
| 43 | decoder_only | 0 | 0.32637965 |
| 44 | decoder_only | 0 | 0.334419452 |
| 45 | decoder_only | 0 | 0.299498604 |
| 46 | decoder_only | 0 | 0.320984832 |
| 47 | decoder_only | 0 | 0.369883505 |
| 48 | decoder_only | 0 | 0.298141669 |
| 49 | decoder_only | 0 | 0.328729441 |
| 50 | decoder_only | 0 | 0.336506545 |
| 51 | decoder_only | 0 | 0.333272688 |
| 52 | decoder_only | 0 | 0.401277023 |
| 53 | decoder_only | 0 | 0.303223215 |
| 54 | decoder_only | 0 | 0.451332674 |
| 55 | decoder_only | 0 | 0.384252482 |
| 56 | decoder_only | 0 | 0.367734032 |
| 57 | decoder_only | 0 | 0.420464685 |
| 58 | decoder_only | 0 | 0.403103522 |
| 59 | decoder_only | 0 | 0.391765913 |
| 60 | decoder_only | 0 | 0.385716899 |
| 61 | decoder_only | 0 | 0.30070544 |
| 62 | decoder_only | 0 | 0.401211096 |
| 63 | decoder_only | 0 | 0.118023361 |
