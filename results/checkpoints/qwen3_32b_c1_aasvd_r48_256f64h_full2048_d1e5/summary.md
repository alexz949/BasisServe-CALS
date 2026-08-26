# Qwen3-32B GQA C1 V48 joint fit

- K remains dense; every one of 8 physical V heads retains rank 48/128.
- Total KV-cache retention is 68.75% (31.25% reduction).
- Encoder initialization: `activation-weighted-svd`.
- Held-out selection boundaries: `decoder-closed`.
- V encoders and head output decoders use a full-layer joint solve with cross-head covariance.
- Fit contexts: 256; held-out selection contexts: 64.
- Mean held-out factor-dtype relative MSE: `0.204985999`.

| Layer | Selected boundary | Sweep | Held-out relative MSE |
|---:|:---|---:|---:|
| 0 | decoder_only | 0 | 0.0434841357 |
| 1 | decoder_only | 0 | 0.0373751697 |
| 2 | decoder_only | 0 | 0.0753265323 |
| 3 | decoder_only | 0 | 0.0853498614 |
| 4 | decoder_only | 0 | 0.100767252 |
| 5 | decoder_only | 0 | 0.0755451793 |
| 6 | decoder_only | 0 | 0.0710820329 |
| 7 | decoder_only | 0 | 0.208799605 |
| 8 | decoder_only | 0 | 0.144524047 |
| 9 | decoder_only | 0 | 0.102487117 |
| 10 | decoder_only | 0 | 0.133417829 |
| 11 | decoder_only | 0 | 0.232315147 |
| 12 | decoder_only | 0 | 0.142541587 |
| 13 | decoder_only | 0 | 0.1899183 |
| 14 | decoder_only | 0 | 0.183424754 |
| 15 | decoder_only | 0 | 0.133987183 |
| 16 | decoder_only | 0 | 0.11981796 |
| 17 | decoder_only | 0 | 0.0861655449 |
| 18 | decoder_only | 0 | 0.111084241 |
| 19 | decoder_only | 0 | 0.138169236 |
| 20 | decoder_only | 0 | 0.140871153 |
| 21 | decoder_only | 0 | 0.176883529 |
| 22 | decoder_only | 0 | 0.188723579 |
| 23 | decoder_only | 0 | 0.198763836 |
| 24 | decoder_only | 0 | 0.27810305 |
| 25 | decoder_only | 0 | 0.229004516 |
| 26 | decoder_only | 0 | 0.228975171 |
| 27 | decoder_only | 0 | 0.234833668 |
| 28 | decoder_only | 0 | 0.316974187 |
| 29 | decoder_only | 0 | 0.238393568 |
| 30 | decoder_only | 0 | 0.361963939 |
| 31 | decoder_only | 0 | 0.304857753 |
| 32 | decoder_only | 0 | 0.184796448 |
| 33 | decoder_only | 0 | 0.159417996 |
| 34 | decoder_only | 0 | 0.200558782 |
| 35 | decoder_only | 0 | 0.285742225 |
| 36 | decoder_only | 0 | 0.222531337 |
| 37 | decoder_only | 0 | 0.268596412 |
| 38 | decoder_only | 0 | 0.261905101 |
| 39 | decoder_only | 0 | 0.222457678 |
| 40 | decoder_only | 0 | 0.205392296 |
| 41 | decoder_only | 0 | 0.159646804 |
| 42 | decoder_only | 0 | 0.219168271 |
| 43 | decoder_only | 0 | 0.235880906 |
| 44 | decoder_only | 0 | 0.249140506 |
| 45 | decoder_only | 0 | 0.222827452 |
| 46 | decoder_only | 0 | 0.236041163 |
| 47 | decoder_only | 0 | 0.273153666 |
| 48 | decoder_only | 0 | 0.222056343 |
| 49 | decoder_only | 0 | 0.242535354 |
| 50 | decoder_only | 0 | 0.246833154 |
| 51 | decoder_only | 0 | 0.23135041 |
| 52 | decoder_only | 0 | 0.284355933 |
| 53 | decoder_only | 0 | 0.217858855 |
| 54 | decoder_only | 0 | 0.334693718 |
| 55 | decoder_only | 0 | 0.284491563 |
| 56 | decoder_only | 0 | 0.274986083 |
| 57 | decoder_only | 0 | 0.318684568 |
| 58 | decoder_only | 0 | 0.307601747 |
| 59 | decoder_only | 0 | 0.294695725 |
| 60 | decoder_only | 0 | 0.304547267 |
| 61 | decoder_only | 0 | 0.226690115 |
| 62 | decoder_only | 0 | 0.317049492 |
| 63 | decoder_only | 0 | 0.089485915 |
