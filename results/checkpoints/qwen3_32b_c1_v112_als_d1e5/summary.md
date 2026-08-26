# Qwen3-32B GQA C1 V112 joint fit

- K remains dense; every one of 8 physical V heads retains rank 112/128.
- Total KV-cache retention is 93.75% (6.25% reduction).
- Encoder initialization: `activation-weighted-svd`.
- Held-out selection boundaries: `decoder-closed`.
- V encoders and head output decoders use a full-layer joint solve with cross-head covariance.
- Fit contexts: 128; held-out selection contexts: 64.
- Mean held-out factor-dtype relative MSE: `0.0507980847`.

| Layer | Selected boundary | Sweep | Held-out relative MSE |
|---:|:---|---:|---:|
| 0 | after_redecoder | 5 | 0.00253947875 |
| 1 | after_redecoder | 5 | 0.00187773387 |
| 2 | after_redecoder | 5 | 0.00679222622 |
| 3 | after_redecoder | 5 | 0.0090074072 |
| 4 | after_redecoder | 5 | 0.015636889 |
| 5 | after_redecoder | 5 | 0.0109329283 |
| 6 | after_redecoder | 5 | 0.00837927142 |
| 7 | after_redecoder | 5 | 0.0424129574 |
| 8 | after_redecoder | 5 | 0.0320793706 |
| 9 | after_redecoder | 5 | 0.0167700339 |
| 10 | after_redecoder | 5 | 0.0199133713 |
| 11 | after_redecoder | 5 | 0.0462594526 |
| 12 | after_redecoder | 5 | 0.0279019507 |
| 13 | after_redecoder | 5 | 0.03126509 |
| 14 | after_redecoder | 5 | 0.0344133933 |
| 15 | after_redecoder | 5 | 0.0181814206 |
| 16 | after_redecoder | 5 | 0.023052094 |
| 17 | after_redecoder | 5 | 0.0114678383 |
| 18 | after_redecoder | 5 | 0.0158951783 |
| 19 | after_redecoder | 5 | 0.0248303367 |
| 20 | after_redecoder | 5 | 0.0316595426 |
| 21 | after_redecoder | 5 | 0.0306816411 |
| 22 | after_redecoder | 5 | 0.0328521523 |
| 23 | after_redecoder | 5 | 0.0630480971 |
| 24 | after_redecoder | 5 | 0.0634751744 |
| 25 | after_redecoder | 5 | 0.0530019304 |
| 26 | after_redecoder | 5 | 0.0440486892 |
| 27 | after_redecoder | 5 | 0.0538380984 |
| 28 | after_redecoder | 5 | 0.101584753 |
| 29 | after_redecoder | 5 | 0.0569334835 |
| 30 | after_redecoder | 5 | 0.111873984 |
| 31 | after_redecoder | 5 | 0.081359006 |
| 32 | after_redecoder | 5 | 0.0389892424 |
| 33 | after_redecoder | 5 | 0.0347913701 |
| 34 | after_redecoder | 5 | 0.0444012725 |
| 35 | after_redecoder | 5 | 0.0730988969 |
| 36 | after_redecoder | 5 | 0.0510216641 |
| 37 | after_redecoder | 5 | 0.0589095246 |
| 38 | after_redecoder | 5 | 0.0639950001 |
| 39 | after_redecoder | 5 | 0.0514521253 |
| 40 | after_redecoder | 5 | 0.0487737404 |
| 41 | after_redecoder | 5 | 0.0353533653 |
| 42 | after_redecoder | 5 | 0.0538886263 |
| 43 | after_redecoder | 5 | 0.0569679601 |
| 44 | after_redecoder | 5 | 0.0648109443 |
| 45 | after_redecoder | 5 | 0.0551241611 |
| 46 | after_redecoder | 5 | 0.0568790347 |
| 47 | after_redecoder | 5 | 0.0731864276 |
| 48 | after_redecoder | 5 | 0.0579509829 |
| 49 | after_redecoder | 5 | 0.0577287722 |
| 50 | after_redecoder | 5 | 0.0625288864 |
| 51 | after_redecoder | 5 | 0.0505323364 |
| 52 | after_redecoder | 5 | 0.0659529942 |
| 53 | after_redecoder | 5 | 0.0536036177 |
| 54 | after_redecoder | 5 | 0.0832188033 |
| 55 | after_redecoder | 5 | 0.0693286631 |
| 56 | after_redecoder | 5 | 0.0791836504 |
| 57 | after_redecoder | 5 | 0.0922021109 |
| 58 | after_redecoder | 5 | 0.115628124 |
| 59 | after_redecoder | 5 | 0.0795980526 |
| 60 | after_redecoder | 5 | 0.11296948 |
| 61 | after_redecoder | 5 | 0.0605133391 |
| 62 | after_redecoder | 5 | 0.201058163 |
| 63 | after_redecoder | 5 | 0.0234711168 |
