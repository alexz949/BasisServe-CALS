# Qwen3-32B GQA C1 V64 joint fit

- K remains dense; every one of 8 physical V heads retains rank 64/128.
- Total KV-cache retention is 75% (25% reduction).
- Encoder initialization: `activation-weighted-svd`.
- Checkpoint policy: `fixed decoder-refitted endpoint after encoder sweep 6`.
- V encoders and head output decoders use a full-layer joint solve with cross-head covariance.
- Fit contexts: 256; held-out diagnostic contexts: 64.
- Mean held-out factor-dtype relative MSE: `0.137652951`.

| Layer | Exported boundary | Sweep | Held-out relative MSE |
|---:|:---|---:|---:|
| 0 | after_redecoder | 6 | 0.0202857356 |
| 1 | after_redecoder | 6 | 0.0164243327 |
| 2 | after_redecoder | 6 | 0.0393535882 |
| 3 | after_redecoder | 6 | 0.0481833691 |
| 4 | after_redecoder | 6 | 0.0599195359 |
| 5 | after_redecoder | 6 | 0.0445713216 |
| 6 | after_redecoder | 6 | 0.0397146656 |
| 7 | after_redecoder | 6 | 0.134734671 |
| 8 | after_redecoder | 6 | 0.0906565983 |
| 9 | after_redecoder | 6 | 0.0583842619 |
| 10 | after_redecoder | 6 | 0.079786152 |
| 11 | after_redecoder | 6 | 0.147277875 |
| 12 | after_redecoder | 6 | 0.0875149436 |
| 13 | after_redecoder | 6 | 0.121059976 |
| 14 | after_redecoder | 6 | 0.116785487 |
| 15 | after_redecoder | 6 | 0.0797178601 |
| 16 | after_redecoder | 6 | 0.070066418 |
| 17 | after_redecoder | 6 | 0.0489620715 |
| 18 | after_redecoder | 6 | 0.0638347003 |
| 19 | after_redecoder | 6 | 0.0823592284 |
| 20 | after_redecoder | 6 | 0.0848541585 |
| 21 | after_redecoder | 6 | 0.109531188 |
| 22 | after_redecoder | 6 | 0.120703056 |
| 23 | after_redecoder | 6 | 0.125775912 |
| 24 | after_redecoder | 6 | 0.189349842 |
| 25 | after_redecoder | 6 | 0.151476596 |
| 26 | after_redecoder | 6 | 0.149018383 |
| 27 | after_redecoder | 6 | 0.156541883 |
| 28 | after_redecoder | 6 | 0.218340246 |
| 29 | after_redecoder | 6 | 0.163663198 |
| 30 | after_redecoder | 6 | 0.25531288 |
| 31 | after_redecoder | 6 | 0.213080396 |
| 32 | after_redecoder | 6 | 0.122349954 |
| 33 | after_redecoder | 6 | 0.103497097 |
| 34 | after_redecoder | 6 | 0.133629509 |
| 35 | after_redecoder | 6 | 0.198054398 |
| 36 | after_redecoder | 6 | 0.153523102 |
| 37 | after_redecoder | 6 | 0.187054399 |
| 38 | after_redecoder | 6 | 0.184817882 |
| 39 | after_redecoder | 6 | 0.151865778 |
| 40 | after_redecoder | 6 | 0.139650701 |
| 41 | after_redecoder | 6 | 0.106553585 |
| 42 | after_redecoder | 6 | 0.150737573 |
| 43 | after_redecoder | 6 | 0.160140033 |
| 44 | after_redecoder | 6 | 0.1756343 |
| 45 | after_redecoder | 6 | 0.155861716 |
| 46 | after_redecoder | 6 | 0.164340349 |
| 47 | after_redecoder | 6 | 0.190899182 |
| 48 | after_redecoder | 6 | 0.156251883 |
| 49 | after_redecoder | 6 | 0.168648607 |
| 50 | after_redecoder | 6 | 0.170622985 |
| 51 | after_redecoder | 6 | 0.151181588 |
| 52 | after_redecoder | 6 | 0.191971744 |
| 53 | after_redecoder | 6 | 0.148600881 |
| 54 | after_redecoder | 6 | 0.230893087 |
| 55 | after_redecoder | 6 | 0.194265514 |
| 56 | after_redecoder | 6 | 0.194163794 |
| 57 | after_redecoder | 6 | 0.226758661 |
| 58 | after_redecoder | 6 | 0.221304811 |
| 59 | after_redecoder | 6 | 0.206545249 |
| 60 | after_redecoder | 6 | 0.226811924 |
| 61 | after_redecoder | 6 | 0.159128723 |
| 62 | after_redecoder | 6 | 0.234127955 |
| 63 | after_redecoder | 6 | 0.0626613414 |
