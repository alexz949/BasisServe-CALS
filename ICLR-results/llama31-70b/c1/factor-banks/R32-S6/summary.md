# Llama-3.1-70B GQA C1 V32 joint fit

- K remains dense; every one of 8 physical V heads retains rank 32/128.
- Total KV-cache retention is 62.5% (37.5% reduction).
- Encoder initialization: `activation-weighted-svd`.
- Checkpoint policy: `fixed decoder-refitted endpoint after encoder sweep 6`.
- V encoders and head output decoders use a full-layer joint solve with cross-head covariance.
- Fit contexts: 256; held-out diagnostic contexts: 64.
- Mean held-out factor-dtype relative MSE: `0.373649067`.

| Layer | Exported boundary | Sweep | Held-out relative MSE |
|---:|:---|---:|---:|
| 0 | after_redecoder | 6 | 0.0106866826 |
| 1 | after_redecoder | 6 | 0.134371223 |
| 2 | after_redecoder | 6 | 0.196184255 |
| 3 | after_redecoder | 6 | 0.135640179 |
| 4 | after_redecoder | 6 | 0.08990915 |
| 5 | after_redecoder | 6 | 0.218097626 |
| 6 | after_redecoder | 6 | 0.238818169 |
| 7 | after_redecoder | 6 | 0.198724921 |
| 8 | after_redecoder | 6 | 0.132019131 |
| 9 | after_redecoder | 6 | 0.186967054 |
| 10 | after_redecoder | 6 | 0.262843399 |
| 11 | after_redecoder | 6 | 0.307853177 |
| 12 | after_redecoder | 6 | 0.142093433 |
| 13 | after_redecoder | 6 | 0.193963146 |
| 14 | after_redecoder | 6 | 0.216050122 |
| 15 | after_redecoder | 6 | 0.235745626 |
| 16 | after_redecoder | 6 | 0.208536808 |
| 17 | after_redecoder | 6 | 0.30701219 |
| 18 | after_redecoder | 6 | 0.384830254 |
| 19 | after_redecoder | 6 | 0.342158366 |
| 20 | after_redecoder | 6 | 0.33277642 |
| 21 | after_redecoder | 6 | 0.416499001 |
| 22 | after_redecoder | 6 | 0.492134325 |
| 23 | after_redecoder | 6 | 0.437527443 |
| 24 | after_redecoder | 6 | 0.295483579 |
| 25 | after_redecoder | 6 | 0.359939973 |
| 26 | after_redecoder | 6 | 0.28630074 |
| 27 | after_redecoder | 6 | 0.314111493 |
| 28 | after_redecoder | 6 | 0.293017946 |
| 29 | after_redecoder | 6 | 0.317839608 |
| 30 | after_redecoder | 6 | 0.323485645 |
| 31 | after_redecoder | 6 | 0.345798264 |
| 32 | after_redecoder | 6 | 0.340997543 |
| 33 | after_redecoder | 6 | 0.354975213 |
| 34 | after_redecoder | 6 | 0.323674005 |
| 35 | after_redecoder | 6 | 0.39226334 |
| 36 | after_redecoder | 6 | 0.404034426 |
| 37 | after_redecoder | 6 | 0.429849866 |
| 38 | after_redecoder | 6 | 0.437524148 |
| 39 | after_redecoder | 6 | 0.511828054 |
| 40 | after_redecoder | 6 | 0.444112455 |
| 41 | after_redecoder | 6 | 0.551745434 |
| 42 | after_redecoder | 6 | 0.603710716 |
| 43 | after_redecoder | 6 | 0.462668175 |
| 44 | after_redecoder | 6 | 0.472747725 |
| 45 | after_redecoder | 6 | 0.500436407 |
| 46 | after_redecoder | 6 | 0.68846022 |
| 47 | after_redecoder | 6 | 0.516085212 |
| 48 | after_redecoder | 6 | 0.598792612 |
| 49 | after_redecoder | 6 | 0.491774747 |
| 50 | after_redecoder | 6 | 0.652048706 |
| 51 | after_redecoder | 6 | 0.508128327 |
| 52 | after_redecoder | 6 | 0.39227539 |
| 53 | after_redecoder | 6 | 0.630151162 |
| 54 | after_redecoder | 6 | 0.539473288 |
| 55 | after_redecoder | 6 | 0.353170697 |
| 56 | after_redecoder | 6 | 0.553587218 |
| 57 | after_redecoder | 6 | 0.558325116 |
| 58 | after_redecoder | 6 | 0.550516462 |
| 59 | after_redecoder | 6 | 0.643729872 |
| 60 | after_redecoder | 6 | 0.578309546 |
| 61 | after_redecoder | 6 | 0.686198596 |
| 62 | after_redecoder | 6 | 0.641746134 |
| 63 | after_redecoder | 6 | 0.527878553 |
| 64 | after_redecoder | 6 | 0.497194512 |
| 65 | after_redecoder | 6 | 0.439410692 |
| 66 | after_redecoder | 6 | 0.383268598 |
| 67 | after_redecoder | 6 | 0.49326389 |
| 68 | after_redecoder | 6 | 0.453153383 |
| 69 | after_redecoder | 6 | 0.429197044 |
| 70 | after_redecoder | 6 | 0.322220582 |
| 71 | after_redecoder | 6 | 0.265051955 |
| 72 | after_redecoder | 6 | 0.217189316 |
| 73 | after_redecoder | 6 | 0.209978296 |
| 74 | after_redecoder | 6 | 0.169630014 |
| 75 | after_redecoder | 6 | 0.247616602 |
| 76 | after_redecoder | 6 | 0.182748219 |
| 77 | after_redecoder | 6 | 0.289199574 |
| 78 | after_redecoder | 6 | 0.391474025 |
| 79 | after_redecoder | 6 | 0.202689914 |
