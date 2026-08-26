# Qwen3-32B GQA C1 V112 joint fit

- K remains dense; every one of 8 physical V heads retains rank 112/128.
- Total KV-cache retention is 93.75% (6.25% reduction).
- Encoder initialization: `activation-weighted-svd`.
- Held-out selection boundaries: `decoder-closed`.
- V encoders and head output decoders use a full-layer joint solve with cross-head covariance.
- Fit contexts: 256; held-out selection contexts: 64.
- Mean held-out factor-dtype relative MSE: `0.023112072`.

| Layer | Selected boundary | Sweep | Held-out relative MSE |
|---:|:---|---:|---:|
| 0 | after_redecoder | 5 | 0.00122317136 |
| 1 | after_redecoder | 5 | 0.000833581963 |
| 2 | after_redecoder | 5 | 0.00316832209 |
| 3 | after_redecoder | 5 | 0.00445358144 |
| 4 | after_redecoder | 5 | 0.00696173955 |
| 5 | after_redecoder | 5 | 0.00469925547 |
| 6 | after_redecoder | 5 | 0.00393313057 |
| 7 | after_redecoder | 5 | 0.0183854857 |
| 8 | after_redecoder | 5 | 0.0125439667 |
| 9 | after_redecoder | 5 | 0.00598488838 |
| 10 | after_redecoder | 5 | 0.00907556289 |
| 11 | after_redecoder | 5 | 0.0213363379 |
| 12 | after_redecoder | 5 | 0.0108499538 |
| 13 | after_redecoder | 5 | 0.0159399026 |
| 14 | after_redecoder | 5 | 0.0155007221 |
| 15 | after_redecoder | 5 | 0.00941525492 |
| 16 | after_redecoder | 5 | 0.00770745627 |
| 17 | after_redecoder | 5 | 0.00505567183 |
| 18 | after_redecoder | 5 | 0.00604479171 |
| 19 | after_redecoder | 5 | 0.00916981748 |
| 20 | after_redecoder | 5 | 0.009834209 |
| 21 | after_redecoder | 5 | 0.0145398331 |
| 22 | after_redecoder | 5 | 0.016655047 |
| 23 | after_redecoder | 5 | 0.0166513426 |
| 24 | after_redecoder | 5 | 0.0299763516 |
| 25 | after_redecoder | 5 | 0.0226340185 |
| 26 | after_redecoder | 5 | 0.0226964466 |
| 27 | after_redecoder | 5 | 0.0236728884 |
| 28 | after_redecoder | 5 | 0.0373064366 |
| 29 | after_redecoder | 5 | 0.0279089656 |
| 30 | after_redecoder | 5 | 0.0455442114 |
| 31 | after_redecoder | 5 | 0.0378178313 |
| 32 | after_redecoder | 5 | 0.019306069 |
| 33 | after_redecoder | 5 | 0.0154449999 |
| 34 | after_redecoder | 5 | 0.0214362712 |
| 35 | after_redecoder | 5 | 0.0356876959 |
| 36 | after_redecoder | 5 | 0.0268608134 |
| 37 | after_redecoder | 5 | 0.0323245928 |
| 38 | after_redecoder | 5 | 0.0348556131 |
| 39 | after_redecoder | 5 | 0.0278437191 |
| 40 | after_redecoder | 5 | 0.0255737176 |
| 41 | after_redecoder | 5 | 0.0192226091 |
| 42 | after_redecoder | 5 | 0.0281183256 |
| 43 | after_redecoder | 5 | 0.0306213551 |
| 44 | after_redecoder | 5 | 0.0337171839 |
| 45 | after_redecoder | 5 | 0.0299271592 |
| 46 | after_redecoder | 5 | 0.0306573731 |
| 47 | after_redecoder | 5 | 0.0367764254 |
| 48 | after_redecoder | 5 | 0.0293505696 |
| 49 | after_redecoder | 5 | 0.0313480619 |
| 50 | after_redecoder | 5 | 0.0319781582 |
| 51 | after_redecoder | 5 | 0.0262148362 |
| 52 | after_redecoder | 5 | 0.0338335967 |
| 53 | after_redecoder | 5 | 0.0269825205 |
| 54 | after_redecoder | 5 | 0.0404477798 |
| 55 | after_redecoder | 5 | 0.0352294135 |
| 56 | after_redecoder | 5 | 0.0369183333 |
| 57 | after_redecoder | 5 | 0.0443820373 |
| 58 | after_redecoder | 5 | 0.0416856393 |
| 59 | after_redecoder | 5 | 0.0379082916 |
| 60 | after_redecoder | 5 | 0.0486104112 |
| 61 | after_redecoder | 5 | 0.0283087329 |
| 62 | after_redecoder | 5 | 0.0482495961 |
| 63 | after_redecoder | 5 | 0.011830528 |
