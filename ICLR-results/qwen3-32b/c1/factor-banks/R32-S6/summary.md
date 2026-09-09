# Qwen3-32B GQA C1 V32 joint fit

- K remains dense; every one of 8 physical V heads retains rank 32/128.
- Total KV-cache retention is 62.5% (37.5% reduction).
- Encoder initialization: `activation-weighted-svd`.
- Checkpoint policy: `fixed decoder-refitted endpoint after encoder sweep 6`.
- V encoders and head output decoders use a full-layer joint solve with cross-head covariance.
- Fit contexts: 256; held-out diagnostic contexts: 64.
- Mean held-out factor-dtype relative MSE: `0.280979278`.

| Layer | Exported boundary | Sweep | Held-out relative MSE |
|---:|:---|---:|---:|
| 0 | after_redecoder | 6 | 0.0618740541 |
| 1 | after_redecoder | 6 | 0.0622623787 |
| 2 | after_redecoder | 6 | 0.124122746 |
| 3 | after_redecoder | 6 | 0.133813447 |
| 4 | after_redecoder | 6 | 0.14756308 |
| 5 | after_redecoder | 6 | 0.1152856 |
| 6 | after_redecoder | 6 | 0.11151383 |
| 7 | after_redecoder | 6 | 0.293714247 |
| 8 | after_redecoder | 6 | 0.206153267 |
| 9 | after_redecoder | 6 | 0.165957212 |
| 10 | after_redecoder | 6 | 0.209744957 |
| 11 | after_redecoder | 6 | 0.326184278 |
| 12 | after_redecoder | 6 | 0.217165893 |
| 13 | after_redecoder | 6 | 0.275393766 |
| 14 | after_redecoder | 6 | 0.270070168 |
| 15 | after_redecoder | 6 | 0.203862832 |
| 16 | after_redecoder | 6 | 0.187206969 |
| 17 | after_redecoder | 6 | 0.138149502 |
| 18 | after_redecoder | 6 | 0.183184498 |
| 19 | after_redecoder | 6 | 0.217119646 |
| 20 | after_redecoder | 6 | 0.223831626 |
| 21 | after_redecoder | 6 | 0.260776503 |
| 22 | after_redecoder | 6 | 0.274420275 |
| 23 | after_redecoder | 6 | 0.292640728 |
| 24 | after_redecoder | 6 | 0.386205583 |
| 25 | after_redecoder | 6 | 0.323099083 |
| 26 | after_redecoder | 6 | 0.320583768 |
| 27 | after_redecoder | 6 | 0.33106484 |
| 28 | after_redecoder | 6 | 0.429653699 |
| 29 | after_redecoder | 6 | 0.325754647 |
| 30 | after_redecoder | 6 | 0.479598738 |
| 31 | after_redecoder | 6 | 0.402107462 |
| 32 | after_redecoder | 6 | 0.251701129 |
| 33 | after_redecoder | 6 | 0.224288859 |
| 34 | after_redecoder | 6 | 0.27952851 |
| 35 | after_redecoder | 6 | 0.385476954 |
| 36 | after_redecoder | 6 | 0.302067692 |
| 37 | after_redecoder | 6 | 0.356683347 |
| 38 | after_redecoder | 6 | 0.342181177 |
| 39 | after_redecoder | 6 | 0.293196964 |
| 40 | after_redecoder | 6 | 0.272469785 |
| 41 | after_redecoder | 6 | 0.216187412 |
| 42 | after_redecoder | 6 | 0.292653028 |
| 43 | after_redecoder | 6 | 0.307524567 |
| 44 | after_redecoder | 6 | 0.326644879 |
| 45 | after_redecoder | 6 | 0.290408553 |
| 46 | after_redecoder | 6 | 0.31181274 |
| 47 | after_redecoder | 6 | 0.359259971 |
| 48 | after_redecoder | 6 | 0.291062606 |
| 49 | after_redecoder | 6 | 0.319545632 |
| 50 | after_redecoder | 6 | 0.326870656 |
| 51 | after_redecoder | 6 | 0.313895698 |
| 52 | after_redecoder | 6 | 0.389202471 |
| 53 | after_redecoder | 6 | 0.294788873 |
| 54 | after_redecoder | 6 | 0.440124568 |
| 55 | after_redecoder | 6 | 0.369342341 |
| 56 | after_redecoder | 6 | 0.35825498 |
| 57 | after_redecoder | 6 | 0.410227087 |
| 58 | after_redecoder | 6 | 0.397614587 |
| 59 | after_redecoder | 6 | 0.382045933 |
| 60 | after_redecoder | 6 | 0.37943578 |
| 61 | after_redecoder | 6 | 0.29590649 |
| 62 | after_redecoder | 6 | 0.389815555 |
| 63 | after_redecoder | 6 | 0.112375677 |
