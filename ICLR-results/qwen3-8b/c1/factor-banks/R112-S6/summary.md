# Qwen3-8B-Base GQA C1 V112 joint fit

- K remains dense; every one of 8 physical V heads retains rank 112/128.
- Total KV-cache retention is 93.75% (6.25% reduction).
- Encoder initialization: `activation-weighted-svd`.
- Checkpoint policy: `fixed decoder-refitted endpoint after encoder sweep 6`.
- V encoders and head output decoders use a full-layer joint solve with cross-head covariance.
- Fit contexts: 256; held-out diagnostic contexts: 64.
- Mean held-out factor-dtype relative MSE: `0.0245266682`.

| Layer | Exported boundary | Sweep | Held-out relative MSE |
|---:|:---|---:|---:|
| 0 | after_redecoder | 6 | 0.00247419511 |
| 1 | after_redecoder | 6 | 0.00615735661 |
| 2 | after_redecoder | 6 | 0.00964894865 |
| 3 | after_redecoder | 6 | 0.00937333802 |
| 4 | after_redecoder | 6 | 0.0139628622 |
| 5 | after_redecoder | 6 | 0.0125126418 |
| 6 | after_redecoder | 6 | 0.0179101943 |
| 7 | after_redecoder | 6 | 0.0233028295 |
| 8 | after_redecoder | 6 | 0.029506885 |
| 9 | after_redecoder | 6 | 0.0347287921 |
| 10 | after_redecoder | 6 | 0.0321344186 |
| 11 | after_redecoder | 6 | 0.0309998695 |
| 12 | after_redecoder | 6 | 0.0212195459 |
| 13 | after_redecoder | 6 | 0.0247477693 |
| 14 | after_redecoder | 6 | 0.0295936077 |
| 15 | after_redecoder | 6 | 0.0282383221 |
| 16 | after_redecoder | 6 | 0.0306630688 |
| 17 | after_redecoder | 6 | 0.0215393496 |
| 18 | after_redecoder | 6 | 0.0273545153 |
| 19 | after_redecoder | 6 | 0.0181995598 |
| 20 | after_redecoder | 6 | 0.0293770242 |
| 21 | after_redecoder | 6 | 0.0305450014 |
| 22 | after_redecoder | 6 | 0.0264503441 |
| 23 | after_redecoder | 6 | 0.028002994 |
| 24 | after_redecoder | 6 | 0.0167663914 |
| 25 | after_redecoder | 6 | 0.028534794 |
| 26 | after_redecoder | 6 | 0.0384662154 |
| 27 | after_redecoder | 6 | 0.0336092874 |
| 28 | after_redecoder | 6 | 0.0321989234 |
| 29 | after_redecoder | 6 | 0.0374795815 |
| 30 | after_redecoder | 6 | 0.0279082371 |
| 31 | after_redecoder | 6 | 0.0361280514 |
| 32 | after_redecoder | 6 | 0.0272771104 |
| 33 | after_redecoder | 6 | 0.0361481922 |
| 34 | after_redecoder | 6 | 0.0174077481 |
| 35 | after_redecoder | 6 | 0.0123920903 |
