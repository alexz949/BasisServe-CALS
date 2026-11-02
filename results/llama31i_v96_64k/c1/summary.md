# Llama-3.1-8B GQA C1 V96 joint fit

- K remains dense; every one of 8 physical V heads retains rank 96/128.
- Total KV-cache retention is 87.5% (12.5% reduction).
- Encoder initialization: `activation-weighted-svd`.
- Checkpoint policy: `fixed decoder-refitted endpoint after encoder sweep 6`.
- V encoders and head output decoders use a full-layer joint solve with cross-head covariance.
- Fit contexts: 64; held-out diagnostic contexts: 16.
- Mean held-out factor-dtype relative MSE: `0.0522933917`.

| Layer | Exported boundary | Sweep | Held-out relative MSE |
|---:|:---|---:|---:|
| 0 | after_redecoder | 6 | 0.0100225589 |
| 1 | after_redecoder | 6 | 0.0192884228 |
| 2 | after_redecoder | 6 | 0.0203619022 |
| 3 | after_redecoder | 6 | 0.0364193943 |
| 4 | after_redecoder | 6 | 0.0365035205 |
| 5 | after_redecoder | 6 | 0.0383223627 |
| 6 | after_redecoder | 6 | 0.0488792583 |
| 7 | after_redecoder | 6 | 0.0394158371 |
| 8 | after_redecoder | 6 | 0.048444829 |
| 9 | after_redecoder | 6 | 0.0483760213 |
| 10 | after_redecoder | 6 | 0.0548570787 |
| 11 | after_redecoder | 6 | 0.0465378799 |
| 12 | after_redecoder | 6 | 0.046597155 |
| 13 | after_redecoder | 6 | 0.0523190143 |
| 14 | after_redecoder | 6 | 0.049393744 |
| 15 | after_redecoder | 6 | 0.0678276268 |
| 16 | after_redecoder | 6 | 0.0543813825 |
| 17 | after_redecoder | 6 | 0.065996188 |
| 18 | after_redecoder | 6 | 0.0630738544 |
| 19 | after_redecoder | 6 | 0.0677789793 |
| 20 | after_redecoder | 6 | 0.0794967217 |
| 21 | after_redecoder | 6 | 0.0577971351 |
| 22 | after_redecoder | 6 | 0.0906187011 |
| 23 | after_redecoder | 6 | 0.0907145778 |
| 24 | after_redecoder | 6 | 0.0813326639 |
| 25 | after_redecoder | 6 | 0.0732198478 |
| 26 | after_redecoder | 6 | 0.0577787452 |
| 27 | after_redecoder | 6 | 0.0590250754 |
| 28 | after_redecoder | 6 | 0.0656685895 |
| 29 | after_redecoder | 6 | 0.045912143 |
| 30 | after_redecoder | 6 | 0.040883429 |
| 31 | after_redecoder | 6 | 0.0161438946 |
