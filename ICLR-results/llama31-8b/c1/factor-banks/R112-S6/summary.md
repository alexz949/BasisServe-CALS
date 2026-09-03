# Llama-3.1-8B GQA C1 V112 joint fit

- K remains dense; every one of 8 physical V heads retains rank 112/128.
- Total KV-cache retention is 93.75% (6.25% reduction).
- Encoder initialization: `activation-weighted-svd`.
- Checkpoint policy: `fixed decoder-refitted endpoint after encoder sweep 6`.
- V encoders and head output decoders use a full-layer joint solve with cross-head covariance.
- Fit contexts: 256; held-out diagnostic contexts: 64.
- Mean held-out factor-dtype relative MSE: `0.0348685214`.

| Layer | Exported boundary | Sweep | Held-out relative MSE |
|---:|:---|---:|---:|
| 0 | after_redecoder | 6 | 0.00543877531 |
| 1 | after_redecoder | 6 | 0.0102386981 |
| 2 | after_redecoder | 6 | 0.0113538513 |
| 3 | after_redecoder | 6 | 0.020725748 |
| 4 | after_redecoder | 6 | 0.0235894021 |
| 5 | after_redecoder | 6 | 0.028894159 |
| 6 | after_redecoder | 6 | 0.0306115884 |
| 7 | after_redecoder | 6 | 0.0258955063 |
| 8 | after_redecoder | 6 | 0.0339674501 |
| 9 | after_redecoder | 6 | 0.0315340794 |
| 10 | after_redecoder | 6 | 0.0349910196 |
| 11 | after_redecoder | 6 | 0.0325302077 |
| 12 | after_redecoder | 6 | 0.0324697118 |
| 13 | after_redecoder | 6 | 0.0342276571 |
| 14 | after_redecoder | 6 | 0.0324545743 |
| 15 | after_redecoder | 6 | 0.0410384753 |
| 16 | after_redecoder | 6 | 0.0346988368 |
| 17 | after_redecoder | 6 | 0.0418881513 |
| 18 | after_redecoder | 6 | 0.0411984816 |
| 19 | after_redecoder | 6 | 0.0652144058 |
| 20 | after_redecoder | 6 | 0.0478663363 |
| 21 | after_redecoder | 6 | 0.0392215964 |
| 22 | after_redecoder | 6 | 0.0508091282 |
| 23 | after_redecoder | 6 | 0.0505471464 |
| 24 | after_redecoder | 6 | 0.0601285198 |
| 25 | after_redecoder | 6 | 0.0435332563 |
| 26 | after_redecoder | 6 | 0.0453054241 |
| 27 | after_redecoder | 6 | 0.048219548 |
| 28 | after_redecoder | 6 | 0.0363327492 |
| 29 | after_redecoder | 6 | 0.0400405433 |
| 30 | after_redecoder | 6 | 0.0314731552 |
| 31 | after_redecoder | 6 | 0.00935450309 |
