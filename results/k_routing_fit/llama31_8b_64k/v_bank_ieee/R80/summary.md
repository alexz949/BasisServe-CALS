# Llama-3.1-8B GQA C1 V80 joint fit

- K remains dense; every one of 8 physical V heads retains rank 80/128.
- Total KV-cache retention is 81.25% (18.75% reduction).
- Encoder initialization: `activation-weighted-svd`.
- Checkpoint policy: `fixed decoder-refitted endpoint after encoder sweep 6`.
- V encoders and head output decoders use a full-layer joint solve with cross-head covariance.
- Fit contexts: 64; held-out diagnostic contexts: 16.
- Mean held-out factor-dtype relative MSE: `0.0973698989`.

| Layer | Exported boundary | Sweep | Held-out relative MSE |
|---:|:---|---:|---:|
| 0 | after_redecoder | 6 | 0.0208429289 |
| 1 | after_redecoder | 6 | 0.0380180773 |
| 2 | after_redecoder | 6 | 0.037206307 |
| 3 | after_redecoder | 6 | 0.0660922811 |
| 4 | after_redecoder | 6 | 0.069059539 |
| 5 | after_redecoder | 6 | 0.0727921839 |
| 6 | after_redecoder | 6 | 0.0941945567 |
| 7 | after_redecoder | 6 | 0.0800845414 |
| 8 | after_redecoder | 6 | 0.0970763212 |
| 9 | after_redecoder | 6 | 0.0953534144 |
| 10 | after_redecoder | 6 | 0.106159784 |
| 11 | after_redecoder | 6 | 0.0956873389 |
| 12 | after_redecoder | 6 | 0.101131785 |
| 13 | after_redecoder | 6 | 0.104596318 |
| 14 | after_redecoder | 6 | 0.102512215 |
| 15 | after_redecoder | 6 | 0.131282422 |
| 16 | after_redecoder | 6 | 0.109302203 |
| 17 | after_redecoder | 6 | 0.122439764 |
| 18 | after_redecoder | 6 | 0.118611638 |
| 19 | after_redecoder | 6 | 0.128244156 |
| 20 | after_redecoder | 6 | 0.144570631 |
| 21 | after_redecoder | 6 | 0.115024506 |
| 22 | after_redecoder | 6 | 0.15591987 |
| 23 | after_redecoder | 6 | 0.151899145 |
| 24 | after_redecoder | 6 | 0.143791837 |
| 25 | after_redecoder | 6 | 0.127156473 |
| 26 | after_redecoder | 6 | 0.0944435186 |
| 27 | after_redecoder | 6 | 0.112835151 |
| 28 | after_redecoder | 6 | 0.10384096 |
| 29 | after_redecoder | 6 | 0.0860195617 |
| 30 | after_redecoder | 6 | 0.0622677517 |
| 31 | after_redecoder | 6 | 0.0273795837 |
