# Llama-3.1-8B GQA C1 V48 joint fit

- K remains dense; every one of 8 physical V heads retains rank 48/128.
- Total KV-cache retention is 68.75% (31.25% reduction).
- Encoder initialization: `activation-weighted-svd`.
- Checkpoint policy: `fixed decoder-refitted endpoint after encoder sweep 6`.
- V encoders and head output decoders use a full-layer joint solve with cross-head covariance.
- Fit contexts: 256; held-out diagnostic contexts: 64.
- Mean held-out factor-dtype relative MSE: `0.229633673`.

| Layer | Exported boundary | Sweep | Held-out relative MSE |
|---:|:---|---:|---:|
| 0 | after_redecoder | 6 | 0.0514063095 |
| 1 | after_redecoder | 6 | 0.0961636914 |
| 2 | after_redecoder | 6 | 0.0940875509 |
| 3 | after_redecoder | 6 | 0.161670947 |
| 4 | after_redecoder | 6 | 0.173415898 |
| 5 | after_redecoder | 6 | 0.195390499 |
| 6 | after_redecoder | 6 | 0.216466704 |
| 7 | after_redecoder | 6 | 0.185058358 |
| 8 | after_redecoder | 6 | 0.23310317 |
| 9 | after_redecoder | 6 | 0.225960863 |
| 10 | after_redecoder | 6 | 0.238127222 |
| 11 | after_redecoder | 6 | 0.221672936 |
| 12 | after_redecoder | 6 | 0.223636188 |
| 13 | after_redecoder | 6 | 0.243471749 |
| 14 | after_redecoder | 6 | 0.231986424 |
| 15 | after_redecoder | 6 | 0.291840256 |
| 16 | after_redecoder | 6 | 0.253066197 |
| 17 | after_redecoder | 6 | 0.282576605 |
| 18 | after_redecoder | 6 | 0.275541253 |
| 19 | after_redecoder | 6 | 0.342254308 |
| 20 | after_redecoder | 6 | 0.318111826 |
| 21 | after_redecoder | 6 | 0.257681638 |
| 22 | after_redecoder | 6 | 0.335814565 |
| 23 | after_redecoder | 6 | 0.329993743 |
| 24 | after_redecoder | 6 | 0.353099286 |
| 25 | after_redecoder | 6 | 0.292988603 |
| 26 | after_redecoder | 6 | 0.251431926 |
| 27 | after_redecoder | 6 | 0.286160809 |
| 28 | after_redecoder | 6 | 0.224179784 |
| 29 | after_redecoder | 6 | 0.233802741 |
| 30 | after_redecoder | 6 | 0.154629342 |
| 31 | after_redecoder | 6 | 0.0734861415 |
