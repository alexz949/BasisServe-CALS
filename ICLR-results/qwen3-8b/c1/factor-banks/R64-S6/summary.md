# Qwen3-8B-Base GQA C1 V64 joint fit

- K remains dense; every one of 8 physical V heads retains rank 64/128.
- Total KV-cache retention is 75% (25% reduction).
- Encoder initialization: `activation-weighted-svd`.
- Checkpoint policy: `fixed decoder-refitted endpoint after encoder sweep 6`.
- V encoders and head output decoders use a full-layer joint solve with cross-head covariance.
- Fit contexts: 256; held-out diagnostic contexts: 64.
- Mean held-out factor-dtype relative MSE: `0.136809567`.

| Layer | Exported boundary | Sweep | Held-out relative MSE |
|---:|:---|---:|---:|
| 0 | after_redecoder | 6 | 0.0201510203 |
| 1 | after_redecoder | 6 | 0.0539074153 |
| 2 | after_redecoder | 6 | 0.0691732938 |
| 3 | after_redecoder | 6 | 0.0673111597 |
| 4 | after_redecoder | 6 | 0.0937240979 |
| 5 | after_redecoder | 6 | 0.0868164241 |
| 6 | after_redecoder | 6 | 0.120836602 |
| 7 | after_redecoder | 6 | 0.140597854 |
| 8 | after_redecoder | 6 | 0.169176156 |
| 9 | after_redecoder | 6 | 0.193650677 |
| 10 | after_redecoder | 6 | 0.178719499 |
| 11 | after_redecoder | 6 | 0.168841319 |
| 12 | after_redecoder | 6 | 0.118700043 |
| 13 | after_redecoder | 6 | 0.134479478 |
| 14 | after_redecoder | 6 | 0.158576985 |
| 15 | after_redecoder | 6 | 0.149900264 |
| 16 | after_redecoder | 6 | 0.161301603 |
| 17 | after_redecoder | 6 | 0.113886276 |
| 18 | after_redecoder | 6 | 0.141712904 |
| 19 | after_redecoder | 6 | 0.101197368 |
| 20 | after_redecoder | 6 | 0.152510491 |
| 21 | after_redecoder | 6 | 0.161539149 |
| 22 | after_redecoder | 6 | 0.143638166 |
| 23 | after_redecoder | 6 | 0.154810119 |
| 24 | after_redecoder | 6 | 0.108201952 |
| 25 | after_redecoder | 6 | 0.153196543 |
| 26 | after_redecoder | 6 | 0.201032488 |
| 27 | after_redecoder | 6 | 0.178752448 |
| 28 | after_redecoder | 6 | 0.169912004 |
| 29 | after_redecoder | 6 | 0.202762135 |
| 30 | after_redecoder | 6 | 0.14705239 |
| 31 | after_redecoder | 6 | 0.200290402 |
| 32 | after_redecoder | 6 | 0.151661956 |
| 33 | after_redecoder | 6 | 0.201489545 |
| 34 | after_redecoder | 6 | 0.0932843847 |
| 35 | after_redecoder | 6 | 0.0623498182 |
