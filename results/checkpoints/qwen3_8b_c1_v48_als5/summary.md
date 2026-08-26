# Qwen3-8B-Base GQA C1 V48 joint fit

- K remains dense; every one of 8 physical V heads retains rank 48/128.
- Total KV-cache retention is 68.75% (31.25% reduction).
- Encoder initialization: `activation-weighted-svd`.
- Held-out selection boundaries: `decoder-closed`.
- V encoders and head output decoders use a full-layer joint solve with cross-head covariance.
- Fit contexts: 256; held-out selection contexts: 64.
- Mean held-out factor-dtype relative MSE: `0.192883217`.

| Layer | Selected boundary | Sweep | Held-out relative MSE |
|---:|:---|---:|---:|
| 0 | after_redecoder | 5 | 0.0295467349 |
| 1 | after_redecoder | 5 | 0.0862998627 |
| 2 | after_redecoder | 5 | 0.104900192 |
| 3 | after_redecoder | 5 | 0.101908491 |
| 4 | after_redecoder | 5 | 0.139254296 |
| 5 | after_redecoder | 5 | 0.130425819 |
| 6 | after_redecoder | 5 | 0.179572198 |
| 7 | after_redecoder | 5 | 0.203426131 |
| 8 | after_redecoder | 5 | 0.239148068 |
| 9 | after_redecoder | 5 | 0.269759638 |
| 10 | after_redecoder | 5 | 0.249880544 |
| 11 | after_redecoder | 5 | 0.235957952 |
| 12 | after_redecoder | 5 | 0.168669972 |
| 13 | after_redecoder | 5 | 0.189311434 |
| 14 | after_redecoder | 5 | 0.222201032 |
| 15 | after_redecoder | 5 | 0.209655542 |
| 16 | after_redecoder | 5 | 0.223169993 |
| 17 | after_redecoder | 5 | 0.159225313 |
| 18 | after_redecoder | 5 | 0.196357392 |
| 19 | after_redecoder | 5 | 0.146501089 |
| 20 | after_redecoder | 5 | 0.211294236 |
| 21 | after_redecoder | 5 | 0.225017676 |
| 22 | after_redecoder | 5 | 0.200823483 |
| 23 | after_redecoder | 5 | 0.219895466 |
| 24 | after_redecoder | 5 | 0.162140468 |
| 25 | after_redecoder | 5 | 0.21428026 |
| 26 | after_redecoder | 5 | 0.277469897 |
| 27 | after_redecoder | 5 | 0.247565912 |
| 28 | after_redecoder | 5 | 0.235943086 |
| 29 | after_redecoder | 5 | 0.283017921 |
| 30 | after_redecoder | 5 | 0.205377978 |
| 31 | after_redecoder | 5 | 0.277030246 |
| 32 | after_redecoder | 5 | 0.211250374 |
| 33 | after_redecoder | 5 | 0.277179315 |
| 34 | after_redecoder | 5 | 0.126958194 |
| 35 | after_redecoder | 5 | 0.0833796194 |
