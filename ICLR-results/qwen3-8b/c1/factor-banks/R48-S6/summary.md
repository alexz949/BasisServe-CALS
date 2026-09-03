# Qwen3-8B-Base GQA C1 V48 joint fit

- K remains dense; every one of 8 physical V heads retains rank 48/128.
- Total KV-cache retention is 68.75% (31.25% reduction).
- Encoder initialization: `activation-weighted-svd`.
- Checkpoint policy: `fixed decoder-refitted endpoint after encoder sweep 6`.
- V encoders and head output decoders use a full-layer joint solve with cross-head covariance.
- Fit contexts: 256; held-out diagnostic contexts: 64.
- Mean held-out factor-dtype relative MSE: `0.19262199`.

| Layer | Exported boundary | Sweep | Held-out relative MSE |
|---:|:---|---:|---:|
| 0 | after_redecoder | 6 | 0.029472402 |
| 1 | after_redecoder | 6 | 0.0861847635 |
| 2 | after_redecoder | 6 | 0.104777987 |
| 3 | after_redecoder | 6 | 0.101791712 |
| 4 | after_redecoder | 6 | 0.139065941 |
| 5 | after_redecoder | 6 | 0.130179225 |
| 6 | after_redecoder | 6 | 0.179325192 |
| 7 | after_redecoder | 6 | 0.203112735 |
| 8 | after_redecoder | 6 | 0.238934744 |
| 9 | after_redecoder | 6 | 0.269524751 |
| 10 | after_redecoder | 6 | 0.249663135 |
| 11 | after_redecoder | 6 | 0.23572433 |
| 12 | after_redecoder | 6 | 0.168311404 |
| 13 | after_redecoder | 6 | 0.188984956 |
| 14 | after_redecoder | 6 | 0.221835025 |
| 15 | after_redecoder | 6 | 0.209203646 |
| 16 | after_redecoder | 6 | 0.222874312 |
| 17 | after_redecoder | 6 | 0.158933911 |
| 18 | after_redecoder | 6 | 0.196073938 |
| 19 | after_redecoder | 6 | 0.146219671 |
| 20 | after_redecoder | 6 | 0.210937948 |
| 21 | after_redecoder | 6 | 0.224642767 |
| 22 | after_redecoder | 6 | 0.200508574 |
| 23 | after_redecoder | 6 | 0.219590587 |
| 24 | after_redecoder | 6 | 0.161814429 |
| 25 | after_redecoder | 6 | 0.214046618 |
| 26 | after_redecoder | 6 | 0.277243856 |
| 27 | after_redecoder | 6 | 0.247274719 |
| 28 | after_redecoder | 6 | 0.235736468 |
| 29 | after_redecoder | 6 | 0.282597784 |
| 30 | after_redecoder | 6 | 0.205131523 |
| 31 | after_redecoder | 6 | 0.276777733 |
| 32 | after_redecoder | 6 | 0.211027254 |
| 33 | after_redecoder | 6 | 0.276871941 |
| 34 | after_redecoder | 6 | 0.126730835 |
| 35 | after_redecoder | 6 | 0.0832648372 |
