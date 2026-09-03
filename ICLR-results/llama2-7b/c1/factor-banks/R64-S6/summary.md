# Llama-2-7B MHA C1 V64 joint fit

- K remains dense; every one of 32 physical V heads retains rank 64/128.
- Total KV-cache retention is 75% (25% reduction).
- Encoder initialization: `activation-weighted-svd`.
- Checkpoint policy: `fixed decoder-refitted endpoint after encoder sweep 6`.
- V encoders and head output decoders use a full-layer joint solve with cross-head covariance.
- Fit contexts: 256; held-out diagnostic contexts: 64.
- Mean held-out factor-dtype relative MSE: `0.174222869`.

| Layer | Exported boundary | Sweep | Held-out relative MSE |
|---:|:---|---:|---:|
| 0 | after_redecoder | 6 | 0.000440596502 |
| 1 | after_redecoder | 6 | 0.0472510803 |
| 2 | after_redecoder | 6 | 0.105372154 |
| 3 | after_redecoder | 6 | 0.075230408 |
| 4 | after_redecoder | 6 | 0.108668729 |
| 5 | after_redecoder | 6 | 0.103460474 |
| 6 | after_redecoder | 6 | 0.119947032 |
| 7 | after_redecoder | 6 | 0.138473488 |
| 8 | after_redecoder | 6 | 0.156399602 |
| 9 | after_redecoder | 6 | 0.174119231 |
| 10 | after_redecoder | 6 | 0.175578157 |
| 11 | after_redecoder | 6 | 0.184002134 |
| 12 | after_redecoder | 6 | 0.191163946 |
| 13 | after_redecoder | 6 | 0.190170133 |
| 14 | after_redecoder | 6 | 0.208630726 |
| 15 | after_redecoder | 6 | 0.181984842 |
| 16 | after_redecoder | 6 | 0.180342017 |
| 17 | after_redecoder | 6 | 0.220893739 |
| 18 | after_redecoder | 6 | 0.214033387 |
| 19 | after_redecoder | 6 | 0.221518088 |
| 20 | after_redecoder | 6 | 0.184686632 |
| 21 | after_redecoder | 6 | 0.261836259 |
| 22 | after_redecoder | 6 | 0.191877805 |
| 23 | after_redecoder | 6 | 0.265210811 |
| 24 | after_redecoder | 6 | 0.203571069 |
| 25 | after_redecoder | 6 | 0.293148202 |
| 26 | after_redecoder | 6 | 0.181279501 |
| 27 | after_redecoder | 6 | 0.236629996 |
| 28 | after_redecoder | 6 | 0.237073323 |
| 29 | after_redecoder | 6 | 0.228887576 |
| 30 | after_redecoder | 6 | 0.196160469 |
| 31 | after_redecoder | 6 | 0.0970901997 |
