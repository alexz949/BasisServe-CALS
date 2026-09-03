# Llama-2-7B MHA C1 V48 joint fit

- K remains dense; every one of 32 physical V heads retains rank 48/128.
- Total KV-cache retention is 68.75% (31.25% reduction).
- Encoder initialization: `activation-weighted-svd`.
- Checkpoint policy: `fixed decoder-refitted endpoint after encoder sweep 6`.
- V encoders and head output decoders use a full-layer joint solve with cross-head covariance.
- Fit contexts: 256; held-out diagnostic contexts: 64.
- Mean held-out factor-dtype relative MSE: `0.240229879`.

| Layer | Exported boundary | Sweep | Held-out relative MSE |
|---:|:---|---:|---:|
| 0 | after_redecoder | 6 | 0.0011208266 |
| 1 | after_redecoder | 6 | 0.0729014885 |
| 2 | after_redecoder | 6 | 0.157006333 |
| 3 | after_redecoder | 6 | 0.108615722 |
| 4 | after_redecoder | 6 | 0.156766111 |
| 5 | after_redecoder | 6 | 0.150375184 |
| 6 | after_redecoder | 6 | 0.170805885 |
| 7 | after_redecoder | 6 | 0.193801367 |
| 8 | after_redecoder | 6 | 0.218484806 |
| 9 | after_redecoder | 6 | 0.241854965 |
| 10 | after_redecoder | 6 | 0.245339012 |
| 11 | after_redecoder | 6 | 0.254679363 |
| 12 | after_redecoder | 6 | 0.264457209 |
| 13 | after_redecoder | 6 | 0.263087025 |
| 14 | after_redecoder | 6 | 0.28781598 |
| 15 | after_redecoder | 6 | 0.254512637 |
| 16 | after_redecoder | 6 | 0.256795769 |
| 17 | after_redecoder | 6 | 0.30479924 |
| 18 | after_redecoder | 6 | 0.295801639 |
| 19 | after_redecoder | 6 | 0.305689589 |
| 20 | after_redecoder | 6 | 0.26019568 |
| 21 | after_redecoder | 6 | 0.357735291 |
| 22 | after_redecoder | 6 | 0.262039074 |
| 23 | after_redecoder | 6 | 0.356672402 |
| 24 | after_redecoder | 6 | 0.275416712 |
| 25 | after_redecoder | 6 | 0.393117691 |
| 26 | after_redecoder | 6 | 0.245548312 |
| 27 | after_redecoder | 6 | 0.317418679 |
| 28 | after_redecoder | 6 | 0.315511881 |
| 29 | after_redecoder | 6 | 0.303702535 |
| 30 | after_redecoder | 6 | 0.261474676 |
| 31 | after_redecoder | 6 | 0.133813028 |
