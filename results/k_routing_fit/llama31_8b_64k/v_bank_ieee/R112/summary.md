# Llama-3.1-8B GQA C1 V112 joint fit

- K remains dense; every one of 8 physical V heads retains rank 112/128.
- Total KV-cache retention is 93.75% (6.25% reduction).
- Encoder initialization: `activation-weighted-svd`.
- Checkpoint policy: `fixed decoder-refitted endpoint after encoder sweep 6`.
- V encoders and head output decoders use a full-layer joint solve with cross-head covariance.
- Fit contexts: 64; held-out diagnostic contexts: 16.
- Mean held-out factor-dtype relative MSE: `0.0272166331`.

| Layer | Exported boundary | Sweep | Held-out relative MSE |
|---:|:---|---:|---:|
| 0 | after_redecoder | 6 | 0.00511751869 |
| 1 | after_redecoder | 6 | 0.00917219768 |
| 2 | after_redecoder | 6 | 0.00952097577 |
| 3 | after_redecoder | 6 | 0.0174202674 |
| 4 | after_redecoder | 6 | 0.0185279504 |
| 5 | after_redecoder | 6 | 0.019601263 |
| 6 | after_redecoder | 6 | 0.0263081797 |
| 7 | after_redecoder | 6 | 0.0222475688 |
| 8 | after_redecoder | 6 | 0.0271514597 |
| 9 | after_redecoder | 6 | 0.0264529444 |
| 10 | after_redecoder | 6 | 0.030472043 |
| 11 | after_redecoder | 6 | 0.0271310525 |
| 12 | after_redecoder | 6 | 0.0288690575 |
| 13 | after_redecoder | 6 | 0.0290815611 |
| 14 | after_redecoder | 6 | 0.0283805964 |
| 15 | after_redecoder | 6 | 0.0365335903 |
| 16 | after_redecoder | 6 | 0.0296724009 |
| 17 | after_redecoder | 6 | 0.0341393261 |
| 18 | after_redecoder | 6 | 0.0328896447 |
| 19 | after_redecoder | 6 | 0.0365072987 |
| 20 | after_redecoder | 6 | 0.0401874933 |
| 21 | after_redecoder | 6 | 0.032105674 |
| 22 | after_redecoder | 6 | 0.0442844972 |
| 23 | after_redecoder | 6 | 0.0425212258 |
| 24 | after_redecoder | 6 | 0.04164357 |
| 25 | after_redecoder | 6 | 0.0354589457 |
| 26 | after_redecoder | 6 | 0.0274409103 |
| 27 | after_redecoder | 6 | 0.0319449428 |
| 28 | after_redecoder | 6 | 0.0305550252 |
| 29 | after_redecoder | 6 | 0.0245939672 |
| 30 | after_redecoder | 6 | 0.0179855061 |
| 31 | after_redecoder | 6 | 0.00701360381 |
