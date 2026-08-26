# Qwen3-8B-Base GQA C1 V112 joint fit

- K remains dense; every one of 8 physical V heads retains rank 112/128.
- Total KV-cache retention is 93.75% (6.25% reduction).
- Encoder initialization: `activation-weighted-svd`.
- Held-out selection boundaries: `decoder-closed`.
- V encoders and head output decoders use a full-layer joint solve with cross-head covariance.
- Fit contexts: 256; held-out selection contexts: 64.
- Mean held-out factor-dtype relative MSE: `0.0246084838`.

| Layer | Selected boundary | Sweep | Held-out relative MSE |
|---:|:---|---:|---:|
| 0 | after_redecoder | 5 | 0.00249170809 |
| 1 | after_redecoder | 5 | 0.00617068944 |
| 2 | after_redecoder | 5 | 0.00966514865 |
| 3 | after_redecoder | 5 | 0.00939231239 |
| 4 | after_redecoder | 5 | 0.0139819889 |
| 5 | after_redecoder | 5 | 0.012544475 |
| 6 | after_redecoder | 5 | 0.0179688888 |
| 7 | after_redecoder | 5 | 0.0233538013 |
| 8 | after_redecoder | 5 | 0.0295555195 |
| 9 | after_redecoder | 5 | 0.0348299374 |
| 10 | after_redecoder | 5 | 0.0322094098 |
| 11 | after_redecoder | 5 | 0.0310765181 |
| 12 | after_redecoder | 5 | 0.0213242572 |
| 13 | after_redecoder | 5 | 0.0248778858 |
| 14 | after_redecoder | 5 | 0.0297310965 |
| 15 | after_redecoder | 5 | 0.0283521151 |
| 16 | after_redecoder | 5 | 0.030781287 |
| 17 | after_redecoder | 5 | 0.0216345745 |
| 18 | after_redecoder | 5 | 0.0274558308 |
| 19 | after_redecoder | 5 | 0.0183241983 |
| 20 | after_redecoder | 5 | 0.0294594908 |
| 21 | after_redecoder | 5 | 0.0306689422 |
| 22 | after_redecoder | 5 | 0.0265431275 |
| 23 | after_redecoder | 5 | 0.0281040909 |
| 24 | after_redecoder | 5 | 0.0168115554 |
| 25 | after_redecoder | 5 | 0.0286146559 |
| 26 | after_redecoder | 5 | 0.0385588669 |
| 27 | after_redecoder | 5 | 0.0337084958 |
| 28 | after_redecoder | 5 | 0.0322830299 |
| 29 | after_redecoder | 5 | 0.0375818986 |
| 30 | after_redecoder | 5 | 0.0280113268 |
| 31 | after_redecoder | 5 | 0.0362387182 |
| 32 | after_redecoder | 5 | 0.0273382529 |
| 33 | after_redecoder | 5 | 0.0362702935 |
| 34 | after_redecoder | 5 | 0.0175395131 |
| 35 | after_redecoder | 5 | 0.0124515171 |
