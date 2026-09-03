# Llama-3.1-8B GQA C1 V96 joint fit

- K remains dense; every one of 8 physical V heads retains rank 96/128.
- Total KV-cache retention is 87.5% (12.5% reduction).
- Encoder initialization: `activation-weighted-svd`.
- Checkpoint policy: `fixed decoder-refitted endpoint after encoder sweep 6`.
- V encoders and head output decoders use a full-layer joint solve with cross-head covariance.
- Fit contexts: 256; held-out diagnostic contexts: 64.
- Mean held-out factor-dtype relative MSE: `0.0742511947`.

| Layer | Exported boundary | Sweep | Held-out relative MSE |
|---:|:---|---:|---:|
| 0 | after_redecoder | 6 | 0.0130370293 |
| 1 | after_redecoder | 6 | 0.0239537358 |
| 2 | after_redecoder | 6 | 0.0258289889 |
| 3 | after_redecoder | 6 | 0.0465092278 |
| 4 | after_redecoder | 6 | 0.0518253463 |
| 5 | after_redecoder | 6 | 0.0605324134 |
| 6 | after_redecoder | 6 | 0.0666133619 |
| 7 | after_redecoder | 6 | 0.0562198047 |
| 8 | after_redecoder | 6 | 0.0734617511 |
| 9 | after_redecoder | 6 | 0.069020518 |
| 10 | after_redecoder | 6 | 0.07533967 |
| 11 | after_redecoder | 6 | 0.0698770487 |
| 12 | after_redecoder | 6 | 0.0699746173 |
| 13 | after_redecoder | 6 | 0.0748650203 |
| 14 | after_redecoder | 6 | 0.0709412859 |
| 15 | after_redecoder | 6 | 0.0895471971 |
| 16 | after_redecoder | 6 | 0.0764035259 |
| 17 | after_redecoder | 6 | 0.0894238659 |
| 18 | after_redecoder | 6 | 0.0876229236 |
| 19 | after_redecoder | 6 | 0.125923204 |
| 20 | after_redecoder | 6 | 0.102467108 |
| 21 | after_redecoder | 6 | 0.0828743009 |
| 22 | after_redecoder | 6 | 0.110152501 |
| 23 | after_redecoder | 6 | 0.107294012 |
| 24 | after_redecoder | 6 | 0.12515017 |
| 25 | after_redecoder | 6 | 0.0969980645 |
| 26 | after_redecoder | 6 | 0.0906902654 |
| 27 | after_redecoder | 6 | 0.102634277 |
| 28 | after_redecoder | 6 | 0.0766449523 |
| 29 | after_redecoder | 6 | 0.0839804798 |
| 30 | after_redecoder | 6 | 0.0591742384 |
| 31 | after_redecoder | 6 | 0.0210573251 |
