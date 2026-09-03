# Llama-3.1-8B GQA C1 V80 joint fit

- K remains dense; every one of 8 physical V heads retains rank 80/128.
- Total KV-cache retention is 81.25% (18.75% reduction).
- Encoder initialization: `activation-weighted-svd`.
- Checkpoint policy: `fixed decoder-refitted endpoint after encoder sweep 6`.
- V encoders and head output decoders use a full-layer joint solve with cross-head covariance.
- Fit contexts: 256; held-out diagnostic contexts: 64.
- Mean held-out factor-dtype relative MSE: `0.118599221`.

| Layer | Exported boundary | Sweep | Held-out relative MSE |
|---:|:---|---:|---:|
| 0 | after_redecoder | 6 | 0.0226897101 |
| 1 | after_redecoder | 6 | 0.0415537306 |
| 2 | after_redecoder | 6 | 0.0435179183 |
| 3 | after_redecoder | 6 | 0.0771629897 |
| 4 | after_redecoder | 6 | 0.0848509757 |
| 5 | after_redecoder | 6 | 0.0982344482 |
| 6 | after_redecoder | 6 | 0.108313061 |
| 7 | after_redecoder | 6 | 0.0914055803 |
| 8 | after_redecoder | 6 | 0.11817552 |
| 9 | after_redecoder | 6 | 0.112370005 |
| 10 | after_redecoder | 6 | 0.12130148 |
| 11 | after_redecoder | 6 | 0.112280852 |
| 12 | after_redecoder | 6 | 0.113190056 |
| 13 | after_redecoder | 6 | 0.12156187 |
| 14 | after_redecoder | 6 | 0.116131272 |
| 15 | after_redecoder | 6 | 0.146294622 |
| 16 | after_redecoder | 6 | 0.124880484 |
| 17 | after_redecoder | 6 | 0.143814947 |
| 18 | after_redecoder | 6 | 0.141508947 |
| 19 | after_redecoder | 6 | 0.191183994 |
| 20 | after_redecoder | 6 | 0.164496543 |
| 21 | after_redecoder | 6 | 0.133853758 |
| 22 | after_redecoder | 6 | 0.1762813 |
| 23 | after_redecoder | 6 | 0.171012958 |
| 24 | after_redecoder | 6 | 0.190935271 |
| 25 | after_redecoder | 6 | 0.154816631 |
| 26 | after_redecoder | 6 | 0.142530556 |
| 27 | after_redecoder | 6 | 0.158585647 |
| 28 | after_redecoder | 6 | 0.120690627 |
| 29 | after_redecoder | 6 | 0.129739888 |
| 30 | after_redecoder | 6 | 0.0871421967 |
| 31 | after_redecoder | 6 | 0.0346672463 |
