# Llama-3.1-8B GQA C1 V32 joint fit

- K remains dense; every one of 8 physical V heads retains rank 32/128.
- Total KV-cache retention is 62.5% (37.5% reduction).
- Encoder initialization: `activation-weighted-svd`.
- Checkpoint policy: `fixed decoder-refitted endpoint after encoder sweep 6`.
- V encoders and head output decoders use a full-layer joint solve with cross-head covariance.
- Fit contexts: 256; held-out diagnostic contexts: 64.
- Mean held-out factor-dtype relative MSE: `0.305067202`.

| Layer | Exported boundary | Sweep | Held-out relative MSE |
|---:|:---|---:|---:|
| 0 | after_redecoder | 6 | 0.074186064 |
| 1 | after_redecoder | 6 | 0.144137276 |
| 2 | after_redecoder | 6 | 0.133266218 |
| 3 | after_redecoder | 6 | 0.225219883 |
| 4 | after_redecoder | 6 | 0.24025611 |
| 5 | after_redecoder | 6 | 0.263156251 |
| 6 | after_redecoder | 6 | 0.292500569 |
| 7 | after_redecoder | 6 | 0.254038917 |
| 8 | after_redecoder | 6 | 0.312936886 |
| 9 | after_redecoder | 6 | 0.306409979 |
| 10 | after_redecoder | 6 | 0.319104969 |
| 11 | after_redecoder | 6 | 0.298537634 |
| 12 | after_redecoder | 6 | 0.302368774 |
| 13 | after_redecoder | 6 | 0.329038504 |
| 14 | after_redecoder | 6 | 0.31434541 |
| 15 | after_redecoder | 6 | 0.390293766 |
| 16 | after_redecoder | 6 | 0.347039902 |
| 17 | after_redecoder | 6 | 0.380038169 |
| 18 | after_redecoder | 6 | 0.369671193 |
| 19 | after_redecoder | 6 | 0.439395969 |
| 20 | after_redecoder | 6 | 0.416986638 |
| 21 | after_redecoder | 6 | 0.343379702 |
| 22 | after_redecoder | 6 | 0.437253789 |
| 23 | after_redecoder | 6 | 0.432967599 |
| 24 | after_redecoder | 6 | 0.449186714 |
| 25 | after_redecoder | 6 | 0.384495891 |
| 26 | after_redecoder | 6 | 0.321587487 |
| 27 | after_redecoder | 6 | 0.360578918 |
| 28 | after_redecoder | 6 | 0.284664905 |
| 29 | after_redecoder | 6 | 0.297515059 |
| 30 | after_redecoder | 6 | 0.196024465 |
| 31 | after_redecoder | 6 | 0.101566865 |
