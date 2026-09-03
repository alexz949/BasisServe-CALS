# Qwen3-8B-Base GQA C1 V96 joint fit

- K remains dense; every one of 8 physical V heads retains rank 96/128.
- Total KV-cache retention is 87.5% (12.5% reduction).
- Encoder initialization: `activation-weighted-svd`.
- Checkpoint policy: `fixed decoder-refitted endpoint after encoder sweep 6`.
- V encoders and head output decoders use a full-layer joint solve with cross-head covariance.
- Fit contexts: 256; held-out diagnostic contexts: 64.
- Mean held-out factor-dtype relative MSE: `0.0552253438`.

| Layer | Exported boundary | Sweep | Held-out relative MSE |
|---:|:---|---:|---:|
| 0 | after_redecoder | 6 | 0.00686187803 |
| 1 | after_redecoder | 6 | 0.0165881486 |
| 2 | after_redecoder | 6 | 0.024090808 |
| 3 | after_redecoder | 6 | 0.023324632 |
| 4 | after_redecoder | 6 | 0.0337952279 |
| 5 | after_redecoder | 6 | 0.0308646422 |
| 6 | after_redecoder | 6 | 0.0434744254 |
| 7 | after_redecoder | 6 | 0.053892009 |
| 8 | after_redecoder | 6 | 0.0673839155 |
| 9 | after_redecoder | 6 | 0.0788557139 |
| 10 | after_redecoder | 6 | 0.0725978844 |
| 11 | after_redecoder | 6 | 0.0692934907 |
| 12 | after_redecoder | 6 | 0.0476689868 |
| 13 | after_redecoder | 6 | 0.0549537694 |
| 14 | after_redecoder | 6 | 0.0655998599 |
| 15 | after_redecoder | 6 | 0.0621042703 |
| 16 | after_redecoder | 6 | 0.0675316412 |
| 17 | after_redecoder | 6 | 0.0473037665 |
| 18 | after_redecoder | 6 | 0.059608564 |
| 19 | after_redecoder | 6 | 0.0403807638 |
| 20 | after_redecoder | 6 | 0.0642275102 |
| 21 | after_redecoder | 6 | 0.0670869722 |
| 22 | after_redecoder | 6 | 0.0591626636 |
| 23 | after_redecoder | 6 | 0.0625006096 |
| 24 | after_redecoder | 6 | 0.0395282115 |
| 25 | after_redecoder | 6 | 0.0629725895 |
| 26 | after_redecoder | 6 | 0.0843553742 |
| 27 | after_redecoder | 6 | 0.0745778803 |
| 28 | after_redecoder | 6 | 0.0711247246 |
| 29 | after_redecoder | 6 | 0.0837914445 |
| 30 | after_redecoder | 6 | 0.0611567765 |
| 31 | after_redecoder | 6 | 0.0813896655 |
| 32 | after_redecoder | 6 | 0.0611205937 |
| 33 | after_redecoder | 6 | 0.0824550297 |
| 34 | after_redecoder | 6 | 0.0392485407 |
| 35 | after_redecoder | 6 | 0.0272393916 |
