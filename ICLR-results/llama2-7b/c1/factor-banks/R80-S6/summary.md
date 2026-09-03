# Llama-2-7B MHA C1 V80 joint fit

- K remains dense; every one of 32 physical V heads retains rank 80/128.
- Total KV-cache retention is 81.25% (18.75% reduction).
- Encoder initialization: `activation-weighted-svd`.
- Checkpoint policy: `fixed decoder-refitted endpoint after encoder sweep 6`.
- V encoders and head output decoders use a full-layer joint solve with cross-head covariance.
- Fit contexts: 256; held-out diagnostic contexts: 64.
- Mean held-out factor-dtype relative MSE: `0.11935445`.

| Layer | Exported boundary | Sweep | Held-out relative MSE |
|---:|:---|---:|---:|
| 0 | after_redecoder | 6 | 0.000168591781 |
| 1 | after_redecoder | 6 | 0.0288092883 |
| 2 | after_redecoder | 6 | 0.0671084189 |
| 3 | after_redecoder | 6 | 0.0495639532 |
| 4 | after_redecoder | 6 | 0.0715466779 |
| 5 | after_redecoder | 6 | 0.0679588799 |
| 6 | after_redecoder | 6 | 0.0801424707 |
| 7 | after_redecoder | 6 | 0.0937608153 |
| 8 | after_redecoder | 6 | 0.10605814 |
| 9 | after_redecoder | 6 | 0.118964859 |
| 10 | after_redecoder | 6 | 0.119242035 |
| 11 | after_redecoder | 6 | 0.126110992 |
| 12 | after_redecoder | 6 | 0.130586784 |
| 13 | after_redecoder | 6 | 0.130096061 |
| 14 | after_redecoder | 6 | 0.143160615 |
| 15 | after_redecoder | 6 | 0.12300157 |
| 16 | after_redecoder | 6 | 0.120216991 |
| 17 | after_redecoder | 6 | 0.151619937 |
| 18 | after_redecoder | 6 | 0.145742703 |
| 19 | after_redecoder | 6 | 0.151609436 |
| 20 | after_redecoder | 6 | 0.124128344 |
| 21 | after_redecoder | 6 | 0.179083213 |
| 22 | after_redecoder | 6 | 0.132468142 |
| 23 | after_redecoder | 6 | 0.185725941 |
| 24 | after_redecoder | 6 | 0.141195314 |
| 25 | after_redecoder | 6 | 0.204705032 |
| 26 | after_redecoder | 6 | 0.126467125 |
| 27 | after_redecoder | 6 | 0.165813807 |
| 28 | after_redecoder | 6 | 0.167510063 |
| 29 | after_redecoder | 6 | 0.162148738 |
| 30 | after_redecoder | 6 | 0.138136997 |
| 31 | after_redecoder | 6 | 0.0664904746 |
