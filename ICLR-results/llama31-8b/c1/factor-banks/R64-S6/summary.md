# Llama-3.1-8B GQA C1 V64 joint fit

- K remains dense; every one of 8 physical V heads retains rank 64/128.
- Total KV-cache retention is 75% (25% reduction).
- Encoder initialization: `activation-weighted-svd`.
- Checkpoint policy: `fixed decoder-refitted endpoint after encoder sweep 6`.
- V encoders and head output decoders use a full-layer joint solve with cross-head covariance.
- Fit contexts: 256; held-out diagnostic contexts: 64.
- Mean held-out factor-dtype relative MSE: `0.169440184`.

| Layer | Exported boundary | Sweep | Held-out relative MSE |
|---:|:---|---:|---:|
| 0 | after_redecoder | 6 | 0.0351159376 |
| 1 | after_redecoder | 6 | 0.0645987747 |
| 2 | after_redecoder | 6 | 0.0656726968 |
| 3 | after_redecoder | 6 | 0.114401359 |
| 4 | after_redecoder | 6 | 0.124265184 |
| 5 | after_redecoder | 6 | 0.142723776 |
| 6 | after_redecoder | 6 | 0.157068724 |
| 7 | after_redecoder | 6 | 0.133175856 |
| 8 | after_redecoder | 6 | 0.17050589 |
| 9 | after_redecoder | 6 | 0.163449504 |
| 10 | after_redecoder | 6 | 0.174236007 |
| 11 | after_redecoder | 6 | 0.161828233 |
| 12 | after_redecoder | 6 | 0.163099186 |
| 13 | after_redecoder | 6 | 0.176653918 |
| 14 | after_redecoder | 6 | 0.168811868 |
| 15 | after_redecoder | 6 | 0.212981564 |
| 16 | after_redecoder | 6 | 0.182055973 |
| 17 | after_redecoder | 6 | 0.207267795 |
| 18 | after_redecoder | 6 | 0.200872186 |
| 19 | after_redecoder | 6 | 0.262155161 |
| 20 | after_redecoder | 6 | 0.23501151 |
| 21 | after_redecoder | 6 | 0.189816034 |
| 22 | after_redecoder | 6 | 0.250333975 |
| 23 | after_redecoder | 6 | 0.244356649 |
| 24 | after_redecoder | 6 | 0.269524679 |
| 25 | after_redecoder | 6 | 0.219392025 |
| 26 | after_redecoder | 6 | 0.19654694 |
| 27 | after_redecoder | 6 | 0.219272453 |
| 28 | after_redecoder | 6 | 0.16914388 |
| 29 | after_redecoder | 6 | 0.178112272 |
| 30 | after_redecoder | 6 | 0.117699355 |
| 31 | after_redecoder | 6 | 0.0519365165 |
