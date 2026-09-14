# Llama-3.1-8B GQA C1 V96 joint fit

- K remains dense; every one of 8 physical V heads retains rank 96/128.
- Total KV-cache retention is 87.5% (12.5% reduction).
- Encoder initialization: `activation-weighted-svd`.
- Checkpoint policy: `fixed decoder-refitted endpoint after encoder sweep 6`.
- V encoders and head output decoders use a full-layer joint solve with cross-head covariance.
- Fit contexts: 64; held-out diagnostic contexts: 16.
- Mean held-out factor-dtype relative MSE: `0.059625199`.

| Layer | Exported boundary | Sweep | Held-out relative MSE |
|---:|:---|---:|---:|
| 0 | after_redecoder | 6 | 0.012108019 |
| 1 | after_redecoder | 6 | 0.0217044614 |
| 2 | after_redecoder | 6 | 0.0219083846 |
| 3 | after_redecoder | 6 | 0.0393195395 |
| 4 | after_redecoder | 6 | 0.0413675721 |
| 5 | after_redecoder | 6 | 0.0438377317 |
| 6 | after_redecoder | 6 | 0.0576222105 |
| 7 | after_redecoder | 6 | 0.04877627 |
| 8 | after_redecoder | 6 | 0.0593936436 |
| 9 | after_redecoder | 6 | 0.0581123279 |
| 10 | after_redecoder | 6 | 0.0656001889 |
| 11 | after_redecoder | 6 | 0.0587702488 |
| 12 | after_redecoder | 6 | 0.0623549912 |
| 13 | after_redecoder | 6 | 0.0639135449 |
| 14 | after_redecoder | 6 | 0.0625520889 |
| 15 | after_redecoder | 6 | 0.0801113316 |
| 16 | after_redecoder | 6 | 0.0661287854 |
| 17 | after_redecoder | 6 | 0.0748498199 |
| 18 | after_redecoder | 6 | 0.071990608 |
| 19 | after_redecoder | 6 | 0.0791277008 |
| 20 | after_redecoder | 6 | 0.088839711 |
| 21 | after_redecoder | 6 | 0.0704333596 |
| 22 | after_redecoder | 6 | 0.0964081385 |
| 23 | after_redecoder | 6 | 0.0930608857 |
| 24 | after_redecoder | 6 | 0.0897330025 |
| 25 | after_redecoder | 6 | 0.0777433972 |
| 26 | after_redecoder | 6 | 0.0587565805 |
| 27 | after_redecoder | 6 | 0.0697288526 |
| 28 | after_redecoder | 6 | 0.065140095 |
| 29 | after_redecoder | 6 | 0.0533908742 |
| 30 | after_redecoder | 6 | 0.0390009962 |
| 31 | after_redecoder | 6 | 0.0162210048 |
