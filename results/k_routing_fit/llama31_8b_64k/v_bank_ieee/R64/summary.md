# Llama-3.1-8B GQA C1 V64 joint fit

- K remains dense; every one of 8 physical V heads retains rank 64/128.
- Total KV-cache retention is 75% (25% reduction).
- Encoder initialization: `activation-weighted-svd`.
- Checkpoint policy: `fixed decoder-refitted endpoint after encoder sweep 6`.
- V encoders and head output decoders use a full-layer joint solve with cross-head covariance.
- Fit contexts: 64; held-out diagnostic contexts: 16.
- Mean held-out factor-dtype relative MSE: `0.141883764`.

| Layer | Exported boundary | Sweep | Held-out relative MSE |
|---:|:---|---:|---:|
| 0 | after_redecoder | 6 | 0.0323759778 |
| 1 | after_redecoder | 6 | 0.0595039095 |
| 2 | after_redecoder | 6 | 0.0562972849 |
| 3 | after_redecoder | 6 | 0.099238803 |
| 4 | after_redecoder | 6 | 0.102938241 |
| 5 | after_redecoder | 6 | 0.108352842 |
| 6 | after_redecoder | 6 | 0.137524994 |
| 7 | after_redecoder | 6 | 0.117169241 |
| 8 | after_redecoder | 6 | 0.141921565 |
| 9 | after_redecoder | 6 | 0.139868732 |
| 10 | after_redecoder | 6 | 0.153812281 |
| 11 | after_redecoder | 6 | 0.139113716 |
| 12 | after_redecoder | 6 | 0.147130865 |
| 13 | after_redecoder | 6 | 0.153050605 |
| 14 | after_redecoder | 6 | 0.150048362 |
| 15 | after_redecoder | 6 | 0.1918554 |
| 16 | after_redecoder | 6 | 0.161777955 |
| 17 | after_redecoder | 6 | 0.178283693 |
| 18 | after_redecoder | 6 | 0.174189833 |
| 19 | after_redecoder | 6 | 0.185559441 |
| 20 | after_redecoder | 6 | 0.209148757 |
| 21 | after_redecoder | 6 | 0.166837044 |
| 22 | after_redecoder | 6 | 0.224670525 |
| 23 | after_redecoder | 6 | 0.220245085 |
| 24 | after_redecoder | 6 | 0.206898712 |
| 25 | after_redecoder | 6 | 0.185113739 |
| 26 | after_redecoder | 6 | 0.135700621 |
| 27 | after_redecoder | 6 | 0.162004017 |
| 28 | after_redecoder | 6 | 0.146851738 |
| 29 | after_redecoder | 6 | 0.122917856 |
| 30 | after_redecoder | 6 | 0.0884337283 |
| 31 | after_redecoder | 6 | 0.0414448733 |
