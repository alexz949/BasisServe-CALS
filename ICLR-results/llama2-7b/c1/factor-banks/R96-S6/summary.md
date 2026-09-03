# Llama-2-7B MHA C1 V96 joint fit

- K remains dense; every one of 32 physical V heads retains rank 96/128.
- Total KV-cache retention is 87.5% (12.5% reduction).
- Encoder initialization: `activation-weighted-svd`.
- Checkpoint policy: `fixed decoder-refitted endpoint after encoder sweep 6`.
- V encoders and head output decoders use a full-layer joint solve with cross-head covariance.
- Fit contexts: 256; held-out diagnostic contexts: 64.
- Mean held-out factor-dtype relative MSE: `0.0728255983`.

| Layer | Exported boundary | Sweep | Held-out relative MSE |
|---:|:---|---:|---:|
| 0 | after_redecoder | 6 | 5.22678183e-05 |
| 1 | after_redecoder | 6 | 0.015702733 |
| 2 | after_redecoder | 6 | 0.0380844088 |
| 3 | after_redecoder | 6 | 0.0291070757 |
| 4 | after_redecoder | 6 | 0.0421089461 |
| 5 | after_redecoder | 6 | 0.0399700775 |
| 6 | after_redecoder | 6 | 0.0479195521 |
| 7 | after_redecoder | 6 | 0.0566315273 |
| 8 | after_redecoder | 6 | 0.0646008589 |
| 9 | after_redecoder | 6 | 0.0724962287 |
| 10 | after_redecoder | 6 | 0.072433549 |
| 11 | after_redecoder | 6 | 0.0771297167 |
| 12 | after_redecoder | 6 | 0.079706955 |
| 13 | after_redecoder | 6 | 0.0796530507 |
| 14 | after_redecoder | 6 | 0.0875380698 |
| 15 | after_redecoder | 6 | 0.0741413904 |
| 16 | after_redecoder | 6 | 0.0714658085 |
| 17 | after_redecoder | 6 | 0.0923556663 |
| 18 | after_redecoder | 6 | 0.0883417744 |
| 19 | after_redecoder | 6 | 0.092448318 |
| 20 | after_redecoder | 6 | 0.0749542095 |
| 21 | after_redecoder | 6 | 0.107573796 |
| 22 | after_redecoder | 6 | 0.0811432976 |
| 23 | after_redecoder | 6 | 0.11520346 |
| 24 | after_redecoder | 6 | 0.0871207145 |
| 25 | after_redecoder | 6 | 0.127756062 |
| 26 | after_redecoder | 6 | 0.0788570539 |
| 27 | after_redecoder | 6 | 0.102731011 |
| 28 | after_redecoder | 6 | 0.105166151 |
| 29 | after_redecoder | 6 | 0.101106285 |
| 30 | after_redecoder | 6 | 0.0864119588 |
| 31 | after_redecoder | 6 | 0.0405071698 |
