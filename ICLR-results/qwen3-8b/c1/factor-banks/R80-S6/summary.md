# Qwen3-8B-Base GQA C1 V80 joint fit

- K remains dense; every one of 8 physical V heads retains rank 80/128.
- Total KV-cache retention is 81.25% (18.75% reduction).
- Encoder initialization: `activation-weighted-svd`.
- Checkpoint policy: `fixed decoder-refitted endpoint after encoder sweep 6`.
- V encoders and head output decoders use a full-layer joint solve with cross-head covariance.
- Fit contexts: 256; held-out diagnostic contexts: 64.
- Mean held-out factor-dtype relative MSE: `0.0921038754`.

| Layer | Exported boundary | Sweep | Held-out relative MSE |
|---:|:---|---:|---:|
| 0 | after_redecoder | 6 | 0.0127502289 |
| 1 | after_redecoder | 6 | 0.03192964 |
| 2 | after_redecoder | 6 | 0.0434108043 |
| 3 | after_redecoder | 6 | 0.0422081618 |
| 4 | after_redecoder | 6 | 0.0597691723 |
| 5 | after_redecoder | 6 | 0.0550271853 |
| 6 | after_redecoder | 6 | 0.0768684271 |
| 7 | after_redecoder | 6 | 0.0924074773 |
| 8 | after_redecoder | 6 | 0.113484005 |
| 9 | after_redecoder | 6 | 0.130713209 |
| 10 | after_redecoder | 6 | 0.121140755 |
| 11 | after_redecoder | 6 | 0.11481653 |
| 12 | after_redecoder | 6 | 0.0796864768 |
| 13 | after_redecoder | 6 | 0.0907238454 |
| 14 | after_redecoder | 6 | 0.10812637 |
| 15 | after_redecoder | 6 | 0.102065977 |
| 16 | after_redecoder | 6 | 0.110606057 |
| 17 | after_redecoder | 6 | 0.0776579511 |
| 18 | after_redecoder | 6 | 0.0972845243 |
| 19 | after_redecoder | 6 | 0.0673746065 |
| 20 | after_redecoder | 6 | 0.104643005 |
| 21 | after_redecoder | 6 | 0.11019932 |
| 22 | after_redecoder | 6 | 0.0976829248 |
| 23 | after_redecoder | 6 | 0.103873869 |
| 24 | after_redecoder | 6 | 0.06912706 |
| 25 | after_redecoder | 6 | 0.103746912 |
| 26 | after_redecoder | 6 | 0.137779749 |
| 27 | after_redecoder | 6 | 0.122498676 |
| 28 | after_redecoder | 6 | 0.116502096 |
| 29 | after_redecoder | 6 | 0.137806669 |
| 30 | after_redecoder | 6 | 0.100402063 |
| 31 | after_redecoder | 6 | 0.13562934 |
| 32 | after_redecoder | 6 | 0.102323145 |
| 33 | after_redecoder | 6 | 0.137185493 |
| 34 | after_redecoder | 6 | 0.064483667 |
| 35 | after_redecoder | 6 | 0.0438041193 |
