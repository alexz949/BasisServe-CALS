# Qwen3-32B GQA C1 V96 joint fit

- K remains dense; every one of 8 physical V heads retains rank 96/128.
- Total KV-cache retention is 87.5% (12.5% reduction).
- Encoder initialization: `activation-weighted-svd`.
- Checkpoint policy: `fixed decoder-refitted endpoint after encoder sweep 6`.
- V encoders and head output decoders use a full-layer joint solve with cross-head covariance.
- Fit contexts: 256; held-out diagnostic contexts: 64.
- Mean held-out factor-dtype relative MSE: `0.0531781892`.

| Layer | Exported boundary | Sweep | Held-out relative MSE |
|---:|:---|---:|---:|
| 0 | after_redecoder | 6 | 0.00398801134 |
| 1 | after_redecoder | 6 | 0.00312355005 |
| 2 | after_redecoder | 6 | 0.00962065627 |
| 3 | after_redecoder | 6 | 0.0130784205 |
| 4 | after_redecoder | 6 | 0.0185262798 |
| 5 | after_redecoder | 6 | 0.0131342396 |
| 6 | after_redecoder | 6 | 0.0111035933 |
| 7 | after_redecoder | 6 | 0.046306231 |
| 8 | after_redecoder | 6 | 0.0310476601 |
| 9 | after_redecoder | 6 | 0.0164417824 |
| 10 | after_redecoder | 6 | 0.0241811863 |
| 11 | after_redecoder | 6 | 0.0520016902 |
| 12 | after_redecoder | 6 | 0.0279459165 |
| 13 | after_redecoder | 6 | 0.0405615451 |
| 14 | after_redecoder | 6 | 0.0389380249 |
| 15 | after_redecoder | 6 | 0.0246445684 |
| 16 | after_redecoder | 6 | 0.0206462826 |
| 17 | after_redecoder | 6 | 0.013966925 |
| 18 | after_redecoder | 6 | 0.0173973347 |
| 19 | after_redecoder | 6 | 0.0246262014 |
| 20 | after_redecoder | 6 | 0.0258077821 |
| 21 | after_redecoder | 6 | 0.0361650171 |
| 22 | after_redecoder | 6 | 0.041544287 |
| 23 | after_redecoder | 6 | 0.0421249809 |
| 24 | after_redecoder | 6 | 0.070919132 |
| 25 | after_redecoder | 6 | 0.0546377224 |
| 26 | after_redecoder | 6 | 0.054001784 |
| 27 | after_redecoder | 6 | 0.0571966371 |
| 28 | after_redecoder | 6 | 0.0860007553 |
| 29 | after_redecoder | 6 | 0.063931386 |
| 30 | after_redecoder | 6 | 0.103410756 |
| 31 | after_redecoder | 6 | 0.085653298 |
| 32 | after_redecoder | 6 | 0.0458520649 |
| 33 | after_redecoder | 6 | 0.0374168709 |
| 34 | after_redecoder | 6 | 0.0502159269 |
| 35 | after_redecoder | 6 | 0.0798805596 |
| 36 | after_redecoder | 6 | 0.0609111499 |
| 37 | after_redecoder | 6 | 0.0744935104 |
| 38 | after_redecoder | 6 | 0.0772930996 |
| 39 | after_redecoder | 6 | 0.061711004 |
| 40 | after_redecoder | 6 | 0.0570283986 |
| 41 | after_redecoder | 6 | 0.0428747462 |
| 42 | after_redecoder | 6 | 0.0618656178 |
| 43 | after_redecoder | 6 | 0.0667751047 |
| 44 | after_redecoder | 6 | 0.0740680663 |
| 45 | after_redecoder | 6 | 0.0654839908 |
| 46 | after_redecoder | 6 | 0.0679249247 |
| 47 | after_redecoder | 6 | 0.0801260309 |
| 48 | after_redecoder | 6 | 0.0649811892 |
| 49 | after_redecoder | 6 | 0.0693463139 |
| 50 | after_redecoder | 6 | 0.070269481 |
| 51 | after_redecoder | 6 | 0.0592262072 |
| 52 | after_redecoder | 6 | 0.0765747632 |
| 53 | after_redecoder | 6 | 0.0606178389 |
| 54 | after_redecoder | 6 | 0.0919585274 |
| 55 | after_redecoder | 6 | 0.0795254314 |
| 56 | after_redecoder | 6 | 0.0808950247 |
| 57 | after_redecoder | 6 | 0.0967236221 |
| 58 | after_redecoder | 6 | 0.0930233816 |
| 59 | after_redecoder | 6 | 0.0848278549 |
| 60 | after_redecoder | 6 | 0.103033347 |
| 61 | after_redecoder | 6 | 0.0656971253 |
| 62 | after_redecoder | 6 | 0.103120038 |
| 63 | after_redecoder | 6 | 0.0269892575 |
