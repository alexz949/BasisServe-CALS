# Qwen3-32B GQA C1 V48 joint fit

- K remains dense; every one of 8 physical V heads retains rank 48/128.
- Total KV-cache retention is 68.75% (31.25% reduction).
- Encoder initialization: `activation-weighted-svd`.
- Checkpoint policy: `fixed decoder-refitted endpoint after encoder sweep 6`.
- V encoders and head output decoders use a full-layer joint solve with cross-head covariance.
- Fit contexts: 256; held-out diagnostic contexts: 64.
- Mean held-out factor-dtype relative MSE: `0.198300153`.

| Layer | Exported boundary | Sweep | Held-out relative MSE |
|---:|:---|---:|---:|
| 0 | after_redecoder | 6 | 0.0373342034 |
| 1 | after_redecoder | 6 | 0.0321089011 |
| 2 | after_redecoder | 6 | 0.0704641974 |
| 3 | after_redecoder | 6 | 0.0810240069 |
| 4 | after_redecoder | 6 | 0.0951418513 |
| 5 | after_redecoder | 6 | 0.0721685833 |
| 6 | after_redecoder | 6 | 0.0671051652 |
| 7 | after_redecoder | 6 | 0.200997838 |
| 8 | after_redecoder | 6 | 0.137772089 |
| 9 | after_redecoder | 6 | 0.0988439442 |
| 10 | after_redecoder | 6 | 0.129353778 |
| 11 | after_redecoder | 6 | 0.221566184 |
| 12 | after_redecoder | 6 | 0.138844397 |
| 13 | after_redecoder | 6 | 0.184623955 |
| 14 | after_redecoder | 6 | 0.17904044 |
| 15 | after_redecoder | 6 | 0.128045826 |
| 16 | after_redecoder | 6 | 0.115100193 |
| 17 | after_redecoder | 6 | 0.0824219402 |
| 18 | after_redecoder | 6 | 0.108175395 |
| 19 | after_redecoder | 6 | 0.134911661 |
| 20 | after_redecoder | 6 | 0.138334211 |
| 21 | after_redecoder | 6 | 0.170443183 |
| 22 | after_redecoder | 6 | 0.183157232 |
| 23 | after_redecoder | 6 | 0.193251501 |
| 24 | after_redecoder | 6 | 0.273717991 |
| 25 | after_redecoder | 6 | 0.222894002 |
| 26 | after_redecoder | 6 | 0.221081647 |
| 27 | after_redecoder | 6 | 0.229714048 |
| 28 | after_redecoder | 6 | 0.31012408 |
| 29 | after_redecoder | 6 | 0.233306538 |
| 30 | after_redecoder | 6 | 0.355347624 |
| 31 | after_redecoder | 6 | 0.297095304 |
| 32 | after_redecoder | 6 | 0.177199219 |
| 33 | after_redecoder | 6 | 0.153761652 |
| 34 | after_redecoder | 6 | 0.194891196 |
| 35 | after_redecoder | 6 | 0.279177129 |
| 36 | after_redecoder | 6 | 0.216976427 |
| 37 | after_redecoder | 6 | 0.261624818 |
| 38 | after_redecoder | 6 | 0.254639126 |
| 39 | after_redecoder | 6 | 0.212763882 |
| 40 | after_redecoder | 6 | 0.196219974 |
| 41 | after_redecoder | 6 | 0.151670502 |
| 42 | after_redecoder | 6 | 0.211223071 |
| 43 | after_redecoder | 6 | 0.223455492 |
| 44 | after_redecoder | 6 | 0.242136066 |
| 45 | after_redecoder | 6 | 0.214941127 |
| 46 | after_redecoder | 6 | 0.22842233 |
| 47 | after_redecoder | 6 | 0.264217407 |
| 48 | after_redecoder | 6 | 0.215535622 |
| 49 | after_redecoder | 6 | 0.234357073 |
| 50 | after_redecoder | 6 | 0.238643923 |
| 51 | after_redecoder | 6 | 0.219152629 |
| 52 | after_redecoder | 6 | 0.275760318 |
| 53 | after_redecoder | 6 | 0.211116862 |
| 54 | after_redecoder | 6 | 0.324484026 |
| 55 | after_redecoder | 6 | 0.271093626 |
| 56 | after_redecoder | 6 | 0.265718892 |
| 57 | after_redecoder | 6 | 0.309065187 |
| 58 | after_redecoder | 6 | 0.301244214 |
| 59 | after_redecoder | 6 | 0.285641362 |
| 60 | after_redecoder | 6 | 0.298974198 |
| 61 | after_redecoder | 6 | 0.221086494 |
| 62 | after_redecoder | 6 | 0.307514657 |
| 63 | after_redecoder | 6 | 0.0849893568 |
