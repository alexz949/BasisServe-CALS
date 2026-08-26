# Qwen3-32B GQA C1 V96 joint fit

- K remains dense; every one of 8 physical V heads retains rank 96/128.
- Total KV-cache retention is 87.5% (12.5% reduction).
- Encoder initialization: `activation-weighted-svd`.
- Held-out selection boundaries: `decoder-closed`.
- V encoders and head output decoders use a full-layer joint solve with cross-head covariance.
- Fit contexts: 256; held-out selection contexts: 64.
- Mean held-out factor-dtype relative MSE: `0.053275448`.

| Layer | Selected boundary | Sweep | Held-out relative MSE |
|---:|:---|---:|---:|
| 0 | after_redecoder | 5 | 0.00403252612 |
| 1 | after_redecoder | 5 | 0.00313160453 |
| 2 | after_redecoder | 5 | 0.0096366945 |
| 3 | after_redecoder | 5 | 0.0131034089 |
| 4 | after_redecoder | 5 | 0.0185777303 |
| 5 | after_redecoder | 5 | 0.0131557553 |
| 6 | after_redecoder | 5 | 0.0111206458 |
| 7 | after_redecoder | 5 | 0.0463826209 |
| 8 | after_redecoder | 5 | 0.0311070754 |
| 9 | after_redecoder | 5 | 0.0164575481 |
| 10 | after_redecoder | 5 | 0.0241978356 |
| 11 | after_redecoder | 5 | 0.052104719 |
| 12 | after_redecoder | 5 | 0.0279557293 |
| 13 | after_redecoder | 5 | 0.0406041641 |
| 14 | after_redecoder | 5 | 0.0389719013 |
| 15 | after_redecoder | 5 | 0.0246733298 |
| 16 | after_redecoder | 5 | 0.020663458 |
| 17 | after_redecoder | 5 | 0.0139934517 |
| 18 | after_redecoder | 5 | 0.0174083558 |
| 19 | after_redecoder | 5 | 0.0246379942 |
| 20 | after_redecoder | 5 | 0.0258214553 |
| 21 | after_redecoder | 5 | 0.0361996977 |
| 22 | after_redecoder | 5 | 0.0415844119 |
| 23 | after_redecoder | 5 | 0.0421658329 |
| 24 | after_redecoder | 5 | 0.0709884009 |
| 25 | after_redecoder | 5 | 0.0546924511 |
| 26 | after_redecoder | 5 | 0.0540953279 |
| 27 | after_redecoder | 5 | 0.0572692441 |
| 28 | after_redecoder | 5 | 0.0861141544 |
| 29 | after_redecoder | 5 | 0.0639807412 |
| 30 | after_redecoder | 5 | 0.10346584 |
| 31 | after_redecoder | 5 | 0.0857707141 |
| 32 | after_redecoder | 5 | 0.0459170288 |
| 33 | after_redecoder | 5 | 0.0374700595 |
| 34 | after_redecoder | 5 | 0.0502756953 |
| 35 | after_redecoder | 5 | 0.0799794781 |
| 36 | after_redecoder | 5 | 0.0609887292 |
| 37 | after_redecoder | 5 | 0.0746294609 |
| 38 | after_redecoder | 5 | 0.0774633829 |
| 39 | after_redecoder | 5 | 0.0618656428 |
| 40 | after_redecoder | 5 | 0.0571984476 |
| 41 | after_redecoder | 5 | 0.0429975416 |
| 42 | after_redecoder | 5 | 0.0620199738 |
| 43 | after_redecoder | 5 | 0.0669105727 |
| 44 | after_redecoder | 5 | 0.0742349029 |
| 45 | after_redecoder | 5 | 0.0656472194 |
| 46 | after_redecoder | 5 | 0.0680964983 |
| 47 | after_redecoder | 5 | 0.0802546177 |
| 48 | after_redecoder | 5 | 0.0651071464 |
| 49 | after_redecoder | 5 | 0.0695197119 |
| 50 | after_redecoder | 5 | 0.0704357568 |
| 51 | after_redecoder | 5 | 0.0593619585 |
| 52 | after_redecoder | 5 | 0.07672105 |
| 53 | after_redecoder | 5 | 0.0607262819 |
| 54 | after_redecoder | 5 | 0.0921279461 |
| 55 | after_redecoder | 5 | 0.0797336195 |
| 56 | after_redecoder | 5 | 0.0811005392 |
| 57 | after_redecoder | 5 | 0.0969041791 |
| 58 | after_redecoder | 5 | 0.0931626551 |
| 59 | after_redecoder | 5 | 0.085042811 |
| 60 | after_redecoder | 5 | 0.10328653 |
| 61 | after_redecoder | 5 | 0.0657737572 |
| 62 | after_redecoder | 5 | 0.103512226 |
| 63 | after_redecoder | 5 | 0.0270984313 |
