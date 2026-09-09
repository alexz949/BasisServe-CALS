# Llama-3.1-70B GQA C1 V112 joint fit

- K remains dense; every one of 8 physical V heads retains rank 112/128.
- Total KV-cache retention is 93.75% (6.25% reduction).
- Encoder initialization: `activation-weighted-svd`.
- Checkpoint policy: `fixed decoder-refitted endpoint after encoder sweep 6`.
- V encoders and head output decoders use a full-layer joint solve with cross-head covariance.
- Fit contexts: 256; held-out diagnostic contexts: 64.
- Mean held-out factor-dtype relative MSE: `0.0477618253`.

| Layer | Exported boundary | Sweep | Held-out relative MSE |
|---:|:---|---:|---:|
| 0 | after_redecoder | 6 | 0.000426473783 |
| 1 | after_redecoder | 6 | 0.00625682175 |
| 2 | after_redecoder | 6 | 0.0108189973 |
| 3 | after_redecoder | 6 | 0.00931477097 |
| 4 | after_redecoder | 6 | 0.00516876012 |
| 5 | after_redecoder | 6 | 0.0146640442 |
| 6 | after_redecoder | 6 | 0.0177838329 |
| 7 | after_redecoder | 6 | 0.0140906042 |
| 8 | after_redecoder | 6 | 0.00904486119 |
| 9 | after_redecoder | 6 | 0.0139951096 |
| 10 | after_redecoder | 6 | 0.020872047 |
| 11 | after_redecoder | 6 | 0.0300769921 |
| 12 | after_redecoder | 6 | 0.0115689722 |
| 13 | after_redecoder | 6 | 0.0158668665 |
| 14 | after_redecoder | 6 | 0.0185902814 |
| 15 | after_redecoder | 6 | 0.0202858523 |
| 16 | after_redecoder | 6 | 0.0190680257 |
| 17 | after_redecoder | 6 | 0.0312765923 |
| 18 | after_redecoder | 6 | 0.0422765173 |
| 19 | after_redecoder | 6 | 0.0353224421 |
| 20 | after_redecoder | 6 | 0.0386559083 |
| 21 | after_redecoder | 6 | 0.0516832453 |
| 22 | after_redecoder | 6 | 0.0699513177 |
| 23 | after_redecoder | 6 | 0.0600124008 |
| 24 | after_redecoder | 6 | 0.0380175647 |
| 25 | after_redecoder | 6 | 0.0529844631 |
| 26 | after_redecoder | 6 | 0.0396788178 |
| 27 | after_redecoder | 6 | 0.0394013854 |
| 28 | after_redecoder | 6 | 0.0362937916 |
| 29 | after_redecoder | 6 | 0.0399781156 |
| 30 | after_redecoder | 6 | 0.0387678515 |
| 31 | after_redecoder | 6 | 0.0387119265 |
| 32 | after_redecoder | 6 | 0.0454690515 |
| 33 | after_redecoder | 6 | 0.0404447839 |
| 34 | after_redecoder | 6 | 0.0377619089 |
| 35 | after_redecoder | 6 | 0.0448652668 |
| 36 | after_redecoder | 6 | 0.0513350031 |
| 37 | after_redecoder | 6 | 0.0467037179 |
| 38 | after_redecoder | 6 | 0.0549187908 |
| 39 | after_redecoder | 6 | 0.0619835637 |
| 40 | after_redecoder | 6 | 0.0565195743 |
| 41 | after_redecoder | 6 | 0.0730825982 |
| 42 | after_redecoder | 6 | 0.0978083678 |
| 43 | after_redecoder | 6 | 0.0636702642 |
| 44 | after_redecoder | 6 | 0.059228669 |
| 45 | after_redecoder | 6 | 0.0670248084 |
| 46 | after_redecoder | 6 | 0.103008547 |
| 47 | after_redecoder | 6 | 0.0690681303 |
| 48 | after_redecoder | 6 | 0.0906001327 |
| 49 | after_redecoder | 6 | 0.0662782894 |
| 50 | after_redecoder | 6 | 0.0900224984 |
| 51 | after_redecoder | 6 | 0.0690418346 |
| 52 | after_redecoder | 6 | 0.0474005414 |
| 53 | after_redecoder | 6 | 0.0939270944 |
| 54 | after_redecoder | 6 | 0.0768399722 |
| 55 | after_redecoder | 6 | 0.0440700218 |
| 56 | after_redecoder | 6 | 0.0727027413 |
| 57 | after_redecoder | 6 | 0.0764462167 |
| 58 | after_redecoder | 6 | 0.0803996967 |
| 59 | after_redecoder | 6 | 0.100452838 |
| 60 | after_redecoder | 6 | 0.0775913845 |
| 61 | after_redecoder | 6 | 0.0958619372 |
| 62 | after_redecoder | 6 | 0.111105645 |
| 63 | after_redecoder | 6 | 0.0672157894 |
| 64 | after_redecoder | 6 | 0.0642440017 |
| 65 | after_redecoder | 6 | 0.0640535208 |
| 66 | after_redecoder | 6 | 0.0485187333 |
| 67 | after_redecoder | 6 | 0.0722600085 |
| 68 | after_redecoder | 6 | 0.0554705481 |
| 69 | after_redecoder | 6 | 0.0588114493 |
| 70 | after_redecoder | 6 | 0.0432303707 |
| 71 | after_redecoder | 6 | 0.0337975462 |
| 72 | after_redecoder | 6 | 0.041058076 |
| 73 | after_redecoder | 6 | 0.0226460381 |
| 74 | after_redecoder | 6 | 0.0204602939 |
| 75 | after_redecoder | 6 | 0.0327252864 |
| 76 | after_redecoder | 6 | 0.0209359226 |
| 77 | after_redecoder | 6 | 0.0373586262 |
| 78 | after_redecoder | 6 | 0.0548647139 |
| 79 | after_redecoder | 6 | 0.0267555568 |
