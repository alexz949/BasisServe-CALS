# Qwen3-32B GQA C1 V80 joint fit

- K remains dense; every one of 8 physical V heads retains rank 80/128.
- Total KV-cache retention is 81.25% (18.75% reduction).
- Encoder initialization: `activation-weighted-svd`.
- Checkpoint policy: `fixed decoder-refitted endpoint after encoder sweep 6`.
- V encoders and head output decoders use a full-layer joint solve with cross-head covariance.
- Fit contexts: 256; held-out diagnostic contexts: 64.
- Mean held-out factor-dtype relative MSE: `0.0906974766`.

| Layer | Exported boundary | Sweep | Held-out relative MSE |
|---:|:---|---:|---:|
| 0 | after_redecoder | 6 | 0.009519661 |
| 1 | after_redecoder | 6 | 0.00775113245 |
| 2 | after_redecoder | 6 | 0.0208825804 |
| 3 | after_redecoder | 6 | 0.0270358759 |
| 4 | after_redecoder | 6 | 0.0354457861 |
| 5 | after_redecoder | 6 | 0.0258906372 |
| 6 | after_redecoder | 6 | 0.0225313978 |
| 7 | after_redecoder | 6 | 0.0836942826 |
| 8 | after_redecoder | 6 | 0.0564844949 |
| 9 | after_redecoder | 6 | 0.0328622227 |
| 10 | after_redecoder | 6 | 0.0468120534 |
| 11 | after_redecoder | 6 | 0.0929166491 |
| 12 | after_redecoder | 6 | 0.0525661394 |
| 13 | after_redecoder | 6 | 0.0747232347 |
| 14 | after_redecoder | 6 | 0.0718917798 |
| 15 | after_redecoder | 6 | 0.047069837 |
| 16 | after_redecoder | 6 | 0.0404650237 |
| 17 | after_redecoder | 6 | 0.0276427997 |
| 18 | after_redecoder | 6 | 0.0355534342 |
| 19 | after_redecoder | 6 | 0.0478165262 |
| 20 | after_redecoder | 6 | 0.0495583149 |
| 21 | after_redecoder | 6 | 0.0667858017 |
| 22 | after_redecoder | 6 | 0.0751477893 |
| 23 | after_redecoder | 6 | 0.0774651304 |
| 24 | after_redecoder | 6 | 0.123399666 |
| 25 | after_redecoder | 6 | 0.0969453914 |
| 26 | after_redecoder | 6 | 0.0951782157 |
| 27 | after_redecoder | 6 | 0.100625555 |
| 28 | after_redecoder | 6 | 0.145481161 |
| 29 | after_redecoder | 6 | 0.108430173 |
| 30 | after_redecoder | 6 | 0.171933851 |
| 31 | after_redecoder | 6 | 0.143634326 |
| 32 | after_redecoder | 6 | 0.0798295558 |
| 33 | after_redecoder | 6 | 0.0660111195 |
| 34 | after_redecoder | 6 | 0.0868584053 |
| 35 | after_redecoder | 6 | 0.133182169 |
| 36 | after_redecoder | 6 | 0.102485951 |
| 37 | after_redecoder | 6 | 0.125426219 |
| 38 | after_redecoder | 6 | 0.126688781 |
| 39 | after_redecoder | 6 | 0.102567906 |
| 40 | after_redecoder | 6 | 0.0942650057 |
| 41 | after_redecoder | 6 | 0.0713670612 |
| 42 | after_redecoder | 6 | 0.102138452 |
| 43 | after_redecoder | 6 | 0.109405636 |
| 44 | after_redecoder | 6 | 0.120789126 |
| 45 | after_redecoder | 6 | 0.106989322 |
| 46 | after_redecoder | 6 | 0.111903212 |
| 47 | after_redecoder | 6 | 0.130931306 |
| 48 | after_redecoder | 6 | 0.10677266 |
| 49 | after_redecoder | 6 | 0.114948445 |
| 50 | after_redecoder | 6 | 0.115778089 |
| 51 | after_redecoder | 6 | 0.0999099419 |
| 52 | after_redecoder | 6 | 0.128220314 |
| 53 | after_redecoder | 6 | 0.100545588 |
| 54 | after_redecoder | 6 | 0.154930811 |
| 55 | after_redecoder | 6 | 0.132355013 |
| 56 | after_redecoder | 6 | 0.134008651 |
| 57 | after_redecoder | 6 | 0.157031784 |
| 58 | after_redecoder | 6 | 0.152066138 |
| 59 | after_redecoder | 6 | 0.140909722 |
| 60 | after_redecoder | 6 | 0.163287385 |
| 61 | after_redecoder | 6 | 0.109314414 |
| 62 | after_redecoder | 6 | 0.166012702 |
| 63 | after_redecoder | 6 | 0.0435666927 |
