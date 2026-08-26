# Qwen3-32B GQA C1 V96 joint fit

- K remains dense; every one of 8 physical V heads retains rank 96/128.
- Total KV-cache retention is 87.5% (12.5% reduction).
- Encoder initialization: `activation-weighted-svd`.
- Held-out selection boundaries: `decoder-closed`.
- V encoders and head output decoders use a full-layer joint solve with cross-head covariance.
- Fit contexts: 128; held-out selection contexts: 64.
- Mean held-out factor-dtype relative MSE: `0.101782306`.

| Layer | Selected boundary | Sweep | Held-out relative MSE |
|---:|:---|---:|---:|
| 0 | after_redecoder | 5 | 0.00743356015 |
| 1 | after_redecoder | 5 | 0.0060469357 |
| 2 | after_redecoder | 5 | 0.0180517182 |
| 3 | after_redecoder | 5 | 0.0235056021 |
| 4 | after_redecoder | 5 | 0.0365173702 |
| 5 | after_redecoder | 5 | 0.0267918035 |
| 6 | after_redecoder | 5 | 0.0208638336 |
| 7 | after_redecoder | 5 | 0.0913545866 |
| 8 | after_redecoder | 5 | 0.0658258289 |
| 9 | after_redecoder | 4 | 0.0397634197 |
| 10 | after_redecoder | 5 | 0.0463024248 |
| 11 | after_redecoder | 5 | 0.0980992733 |
| 12 | after_redecoder | 5 | 0.0609550995 |
| 13 | after_redecoder | 5 | 0.0704598945 |
| 14 | after_redecoder | 5 | 0.0786190463 |
| 15 | after_redecoder | 5 | 0.0426643323 |
| 16 | after_redecoder | 5 | 0.0488974422 |
| 17 | after_redecoder | 5 | 0.0285307343 |
| 18 | after_redecoder | 5 | 0.0393903406 |
| 19 | after_redecoder | 5 | 0.0576582635 |
| 20 | after_redecoder | 5 | 0.0720590327 |
| 21 | after_redecoder | 5 | 0.0678586332 |
| 22 | after_redecoder | 5 | 0.0734936573 |
| 23 | after_redecoder | 5 | 0.141649466 |
| 24 | after_redecoder | 5 | 0.12983117 |
| 25 | after_redecoder | 5 | 0.110654748 |
| 26 | after_redecoder | 5 | 0.0940741428 |
| 27 | after_redecoder | 5 | 0.116011037 |
| 28 | after_redecoder | 5 | 0.194985425 |
| 29 | after_redecoder | 5 | 0.115597493 |
| 30 | after_redecoder | 5 | 0.214534343 |
| 31 | after_redecoder | 5 | 0.160017155 |
| 32 | after_redecoder | 5 | 0.081424014 |
| 33 | after_redecoder | 5 | 0.074656382 |
| 34 | after_redecoder | 5 | 0.0919868219 |
| 35 | after_redecoder | 5 | 0.144222327 |
| 36 | after_redecoder | 5 | 0.103947826 |
| 37 | after_redecoder | 5 | 0.122453319 |
| 38 | after_redecoder | 5 | 0.127453901 |
| 39 | after_redecoder | 5 | 0.102980457 |
| 40 | after_redecoder | 5 | 0.0981069814 |
| 41 | after_redecoder | 5 | 0.0712038469 |
| 42 | after_redecoder | 5 | 0.106239519 |
| 43 | after_redecoder | 5 | 0.112225907 |
| 44 | after_redecoder | 5 | 0.127874883 |
| 45 | after_redecoder | 5 | 0.108891434 |
| 46 | after_redecoder | 5 | 0.113591615 |
| 47 | after_redecoder | 5 | 0.141968363 |
| 48 | after_redecoder | 5 | 0.113435218 |
| 49 | after_redecoder | 5 | 0.115365589 |
| 50 | after_redecoder | 5 | 0.123205563 |
| 51 | after_redecoder | 5 | 0.101808956 |
| 52 | after_redecoder | 5 | 0.133123474 |
| 53 | after_redecoder | 5 | 0.106480907 |
| 54 | after_redecoder | 5 | 0.166633872 |
| 55 | after_redecoder | 5 | 0.137121937 |
| 56 | after_redecoder | 5 | 0.152200952 |
| 57 | after_redecoder | 5 | 0.177298754 |
| 58 | after_redecoder | 5 | 0.209082309 |
| 59 | after_redecoder | 5 | 0.157429926 |
| 60 | after_redecoder | 5 | 0.203471895 |
| 61 | after_redecoder | 5 | 0.119415242 |
| 62 | after_redecoder | 5 | 0.324414197 |
| 63 | after_redecoder | 5 | 0.0458533942 |
