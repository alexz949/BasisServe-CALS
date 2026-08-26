# Qwen3-32B GQA C1 V64 joint fit

- K remains dense; every one of 8 physical V heads retains rank 64/128.
- Total KV-cache retention is 75% (25% reduction).
- Encoder initialization: `activation-weighted-svd`.
- Held-out selection boundaries: `decoder-closed`.
- V encoders and head output decoders use a full-layer joint solve with cross-head covariance.
- Fit contexts: 128; held-out selection contexts: 64.
- Mean held-out factor-dtype relative MSE: `0.207225978`.

| Layer | Selected boundary | Sweep | Held-out relative MSE |
|---:|:---|---:|---:|
| 0 | after_redecoder | 5 | 0.0297683123 |
| 1 | after_redecoder | 5 | 0.0248369502 |
| 2 | after_redecoder | 5 | 0.0586656541 |
| 3 | after_redecoder | 5 | 0.0708100438 |
| 4 | after_redecoder | 5 | 0.0935711727 |
| 5 | after_redecoder | 5 | 0.0693133938 |
| 6 | after_redecoder | 5 | 0.0596290809 |
| 7 | after_redecoder | 5 | 0.20444832 |
| 8 | after_redecoder | 5 | 0.139874785 |
| 9 | after_redecoder | 5 | 0.10219866 |
| 10 | after_redecoder | 5 | 0.119203471 |
| 11 | after_redecoder | 5 | 0.217888962 |
| 12 | after_redecoder | 5 | 0.140563231 |
| 13 | after_redecoder | 5 | 0.169532995 |
| 14 | after_redecoder | 5 | 0.197707559 |
| 15 | after_redecoder | 5 | 0.112004713 |
| 16 | after_redecoder | 4 | 0.110067983 |
| 17 | after_redecoder | 5 | 0.0785940042 |
| 18 | after_redecoder | 4 | 0.103934692 |
| 19 | after_redecoder | 5 | 0.147037521 |
| 20 | after_redecoder | 5 | 0.168370766 |
| 21 | after_redecoder | 5 | 0.163527641 |
| 22 | after_redecoder | 5 | 0.177185358 |
| 23 | after_redecoder | 5 | 0.318401017 |
| 24 | after_redecoder | 4 | 0.275437127 |
| 25 | after_redecoder | 5 | 0.232799978 |
| 26 | after_redecoder | 5 | 0.215246073 |
| 27 | after_redecoder | 5 | 0.248303015 |
| 28 | after_redecoder | 5 | 0.361087624 |
| 29 | after_redecoder | 5 | 0.236644374 |
| 30 | after_redecoder | 3 | 0.393697725 |
| 31 | after_redecoder | 5 | 0.313368065 |
| 32 | after_redecoder | 5 | 0.172593968 |
| 33 | after_redecoder | 5 | 0.170561655 |
| 34 | after_redecoder | 5 | 0.195356839 |
| 35 | after_redecoder | 5 | 0.288066787 |
| 36 | after_redecoder | 5 | 0.215882104 |
| 37 | after_redecoder | 5 | 0.255689153 |
| 38 | after_redecoder | 5 | 0.256080577 |
| 39 | after_redecoder | 5 | 0.209988587 |
| 40 | after_redecoder | 5 | 0.198843655 |
| 41 | after_redecoder | 5 | 0.149085763 |
| 42 | after_redecoder | 5 | 0.212877748 |
| 43 | after_redecoder | 5 | 0.224990439 |
| 44 | after_redecoder | 5 | 0.250343467 |
| 45 | after_redecoder | 5 | 0.215538154 |
| 46 | after_redecoder | 5 | 0.229299887 |
| 47 | after_redecoder | 5 | 0.275929492 |
| 48 | after_redecoder | 5 | 0.222566441 |
| 49 | after_redecoder | 5 | 0.232336663 |
| 50 | after_redecoder | 5 | 0.244814784 |
| 51 | after_redecoder | 5 | 0.21328074 |
| 52 | after_redecoder | 5 | 0.27500846 |
| 53 | after_redecoder | 5 | 0.215783735 |
| 54 | after_redecoder | 5 | 0.337211851 |
| 55 | after_redecoder | 5 | 0.272634998 |
| 56 | after_redecoder | 5 | 0.283725194 |
| 57 | after_redecoder | 5 | 0.334480345 |
| 58 | after_redecoder | 5 | 0.358873447 |
| 59 | after_redecoder | 5 | 0.30965981 |
| 60 | after_redecoder | 5 | 0.336398802 |
| 61 | after_redecoder | 5 | 0.228307366 |
| 62 | after_redecoder | 3 | 0.437699939 |
| 63 | after_redecoder | 5 | 0.084831445 |
