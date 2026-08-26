# Qwen3-32B GQA C1 V96 joint fit

- K remains dense; every one of 8 physical V heads retains rank 96/128.
- Total KV-cache retention is 87.5% (12.5% reduction).
- Encoder initialization: `activation-weighted-svd`.
- Held-out selection boundaries: `decoder-closed`.
- V encoders and head output decoders use a full-layer joint solve with cross-head covariance.
- Fit contexts: 256; held-out selection contexts: 64.
- Mean held-out factor-dtype relative MSE: `0.0555481453`.

| Layer | Selected boundary | Sweep | Held-out relative MSE |
|---:|:---|---:|---:|
| 0 | decoder_only | 0 | 0.00499766938 |
| 1 | decoder_only | 0 | 0.00340420898 |
| 2 | decoder_only | 0 | 0.0101769614 |
| 3 | decoder_only | 0 | 0.013988992 |
| 4 | decoder_only | 0 | 0.0200123921 |
| 5 | decoder_only | 0 | 0.0137854068 |
| 6 | decoder_only | 0 | 0.0116492014 |
| 7 | decoder_only | 0 | 0.04854098 |
| 8 | decoder_only | 0 | 0.0329475612 |
| 9 | decoder_only | 0 | 0.0169355433 |
| 10 | decoder_only | 0 | 0.0249650965 |
| 11 | decoder_only | 0 | 0.054912852 |
| 12 | decoder_only | 0 | 0.0285689344 |
| 13 | decoder_only | 0 | 0.0418375316 |
| 14 | decoder_only | 0 | 0.0400631092 |
| 15 | decoder_only | 0 | 0.0257204729 |
| 16 | decoder_only | 0 | 0.0213702838 |
| 17 | decoder_only | 0 | 0.0146361848 |
| 18 | decoder_only | 0 | 0.0179080173 |
| 19 | decoder_only | 0 | 0.0252072936 |
| 20 | decoder_only | 0 | 0.0263498778 |
| 21 | decoder_only | 0 | 0.0374914409 |
| 22 | decoder_only | 0 | 0.0429122362 |
| 23 | decoder_only | 0 | 0.0438852522 |
| 24 | decoder_only | 0 | 0.0727687683 |
| 25 | decoder_only | 0 | 0.0565255621 |
| 26 | decoder_only | 0 | 0.0565131175 |
| 27 | decoder_only | 0 | 0.0592929417 |
| 28 | decoder_only | 0 | 0.0888008787 |
| 29 | decoder_only | 0 | 0.0657433381 |
| 30 | decoder_only | 0 | 0.105890257 |
| 31 | decoder_only | 0 | 0.08921039 |
| 32 | decoder_only | 0 | 0.0478830184 |
| 33 | decoder_only | 0 | 0.0390295164 |
| 34 | decoder_only | 0 | 0.0521316354 |
| 35 | decoder_only | 0 | 0.0825563197 |
| 36 | decoder_only | 0 | 0.0632022815 |
| 37 | decoder_only | 0 | 0.0775320945 |
| 38 | decoder_only | 0 | 0.0806603577 |
| 39 | decoder_only | 0 | 0.0654127567 |
| 40 | decoder_only | 0 | 0.0609665706 |
| 41 | decoder_only | 0 | 0.0453743258 |
| 42 | decoder_only | 0 | 0.065254971 |
| 43 | decoder_only | 0 | 0.0702185321 |
| 44 | decoder_only | 0 | 0.0775967537 |
| 45 | decoder_only | 0 | 0.0690552167 |
| 46 | decoder_only | 0 | 0.0714785706 |
| 47 | decoder_only | 0 | 0.0839094157 |
| 48 | decoder_only | 0 | 0.0680263987 |
| 49 | decoder_only | 0 | 0.0729358328 |
| 50 | decoder_only | 0 | 0.073845639 |
| 51 | decoder_only | 0 | 0.062720639 |
| 52 | decoder_only | 0 | 0.0798993603 |
| 53 | decoder_only | 0 | 0.063120835 |
| 54 | decoder_only | 0 | 0.096442687 |
| 55 | decoder_only | 0 | 0.084084634 |
| 56 | decoder_only | 0 | 0.0853811969 |
| 57 | decoder_only | 0 | 0.100704295 |
| 58 | decoder_only | 0 | 0.0958018246 |
| 59 | decoder_only | 0 | 0.0886208765 |
| 60 | decoder_only | 0 | 0.107284028 |
| 61 | decoder_only | 0 | 0.0674643691 |
| 62 | decoder_only | 0 | 0.110565977 |
| 63 | decoder_only | 0 | 0.028907615 |
