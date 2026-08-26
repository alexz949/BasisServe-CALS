# Qwen3-32B GQA C1 V48 joint fit

- K remains dense; every one of 8 physical V heads retains rank 48/128.
- Total KV-cache retention is 68.75% (31.25% reduction).
- Encoder initialization: `activation-weighted-svd`.
- Held-out selection boundaries: `decoder-closed`.
- V encoders and head output decoders use a full-layer joint solve with cross-head covariance.
- Fit contexts: 256; held-out selection contexts: 64.
- Mean held-out factor-dtype relative MSE: `0.198486151`.

| Layer | Selected boundary | Sweep | Held-out relative MSE |
|---:|:---|---:|---:|
| 0 | after_redecoder | 5 | 0.0375909549 |
| 1 | after_redecoder | 5 | 0.0322326678 |
| 2 | after_redecoder | 5 | 0.0705824374 |
| 3 | after_redecoder | 5 | 0.0811159762 |
| 4 | after_redecoder | 5 | 0.0953149876 |
| 5 | after_redecoder | 5 | 0.0722636604 |
| 6 | after_redecoder | 5 | 0.0672322432 |
| 7 | after_redecoder | 5 | 0.201211968 |
| 8 | after_redecoder | 5 | 0.13792079 |
| 9 | after_redecoder | 5 | 0.0989214643 |
| 10 | after_redecoder | 5 | 0.129449502 |
| 11 | after_redecoder | 5 | 0.221868603 |
| 12 | after_redecoder | 5 | 0.138908828 |
| 13 | after_redecoder | 5 | 0.184772113 |
| 14 | after_redecoder | 5 | 0.179135776 |
| 15 | after_redecoder | 5 | 0.128182573 |
| 16 | after_redecoder | 5 | 0.11520433 |
| 17 | after_redecoder | 5 | 0.0825076814 |
| 18 | after_redecoder | 5 | 0.108240144 |
| 19 | after_redecoder | 5 | 0.134976508 |
| 20 | after_redecoder | 5 | 0.138365772 |
| 21 | after_redecoder | 5 | 0.170591714 |
| 22 | after_redecoder | 5 | 0.183289384 |
| 23 | after_redecoder | 5 | 0.193290772 |
| 24 | after_redecoder | 5 | 0.273828427 |
| 25 | after_redecoder | 5 | 0.223053865 |
| 26 | after_redecoder | 5 | 0.221272792 |
| 27 | after_redecoder | 5 | 0.229840911 |
| 28 | after_redecoder | 5 | 0.310322851 |
| 29 | after_redecoder | 5 | 0.233421056 |
| 30 | after_redecoder | 5 | 0.355502413 |
| 31 | after_redecoder | 5 | 0.297318479 |
| 32 | after_redecoder | 5 | 0.177400292 |
| 33 | after_redecoder | 5 | 0.153914342 |
| 34 | after_redecoder | 5 | 0.195031492 |
| 35 | after_redecoder | 5 | 0.279355489 |
| 36 | after_redecoder | 5 | 0.217119107 |
| 37 | after_redecoder | 5 | 0.261854539 |
| 38 | after_redecoder | 5 | 0.254869571 |
| 39 | after_redecoder | 5 | 0.213052536 |
| 40 | after_redecoder | 5 | 0.196526232 |
| 41 | after_redecoder | 5 | 0.15191763 |
| 42 | after_redecoder | 5 | 0.2114915 |
| 43 | after_redecoder | 5 | 0.223702741 |
| 44 | after_redecoder | 5 | 0.242342219 |
| 45 | after_redecoder | 5 | 0.215203973 |
| 46 | after_redecoder | 5 | 0.228664626 |
| 47 | after_redecoder | 5 | 0.26443821 |
| 48 | after_redecoder | 5 | 0.215738721 |
| 49 | after_redecoder | 5 | 0.234644355 |
| 50 | after_redecoder | 5 | 0.238882741 |
| 51 | after_redecoder | 5 | 0.219461112 |
| 52 | after_redecoder | 5 | 0.276038491 |
| 53 | after_redecoder | 5 | 0.211312817 |
| 54 | after_redecoder | 5 | 0.324823611 |
| 55 | after_redecoder | 5 | 0.271503317 |
| 56 | after_redecoder | 5 | 0.266009996 |
| 57 | after_redecoder | 5 | 0.309417065 |
| 58 | after_redecoder | 5 | 0.301343223 |
| 59 | after_redecoder | 5 | 0.28611733 |
| 60 | after_redecoder | 5 | 0.298961974 |
| 61 | after_redecoder | 5 | 0.22131661 |
| 62 | after_redecoder | 5 | 0.307748419 |
| 63 | after_redecoder | 5 | 0.0851777339 |
