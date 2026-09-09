# Llama-3.1-70B GQA C1 V96 joint fit

- K remains dense; every one of 8 physical V heads retains rank 96/128.
- Total KV-cache retention is 87.5% (12.5% reduction).
- Encoder initialization: `activation-weighted-svd`.
- Checkpoint policy: `fixed decoder-refitted endpoint after encoder sweep 6`.
- V encoders and head output decoders use a full-layer joint solve with cross-head covariance.
- Fit contexts: 256; held-out diagnostic contexts: 64.
- Mean held-out factor-dtype relative MSE: `0.0995887053`.

| Layer | Exported boundary | Sweep | Held-out relative MSE |
|---:|:---|---:|---:|
| 0 | after_redecoder | 6 | 0.00114040629 |
| 1 | after_redecoder | 6 | 0.0166641485 |
| 2 | after_redecoder | 6 | 0.0276009107 |
| 3 | after_redecoder | 6 | 0.0225704321 |
| 4 | after_redecoder | 6 | 0.0127154143 |
| 5 | after_redecoder | 6 | 0.0349568131 |
| 6 | after_redecoder | 6 | 0.0421020881 |
| 7 | after_redecoder | 6 | 0.034106007 |
| 8 | after_redecoder | 6 | 0.0216416584 |
| 9 | after_redecoder | 6 | 0.0333578718 |
| 10 | after_redecoder | 6 | 0.0481472966 |
| 11 | after_redecoder | 6 | 0.0684494914 |
| 12 | after_redecoder | 6 | 0.0273768321 |
| 13 | after_redecoder | 6 | 0.0363417075 |
| 14 | after_redecoder | 6 | 0.0422545114 |
| 15 | after_redecoder | 6 | 0.0458678039 |
| 16 | after_redecoder | 6 | 0.0424010676 |
| 17 | after_redecoder | 6 | 0.0688226451 |
| 18 | after_redecoder | 6 | 0.0922103446 |
| 19 | after_redecoder | 6 | 0.0775823962 |
| 20 | after_redecoder | 6 | 0.0824704409 |
| 21 | after_redecoder | 6 | 0.109911053 |
| 22 | after_redecoder | 6 | 0.142859901 |
| 23 | after_redecoder | 6 | 0.12482189 |
| 24 | after_redecoder | 6 | 0.0796212945 |
| 25 | after_redecoder | 6 | 0.109427201 |
| 26 | after_redecoder | 6 | 0.0827789417 |
| 27 | after_redecoder | 6 | 0.0838867848 |
| 28 | after_redecoder | 6 | 0.0762562634 |
| 29 | after_redecoder | 6 | 0.0826700245 |
| 30 | after_redecoder | 6 | 0.0807871585 |
| 31 | after_redecoder | 6 | 0.0826477178 |
| 32 | after_redecoder | 6 | 0.0940351097 |
| 33 | after_redecoder | 6 | 0.0864701022 |
| 34 | after_redecoder | 6 | 0.0804274963 |
| 35 | after_redecoder | 6 | 0.0959430337 |
| 36 | after_redecoder | 6 | 0.107520418 |
| 37 | after_redecoder | 6 | 0.100810166 |
| 38 | after_redecoder | 6 | 0.114633023 |
| 39 | after_redecoder | 6 | 0.131942173 |
| 40 | after_redecoder | 6 | 0.119372866 |
| 41 | after_redecoder | 6 | 0.152404134 |
| 42 | after_redecoder | 6 | 0.187979932 |
| 43 | after_redecoder | 6 | 0.128066206 |
| 44 | after_redecoder | 6 | 0.126563928 |
| 45 | after_redecoder | 6 | 0.139112305 |
| 46 | after_redecoder | 6 | 0.211417815 |
| 47 | after_redecoder | 6 | 0.140663921 |
| 48 | after_redecoder | 6 | 0.1821709 |
| 49 | after_redecoder | 6 | 0.137645128 |
| 50 | after_redecoder | 6 | 0.185632565 |
| 51 | after_redecoder | 6 | 0.145326176 |
| 52 | after_redecoder | 6 | 0.103463473 |
| 53 | after_redecoder | 6 | 0.193290858 |
| 54 | after_redecoder | 6 | 0.155050873 |
| 55 | after_redecoder | 6 | 0.0906034929 |
| 56 | after_redecoder | 6 | 0.153818668 |
| 57 | after_redecoder | 6 | 0.157539744 |
| 58 | after_redecoder | 6 | 0.162826219 |
| 59 | after_redecoder | 6 | 0.204423165 |
| 60 | after_redecoder | 6 | 0.165258924 |
| 61 | after_redecoder | 6 | 0.199188153 |
| 62 | after_redecoder | 6 | 0.210921424 |
| 63 | after_redecoder | 6 | 0.138703957 |
| 64 | after_redecoder | 6 | 0.13447464 |
| 65 | after_redecoder | 6 | 0.131955471 |
| 66 | after_redecoder | 6 | 0.102529174 |
| 67 | after_redecoder | 6 | 0.144015754 |
| 68 | after_redecoder | 6 | 0.117301876 |
| 69 | after_redecoder | 6 | 0.121690267 |
| 70 | after_redecoder | 6 | 0.0891069071 |
| 71 | after_redecoder | 6 | 0.0719863138 |
| 72 | after_redecoder | 6 | 0.0703409088 |
| 73 | after_redecoder | 6 | 0.049164279 |
| 74 | after_redecoder | 6 | 0.0434534126 |
| 75 | after_redecoder | 6 | 0.0680281788 |
| 76 | after_redecoder | 6 | 0.0450845416 |
| 77 | after_redecoder | 6 | 0.0778013488 |
| 78 | after_redecoder | 6 | 0.105593648 |
| 79 | after_redecoder | 6 | 0.0568948329 |
