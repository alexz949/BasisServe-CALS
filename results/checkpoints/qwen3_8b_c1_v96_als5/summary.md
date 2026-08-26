# Qwen3-8B-Base GQA C1 V96 joint fit

- K remains dense; every one of 8 physical V heads retains rank 96/128.
- Total KV-cache retention is 87.5% (12.5% reduction).
- Encoder initialization: `activation-weighted-svd`.
- Held-out selection boundaries: `decoder-closed`.
- V encoders and head output decoders use a full-layer joint solve with cross-head covariance.
- Fit contexts: 256; held-out selection contexts: 64.
- Mean held-out factor-dtype relative MSE: `0.0553733641`.

| Layer | Selected boundary | Sweep | Held-out relative MSE |
|---:|:---|---:|---:|
| 0 | after_redecoder | 5 | 0.00689953094 |
| 1 | after_redecoder | 5 | 0.0166191079 |
| 2 | after_redecoder | 5 | 0.0241248809 |
| 3 | after_redecoder | 5 | 0.0233624891 |
| 4 | after_redecoder | 5 | 0.0338468501 |
| 5 | after_redecoder | 5 | 0.0309312506 |
| 6 | after_redecoder | 5 | 0.043573909 |
| 7 | after_redecoder | 5 | 0.0539886392 |
| 8 | after_redecoder | 5 | 0.0674744356 |
| 9 | after_redecoder | 5 | 0.0790476083 |
| 10 | after_redecoder | 5 | 0.0727260767 |
| 11 | after_redecoder | 5 | 0.0694349323 |
| 12 | after_redecoder | 5 | 0.047866882 |
| 13 | after_redecoder | 5 | 0.0551599849 |
| 14 | after_redecoder | 5 | 0.0658516075 |
| 15 | after_redecoder | 5 | 0.0623236671 |
| 16 | after_redecoder | 5 | 0.067745291 |
| 17 | after_redecoder | 5 | 0.0474738109 |
| 18 | after_redecoder | 5 | 0.0597883081 |
| 19 | after_redecoder | 5 | 0.0405927024 |
| 20 | after_redecoder | 5 | 0.0643992165 |
| 21 | after_redecoder | 5 | 0.0673260787 |
| 22 | after_redecoder | 5 | 0.059324127 |
| 23 | after_redecoder | 5 | 0.0626737673 |
| 24 | after_redecoder | 5 | 0.0396413082 |
| 25 | after_redecoder | 5 | 0.0631157009 |
| 26 | after_redecoder | 5 | 0.0845199121 |
| 27 | after_redecoder | 5 | 0.0747544606 |
| 28 | after_redecoder | 5 | 0.0712662097 |
| 29 | after_redecoder | 5 | 0.0839849643 |
| 30 | after_redecoder | 5 | 0.0613447711 |
| 31 | after_redecoder | 5 | 0.0815652168 |
| 32 | after_redecoder | 5 | 0.0612424123 |
| 33 | after_redecoder | 5 | 0.0826780993 |
| 34 | after_redecoder | 5 | 0.0394375146 |
| 35 | after_redecoder | 5 | 0.0273353827 |
