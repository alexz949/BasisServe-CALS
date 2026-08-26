# Qwen3-32B GQA C1 V64 joint fit

- K remains dense; every one of 8 physical V heads retains rank 64/128.
- Total KV-cache retention is 75% (25% reduction).
- Encoder initialization: `activation-weighted-svd`.
- Held-out selection boundaries: `decoder-closed`.
- V encoders and head output decoders use a full-layer joint solve with cross-head covariance.
- Fit contexts: 256; held-out selection contexts: 64.
- Mean held-out factor-dtype relative MSE: `0.142757912`.

| Layer | Selected boundary | Sweep | Held-out relative MSE |
|---:|:---|---:|---:|
| 0 | decoder_only | 0 | 0.0250672326 |
| 1 | decoder_only | 0 | 0.0190432866 |
| 2 | decoder_only | 0 | 0.0419982185 |
| 3 | decoder_only | 0 | 0.0511182275 |
| 4 | decoder_only | 0 | 0.0639861351 |
| 5 | decoder_only | 0 | 0.0466147978 |
| 6 | decoder_only | 0 | 0.0420188914 |
| 7 | decoder_only | 0 | 0.140617704 |
| 8 | decoder_only | 0 | 0.0955665193 |
| 9 | decoder_only | 0 | 0.0602634722 |
| 10 | decoder_only | 0 | 0.0823942815 |
| 11 | decoder_only | 0 | 0.154742307 |
| 12 | decoder_only | 0 | 0.0897155021 |
| 13 | decoder_only | 0 | 0.124827033 |
| 14 | decoder_only | 0 | 0.119785087 |
| 15 | decoder_only | 0 | 0.0833592394 |
| 16 | decoder_only | 0 | 0.0729017236 |
| 17 | decoder_only | 0 | 0.0512791058 |
| 18 | decoder_only | 0 | 0.0656700424 |
| 19 | decoder_only | 0 | 0.0843600802 |
| 20 | decoder_only | 0 | 0.0865742044 |
| 21 | decoder_only | 0 | 0.113860191 |
| 22 | decoder_only | 0 | 0.124532591 |
| 23 | decoder_only | 0 | 0.130287649 |
| 24 | decoder_only | 0 | 0.19303909 |
| 25 | decoder_only | 0 | 0.155932538 |
| 26 | decoder_only | 0 | 0.154893163 |
| 27 | decoder_only | 0 | 0.160584484 |
| 28 | decoder_only | 0 | 0.223880719 |
| 29 | decoder_only | 0 | 0.167541965 |
| 30 | decoder_only | 0 | 0.261054416 |
| 31 | decoder_only | 0 | 0.219861659 |
| 32 | decoder_only | 0 | 0.127602815 |
| 33 | decoder_only | 0 | 0.1075373 |
| 34 | decoder_only | 0 | 0.13788326 |
| 35 | decoder_only | 0 | 0.203410952 |
| 36 | decoder_only | 0 | 0.158146334 |
| 37 | decoder_only | 0 | 0.192637732 |
| 38 | decoder_only | 0 | 0.191205275 |
| 39 | decoder_only | 0 | 0.159311871 |
| 40 | decoder_only | 0 | 0.147174143 |
| 41 | decoder_only | 0 | 0.112140497 |
| 42 | decoder_only | 0 | 0.157014343 |
| 43 | decoder_only | 0 | 0.167961074 |
| 44 | decoder_only | 0 | 0.181622045 |
| 45 | decoder_only | 0 | 0.162453528 |
| 46 | decoder_only | 0 | 0.17097283 |
| 47 | decoder_only | 0 | 0.198059943 |
| 48 | decoder_only | 0 | 0.161712326 |
| 49 | decoder_only | 0 | 0.17546044 |
| 50 | decoder_only | 0 | 0.177656124 |
| 51 | decoder_only | 0 | 0.158846276 |
| 52 | decoder_only | 0 | 0.198795633 |
| 53 | decoder_only | 0 | 0.153946491 |
| 54 | decoder_only | 0 | 0.239688696 |
| 55 | decoder_only | 0 | 0.20504504 |
| 56 | decoder_only | 0 | 0.202289347 |
| 57 | decoder_only | 0 | 0.234508308 |
| 58 | decoder_only | 0 | 0.226531713 |
| 59 | decoder_only | 0 | 0.214246204 |
| 60 | decoder_only | 0 | 0.231743733 |
| 61 | decoder_only | 0 | 0.163686588 |
| 62 | decoder_only | 0 | 0.243516381 |
| 63 | decoder_only | 0 | 0.0663275878 |
