# DeepSeek-V2-Lite TP8 post-attention C1

- MLA and KV cache are unchanged.
- Source width/rank: `256/64`.
- Communication reduction versus dense AllGather: `75%`.

| Layer | Boundary | Sweep | Heldout relative MSE |
|---:|:---|---:|---:|
| 0 | after_redecoder | 5 | 0.0108005018 |
| 1 | after_redecoder | 5 | 0.0915854289 |
| 2 | after_redecoder | 5 | 0.15853544 |
| 3 | after_redecoder | 5 | 0.180085159 |
| 4 | after_redecoder | 5 | 0.173216827 |
| 5 | after_redecoder | 5 | 0.199632779 |
| 6 | after_redecoder | 5 | 0.205860943 |
| 7 | after_redecoder | 5 | 0.187870921 |
| 8 | after_redecoder | 5 | 0.24983295 |
| 9 | after_redecoder | 5 | 0.284925196 |
| 10 | after_redecoder | 5 | 0.26761282 |
| 11 | after_redecoder | 5 | 0.226789012 |
| 12 | after_redecoder | 5 | 0.255772118 |
| 13 | after_redecoder | 5 | 0.278760834 |
| 14 | after_redecoder | 5 | 0.19400589 |
| 15 | after_redecoder | 5 | 0.23650587 |
| 16 | after_redecoder | 5 | 0.237355446 |
| 17 | after_redecoder | 5 | 0.263724116 |
| 18 | after_redecoder | 5 | 0.110079103 |
| 19 | after_redecoder | 5 | 0.376867348 |
| 20 | after_redecoder | 5 | 0.161595859 |
| 21 | after_redecoder | 5 | 0.0936401484 |
| 22 | after_redecoder | 5 | 0.260837201 |
| 23 | after_redecoder | 5 | 0.186941914 |
| 24 | after_redecoder | 5 | 0.385391587 |
| 25 | after_redecoder | 5 | 0.0993038183 |
| 26 | after_redecoder | 5 | 0.0526282004 |
