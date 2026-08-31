# DeepSeek-V2-Lite TP8 post-attention C1

- MLA and KV cache are unchanged.
- Source width/rank: `256/128`.
- Communication reduction versus dense AllGather: `50%`.

| Layer | Boundary | Sweep | Heldout relative MSE |
|---:|:---|---:|---:|
| 0 | after_redecoder | 5 | 0.00160243943 |
| 1 | after_redecoder | 5 | 0.0147245481 |
| 2 | after_redecoder | 5 | 0.0430969416 |
| 3 | after_redecoder | 5 | 0.0543467319 |
| 4 | after_redecoder | 5 | 0.0594328433 |
| 5 | after_redecoder | 5 | 0.0657705944 |
| 6 | after_redecoder | 5 | 0.0651006514 |
| 7 | after_redecoder | 5 | 0.0648309546 |
| 8 | after_redecoder | 5 | 0.0969472385 |
| 9 | after_redecoder | 5 | 0.111415939 |
| 10 | after_redecoder | 5 | 0.109140374 |
| 11 | after_redecoder | 5 | 0.0753871862 |
| 12 | after_redecoder | 5 | 0.0935408345 |
| 13 | after_redecoder | 5 | 0.0999279814 |
| 14 | after_redecoder | 5 | 0.0735692856 |
| 15 | after_redecoder | 5 | 0.0907120926 |
| 16 | after_redecoder | 5 | 0.0870094465 |
| 17 | after_redecoder | 5 | 0.0983018 |
| 18 | after_redecoder | 5 | 0.0404858349 |
| 19 | after_redecoder | 5 | 0.167220358 |
| 20 | after_redecoder | 5 | 0.0631603995 |
| 21 | after_redecoder | 5 | 0.0312837829 |
| 22 | after_redecoder | 5 | 0.103203493 |
| 23 | after_redecoder | 5 | 0.0587286171 |
| 24 | after_redecoder | 5 | 0.10625159 |
| 25 | after_redecoder | 5 | 0.0348704279 |
| 26 | after_redecoder | 5 | 0.0209242305 |
