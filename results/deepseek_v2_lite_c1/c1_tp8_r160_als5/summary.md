# DeepSeek-V2-Lite TP8 post-attention C1

- MLA and KV cache are unchanged.
- Source width/rank: `256/160`.
- Communication reduction versus dense AllGather: `37.5%`.

| Layer | Boundary | Sweep | Heldout relative MSE |
|---:|:---|---:|---:|
| 0 | after_redecoder | 5 | 0.000473023045 |
| 1 | after_redecoder | 5 | 0.00427428088 |
| 2 | after_redecoder | 5 | 0.0219427826 |
| 3 | after_redecoder | 5 | 0.0284452315 |
| 4 | after_redecoder | 5 | 0.0334281192 |
| 5 | after_redecoder | 5 | 0.0342547836 |
| 6 | after_redecoder | 5 | 0.0341936603 |
| 7 | after_redecoder | 5 | 0.0364682695 |
| 8 | after_redecoder | 5 | 0.0571004049 |
| 9 | after_redecoder | 5 | 0.0634942409 |
| 10 | after_redecoder | 5 | 0.0639066474 |
| 11 | after_redecoder | 5 | 0.0404215011 |
| 12 | after_redecoder | 5 | 0.0542941864 |
| 13 | after_redecoder | 5 | 0.0561339446 |
| 14 | after_redecoder | 5 | 0.0428017053 |
| 15 | after_redecoder | 5 | 0.0498027403 |
| 16 | after_redecoder | 5 | 0.0498698293 |
| 17 | after_redecoder | 5 | 0.0563004059 |
| 18 | after_redecoder | 5 | 0.023311079 |
| 19 | after_redecoder | 5 | 0.103368683 |
| 20 | after_redecoder | 5 | 0.0369819377 |
| 21 | after_redecoder | 5 | 0.0178169435 |
| 22 | after_redecoder | 5 | 0.0582670143 |
| 23 | after_redecoder | 5 | 0.0341655591 |
| 24 | after_redecoder | 5 | 0.0558046424 |
| 25 | after_redecoder | 5 | 0.0203542302 |
| 26 | after_redecoder | 5 | 0.0125726239 |
