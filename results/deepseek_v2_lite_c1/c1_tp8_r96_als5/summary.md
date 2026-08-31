# DeepSeek-V2-Lite TP8 post-attention C1

- MLA and KV cache are unchanged.
- Source width/rank: `256/96`.
- Communication reduction versus dense AllGather: `62.5%`.

| Layer | Boundary | Sweep | Heldout relative MSE |
|---:|:---|---:|---:|
| 0 | after_redecoder | 5 | 0.00429804456 |
| 1 | after_redecoder | 5 | 0.0393662717 |
| 2 | after_redecoder | 5 | 0.0820649066 |
| 3 | after_redecoder | 5 | 0.0988813476 |
| 4 | after_redecoder | 5 | 0.101098079 |
| 5 | after_redecoder | 5 | 0.116198494 |
| 6 | after_redecoder | 5 | 0.117884418 |
| 7 | after_redecoder | 5 | 0.110194648 |
| 8 | after_redecoder | 5 | 0.156573599 |
| 9 | after_redecoder | 5 | 0.181833733 |
| 10 | after_redecoder | 5 | 0.173337545 |
| 11 | after_redecoder | 5 | 0.132999367 |
| 12 | after_redecoder | 5 | 0.154762141 |
| 13 | after_redecoder | 5 | 0.167905325 |
| 14 | after_redecoder | 5 | 0.119666739 |
| 15 | after_redecoder | 5 | 0.149752768 |
| 16 | after_redecoder | 5 | 0.143643624 |
| 17 | after_redecoder | 5 | 0.162008666 |
| 18 | after_redecoder | 5 | 0.066932096 |
| 19 | after_redecoder | 5 | 0.254487377 |
| 20 | after_redecoder | 5 | 0.101854625 |
| 21 | after_redecoder | 5 | 0.0541815812 |
| 22 | after_redecoder | 5 | 0.168374431 |
| 23 | after_redecoder | 5 | 0.0998898859 |
| 24 | after_redecoder | 5 | 0.229575866 |
| 25 | after_redecoder | 5 | 0.0612237207 |
| 26 | after_redecoder | 5 | 0.0334979411 |
