# Llama-3.1-70B GQA C1 V80 joint fit

- K remains dense; every one of 8 physical V heads retains rank 80/128.
- Total KV-cache retention is 81.25% (18.75% reduction).
- Encoder initialization: `activation-weighted-svd`.
- Checkpoint policy: `fixed decoder-refitted endpoint after encoder sweep 6`.
- V encoders and head output decoders use a full-layer joint solve with cross-head covariance.
- Fit contexts: 256; held-out diagnostic contexts: 64.
- Mean held-out factor-dtype relative MSE: `0.156055054`.

| Layer | Exported boundary | Sweep | Held-out relative MSE |
|---:|:---|---:|---:|
| 0 | after_redecoder | 6 | 0.00212788405 |
| 1 | after_redecoder | 6 | 0.0318323255 |
| 2 | after_redecoder | 6 | 0.0511261154 |
| 3 | after_redecoder | 6 | 0.0399528311 |
| 4 | after_redecoder | 6 | 0.0230954171 |
| 5 | after_redecoder | 6 | 0.0617880199 |
| 6 | after_redecoder | 6 | 0.073914659 |
| 7 | after_redecoder | 6 | 0.0583525269 |
| 8 | after_redecoder | 6 | 0.0381739183 |
| 9 | after_redecoder | 6 | 0.0579588185 |
| 10 | after_redecoder | 6 | 0.0841340687 |
| 11 | after_redecoder | 6 | 0.113065035 |
| 12 | after_redecoder | 6 | 0.0462370318 |
| 13 | after_redecoder | 6 | 0.0617696473 |
| 14 | after_redecoder | 6 | 0.0712867971 |
| 15 | after_redecoder | 6 | 0.0774692993 |
| 16 | after_redecoder | 6 | 0.0706934733 |
| 17 | after_redecoder | 6 | 0.112294163 |
| 18 | after_redecoder | 6 | 0.14922805 |
| 19 | after_redecoder | 6 | 0.1268008 |
| 20 | after_redecoder | 6 | 0.131303243 |
| 21 | after_redecoder | 6 | 0.17231609 |
| 22 | after_redecoder | 6 | 0.221333125 |
| 23 | after_redecoder | 6 | 0.19362789 |
| 24 | after_redecoder | 6 | 0.125036216 |
| 25 | after_redecoder | 6 | 0.166232039 |
| 26 | after_redecoder | 6 | 0.126327152 |
| 27 | after_redecoder | 6 | 0.130775172 |
| 28 | after_redecoder | 6 | 0.119619986 |
| 29 | after_redecoder | 6 | 0.129125867 |
| 30 | after_redecoder | 6 | 0.12831198 |
| 31 | after_redecoder | 6 | 0.132925555 |
| 32 | after_redecoder | 6 | 0.146435808 |
| 33 | after_redecoder | 6 | 0.138714483 |
| 34 | after_redecoder | 6 | 0.1285152 |
| 35 | after_redecoder | 6 | 0.153013271 |
| 36 | after_redecoder | 6 | 0.168147677 |
| 37 | after_redecoder | 6 | 0.162086673 |
| 38 | after_redecoder | 6 | 0.17883682 |
| 39 | after_redecoder | 6 | 0.208466503 |
| 40 | after_redecoder | 6 | 0.186307781 |
| 41 | after_redecoder | 6 | 0.23880145 |
| 42 | after_redecoder | 6 | 0.282426247 |
| 43 | after_redecoder | 6 | 0.19836028 |
| 44 | after_redecoder | 6 | 0.198548871 |
| 45 | after_redecoder | 6 | 0.216295201 |
| 46 | after_redecoder | 6 | 0.323014904 |
| 47 | after_redecoder | 6 | 0.217108621 |
| 48 | after_redecoder | 6 | 0.277605714 |
| 49 | after_redecoder | 6 | 0.21454173 |
| 50 | after_redecoder | 6 | 0.291972405 |
| 51 | after_redecoder | 6 | 0.224552717 |
| 52 | after_redecoder | 6 | 0.164169803 |
| 53 | after_redecoder | 6 | 0.296148 |
| 54 | after_redecoder | 6 | 0.238269177 |
| 55 | after_redecoder | 6 | 0.143289082 |
| 56 | after_redecoder | 6 | 0.23895082 |
| 57 | after_redecoder | 6 | 0.246029417 |
| 58 | after_redecoder | 6 | 0.255216014 |
| 59 | after_redecoder | 6 | 0.308350132 |
| 60 | after_redecoder | 6 | 0.256506024 |
| 61 | after_redecoder | 6 | 0.310789587 |
| 62 | after_redecoder | 6 | 0.30441692 |
| 63 | after_redecoder | 6 | 0.216635018 |
| 64 | after_redecoder | 6 | 0.211729315 |
| 65 | after_redecoder | 6 | 0.204154927 |
| 66 | after_redecoder | 6 | 0.159980609 |
| 67 | after_redecoder | 6 | 0.220086481 |
| 68 | after_redecoder | 6 | 0.186186216 |
| 69 | after_redecoder | 6 | 0.186521002 |
| 70 | after_redecoder | 6 | 0.138838447 |
| 71 | after_redecoder | 6 | 0.114153086 |
| 72 | after_redecoder | 6 | 0.100733126 |
| 73 | after_redecoder | 6 | 0.0791674167 |
| 74 | after_redecoder | 6 | 0.0689728761 |
| 75 | after_redecoder | 6 | 0.105803245 |
| 76 | after_redecoder | 6 | 0.0722308125 |
| 77 | after_redecoder | 6 | 0.12195421 |
| 78 | after_redecoder | 6 | 0.163875517 |
| 79 | after_redecoder | 6 | 0.0892594572 |
