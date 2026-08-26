# DeepSeek-V2-Lite TP8 post-attention C1

- MLA and KV cache are unchanged.
- Source width/rank: `256/192`.
- Communication reduction versus dense AllGather: `25%`.

| Layer | Boundary | Sweep | Heldout relative MSE |
|---:|:---|---:|---:|
| 0 | after_redecoder | 2 | 0.000117131196 |
| 1 | after_redecoder | 5 | 0.00132415049 |
| 2 | after_redecoder | 5 | 0.00993459646 |
| 3 | after_redecoder | 5 | 0.0134610084 |
| 4 | after_redecoder | 5 | 0.0165290676 |
| 5 | after_redecoder | 5 | 0.0161937225 |
| 6 | after_redecoder | 5 | 0.016524951 |
| 7 | after_redecoder | 5 | 0.0184446198 |
| 8 | after_redecoder | 5 | 0.0299070576 |
| 9 | after_redecoder | 5 | 0.0322268937 |
| 10 | after_redecoder | 5 | 0.0329621117 |
| 11 | after_redecoder | 5 | 0.0197402106 |
| 12 | after_redecoder | 5 | 0.0280790139 |
| 13 | after_redecoder | 5 | 0.0273850997 |
| 14 | after_redecoder | 5 | 0.0219556368 |
| 15 | after_redecoder | 5 | 0.0243024604 |
| 16 | after_redecoder | 5 | 0.0255002887 |
| 17 | after_redecoder | 5 | 0.0286805736 |
| 18 | after_redecoder | 5 | 0.011908961 |
| 19 | after_redecoder | 5 | 0.0567688044 |
| 20 | after_redecoder | 5 | 0.0191461546 |
| 21 | after_redecoder | 5 | 0.00911104188 |
| 22 | after_redecoder | 5 | 0.028913225 |
| 23 | after_redecoder | 5 | 0.0181664474 |
| 24 | after_redecoder | 5 | 0.0294613173 |
| 25 | after_redecoder | 5 | 0.0106360589 |
| 26 | after_redecoder | 5 | 0.00675347256 |
