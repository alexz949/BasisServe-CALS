# Qwen3-8B Raw-V128 versus C1-V80 Pre-K Predictability

| Layer | Raw-V fit predictable K | C1-V80 fit predictable K | Raw-V held-out centered MSE | C1-V80 held-out centered MSE |
|---:|---:|---:|---:|---:|
| 1 | 0.611507 | 0.526461 | 0.394677 | 0.479694 |
| 5 | 0.533688 | 0.464698 | 0.469945 | 0.538977 |
| 9 | 0.410426 | 0.352455 | 0.591889 | 0.650351 |
| 13 | 0.562411 | 0.501020 | 0.439621 | 0.500910 |
| 17 | 0.625641 | 0.573445 | 0.376910 | 0.429594 |
| 21 | 0.597894 | 0.546674 | 0.404980 | 0.456034 |
| 25 | 0.450724 | 0.399170 | 0.554271 | 0.605497 |
| 29 | 0.273118 | 0.205951 | 0.733398 | 0.799985 |
| 33 | 0.257791 | 0.197232 | 0.749540 | 0.808995 |

Raw-V128 and C1-V80 use the same activation windows and pre-RoPE K targets. The fit predictable fraction is the unrestricted affine explained fraction of centered K energy. Held-out MSE evaluates the fit-split affine map on separate windows.
