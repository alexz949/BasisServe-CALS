# Qwen3-8B Raw-V128 versus C1-V80 Pre-K Predictability

| Layer | Raw-V fit predictable K | C1-V80 fit predictable K | Raw-V held-out centered MSE | C1-V80 held-out centered MSE |
|---:|---:|---:|---:|---:|
| 0 | 0.808375 | 0.775993 | 0.193978 | 0.227909 |
| 4 | 0.494996 | 0.431524 | 0.511436 | 0.575288 |
| 8 | 0.474169 | 0.412491 | 0.528963 | 0.590515 |
| 12 | 0.561455 | 0.499277 | 0.442396 | 0.505389 |
| 16 | 0.476712 | 0.417172 | 0.525871 | 0.585896 |
| 20 | 0.602313 | 0.547307 | 0.399360 | 0.454484 |
| 24 | 0.407901 | 0.337049 | 0.597146 | 0.668455 |
| 28 | 0.483947 | 0.427358 | 0.520515 | 0.576757 |
| 32 | 0.323083 | 0.265792 | 0.684296 | 0.740728 |

Raw-V128 and C1-V80 use the same activation windows and pre-RoPE K targets. The fit predictable fraction is the unrestricted affine explained fraction of centered K energy. Held-out MSE evaluates the fit-split affine map on separate windows.
