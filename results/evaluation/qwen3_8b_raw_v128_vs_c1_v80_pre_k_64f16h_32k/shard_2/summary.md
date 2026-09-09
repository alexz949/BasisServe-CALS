# Qwen3-8B Raw-V128 versus C1-V80 Pre-K Predictability

| Layer | Raw-V fit predictable K | C1-V80 fit predictable K | Raw-V held-out centered MSE | C1-V80 held-out centered MSE |
|---:|---:|---:|---:|---:|
| 2 | 0.485918 | 0.421230 | 0.519845 | 0.584524 |
| 6 | 0.514301 | 0.433634 | 0.488589 | 0.569580 |
| 10 | 0.476644 | 0.410114 | 0.527647 | 0.594827 |
| 14 | 0.640009 | 0.580137 | 0.361971 | 0.422295 |
| 18 | 0.582391 | 0.532519 | 0.418495 | 0.468597 |
| 22 | 0.480493 | 0.419190 | 0.523921 | 0.585316 |
| 26 | 0.325653 | 0.272487 | 0.679993 | 0.732927 |
| 30 | 0.297386 | 0.247405 | 0.711136 | 0.759831 |
| 34 | 0.280067 | 0.233095 | 0.729235 | 0.774689 |

Raw-V128 and C1-V80 use the same activation windows and pre-RoPE K targets. The fit predictable fraction is the unrestricted affine explained fraction of centered K energy. Held-out MSE evaluates the fit-split affine map on separate windows.
