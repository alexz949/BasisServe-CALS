# Qwen3-8B Raw-V128 versus C1-V80 Pre-K Predictability

| Layer | Raw-V fit predictable K | C1-V80 fit predictable K | Raw-V held-out centered MSE | C1-V80 held-out centered MSE |
|---:|---:|---:|---:|---:|
| 3 | 0.518127 | 0.448116 | 0.487012 | 0.556797 |
| 7 | 0.480261 | 0.408559 | 0.522597 | 0.594206 |
| 11 | 0.504386 | 0.444603 | 0.499924 | 0.559736 |
| 15 | 0.603966 | 0.551231 | 0.397804 | 0.450459 |
| 19 | 0.530702 | 0.474105 | 0.472276 | 0.528752 |
| 23 | 0.492923 | 0.423333 | 0.511670 | 0.581207 |
| 27 | 0.389890 | 0.335031 | 0.615911 | 0.670534 |
| 31 | 0.332360 | 0.260695 | 0.675189 | 0.746293 |
| 35 | 0.297073 | 0.251425 | 0.715109 | 0.758571 |

Raw-V128 and C1-V80 use the same activation windows and pre-RoPE K targets. The fit predictable fraction is the unrestricted affine explained fraction of centered K energy. Held-out MSE evaluates the fit-split affine map on separate windows.
