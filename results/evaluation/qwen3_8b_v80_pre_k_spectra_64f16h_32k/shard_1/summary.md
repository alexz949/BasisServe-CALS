# Qwen3-8B C1-V80 to Key Predictive Spectra

| Layer | Convention | Rank | Fit captured K energy | Held-out K MSE | Held-out score NMSE |
|---:|:---|---:|---:|---:|---:|
| 1 | pre_rope | 16 | 0.428806 | 0.576668 | 0.049152 |
| 1 | post_rope | 16 | 0.079667 | 0.921586 | 0.407275 |
| 5 | pre_rope | 16 | 0.381139 | 0.622060 | 0.036387 |
| 5 | post_rope | 16 | 0.090709 | 0.911338 | 0.359612 |
| 9 | pre_rope | 16 | 0.272377 | 0.730037 | 0.117934 |
| 9 | post_rope | 16 | 0.105307 | 0.895993 | 0.275812 |
| 13 | pre_rope | 16 | 0.371413 | 0.630130 | 0.147423 |
| 13 | post_rope | 16 | 0.174333 | 0.825736 | 0.359208 |
| 17 | pre_rope | 16 | 0.451659 | 0.550211 | 0.124179 |
| 17 | post_rope | 16 | 0.266656 | 0.734078 | 0.378095 |
| 21 | pre_rope | 16 | 0.425064 | 0.576854 | 0.160194 |
| 21 | post_rope | 16 | 0.232570 | 0.767767 | 0.412571 |
| 25 | pre_rope | 16 | 0.304104 | 0.699355 | 0.095935 |
| 25 | post_rope | 16 | 0.110220 | 0.891809 | 0.470216 |
| 29 | pre_rope | 16 | 0.180754 | 0.824113 | 0.278408 |
| 29 | post_rope | 16 | 0.092254 | 0.910830 | 0.636057 |
| 33 | pre_rope | 16 | 0.160510 | 0.844647 | 0.350340 |
| 33 | post_rope | 16 | 0.077760 | 0.925634 | 0.614310 |

Fit spectra use 64 independent C4 windows over positions 0--32768. Held-out prediction uses 16 independent C4 windows and their paired final-token Queries.
