# Qwen3-8B C1-V80 to Key Predictive Spectra

| Layer | Convention | Rank | Fit captured K energy | Held-out K MSE | Held-out score NMSE |
|---:|:---|---:|---:|---:|---:|
| 0 | pre_rope | 16 | 0.795690 | 0.260069 | 0.008235 |
| 0 | post_rope | 16 | 0.607832 | 0.459945 | 0.023250 |

Fit spectra use 1 independent C4 windows over positions 0--2048. Held-out prediction uses 1 independent C4 windows and their paired final-token Queries.
