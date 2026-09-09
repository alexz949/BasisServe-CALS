# Qwen3-8B C1-V80 to Key Predictive Spectra

| Layer | Convention | Rank | Fit captured K energy | Held-out K MSE | Held-out score NMSE |
|---:|:---|---:|---:|---:|---:|
| 3 | pre_rope | 16 | 0.356934 | 0.647124 | 0.076542 |
| 3 | post_rope | 16 | 0.099581 | 0.903465 | 0.367965 |
| 7 | pre_rope | 16 | 0.325690 | 0.676425 | 0.133973 |
| 7 | post_rope | 16 | 0.052828 | 0.948849 | 0.550705 |
| 11 | pre_rope | 16 | 0.355013 | 0.648287 | 0.045495 |
| 11 | post_rope | 16 | 0.108397 | 0.896679 | 0.289212 |
| 15 | pre_rope | 16 | 0.435345 | 0.565392 | 0.135786 |
| 15 | post_rope | 16 | 0.190227 | 0.809081 | 0.485184 |
| 19 | pre_rope | 16 | 0.377967 | 0.623793 | 0.172282 |
| 19 | post_rope | 16 | 0.177270 | 0.823243 | 0.464452 |
| 23 | pre_rope | 16 | 0.352203 | 0.651412 | 0.125391 |
| 23 | post_rope | 16 | 0.158735 | 0.843200 | 0.491650 |
| 27 | pre_rope | 16 | 0.263824 | 0.740111 | 0.141935 |
| 27 | post_rope | 16 | 0.106005 | 0.897656 | 0.452783 |
| 31 | pre_rope | 16 | 0.212081 | 0.793525 | 0.220171 |
| 31 | post_rope | 16 | 0.082571 | 0.921965 | 0.571364 |
| 35 | pre_rope | 16 | 0.221955 | 0.786625 | 0.042685 |
| 35 | post_rope | 16 | 0.065903 | 0.942080 | 0.237077 |

Fit spectra use 64 independent C4 windows over positions 0--32768. Held-out prediction uses 16 independent C4 windows and their paired final-token Queries.
