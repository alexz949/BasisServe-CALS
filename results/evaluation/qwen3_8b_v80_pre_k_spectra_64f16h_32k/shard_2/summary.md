# Qwen3-8B C1-V80 to Key Predictive Spectra

| Layer | Convention | Rank | Fit captured K energy | Held-out K MSE | Held-out score NMSE |
|---:|:---|---:|---:|---:|---:|
| 2 | pre_rope | 16 | 0.317070 | 0.687700 | 0.119127 |
| 2 | post_rope | 16 | 0.079575 | 0.922428 | 0.414843 |
| 6 | pre_rope | 16 | 0.345141 | 0.657224 | 0.063228 |
| 6 | post_rope | 16 | 0.069868 | 0.930851 | 0.407399 |
| 10 | pre_rope | 16 | 0.324084 | 0.680286 | 0.032856 |
| 10 | post_rope | 16 | 0.082969 | 0.924479 | 0.199373 |
| 14 | pre_rope | 16 | 0.443537 | 0.558063 | 0.075708 |
| 14 | post_rope | 16 | 0.142667 | 0.858785 | 0.332655 |
| 18 | pre_rope | 16 | 0.410256 | 0.589993 | 0.212231 |
| 18 | post_rope | 16 | 0.234206 | 0.765299 | 0.508264 |
| 22 | pre_rope | 16 | 0.338321 | 0.665376 | 0.166773 |
| 22 | post_rope | 16 | 0.149541 | 0.854066 | 0.445131 |
| 26 | pre_rope | 16 | 0.218654 | 0.785492 | 0.236963 |
| 26 | post_rope | 16 | 0.094816 | 0.907179 | 0.559667 |
| 30 | pre_rope | 16 | 0.198911 | 0.806690 | 0.132662 |
| 30 | post_rope | 16 | 0.067782 | 0.940996 | 0.500203 |
| 34 | pre_rope | 16 | 0.205689 | 0.800569 | 0.183216 |
| 34 | post_rope | 16 | 0.068479 | 0.938820 | 0.424583 |

Fit spectra use 64 independent C4 windows over positions 0--32768. Held-out prediction uses 16 independent C4 windows and their paired final-token Queries.
