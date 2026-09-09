# Qwen3-8B C1-V80 to Key Predictive Spectra

| Layer | Convention | Rank | Fit captured K energy | Held-out K MSE | Held-out score NMSE |
|---:|:---|---:|---:|---:|---:|
| 0 | pre_rope | 16 | 0.766860 | 0.236930 | 0.010820 |
| 0 | post_rope | 16 | 0.311770 | 0.689350 | 0.266130 |
| 4 | pre_rope | 16 | 0.360084 | 0.645829 | 0.087970 |
| 4 | post_rope | 16 | 0.107098 | 0.897264 | 0.357306 |
| 8 | pre_rope | 16 | 0.315841 | 0.686669 | 0.057549 |
| 8 | post_rope | 16 | 0.104624 | 0.896926 | 0.310718 |
| 12 | pre_rope | 16 | 0.381482 | 0.622891 | 0.085660 |
| 12 | post_rope | 16 | 0.166379 | 0.836185 | 0.295198 |
| 16 | pre_rope | 16 | 0.326826 | 0.675396 | 0.156164 |
| 16 | post_rope | 16 | 0.129768 | 0.870773 | 0.350542 |
| 20 | pre_rope | 16 | 0.416380 | 0.584554 | 0.128986 |
| 20 | post_rope | 16 | 0.249584 | 0.749716 | 0.322445 |
| 24 | pre_rope | 16 | 0.277370 | 0.727027 | 0.253500 |
| 24 | post_rope | 16 | 0.159473 | 0.844084 | 0.623575 |
| 28 | pre_rope | 16 | 0.321743 | 0.681393 | 0.113086 |
| 28 | post_rope | 16 | 0.133989 | 0.869728 | 0.392133 |
| 32 | pre_rope | 16 | 0.224730 | 0.780398 | 0.203752 |
| 32 | post_rope | 16 | 0.061874 | 0.946337 | 0.505735 |

Fit spectra use 64 independent C4 windows over positions 0--32768. Held-out prediction uses 16 independent C4 windows and their paired final-token Queries.
