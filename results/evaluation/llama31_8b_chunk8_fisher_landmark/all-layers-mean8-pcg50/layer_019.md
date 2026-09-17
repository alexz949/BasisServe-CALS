# Layer 19 Mean-128 Chunk8 Fisher Fit

| Split | Fisher NMSE init | Fisher NMSE final |
|---|---:|---:|
| fit | 0.816833 | 0.463327 |
| heldout | 0.82433 | 0.548973 |

| Sweep | Train Fisher NMSE | Held-out Fisher NMSE | Query CG mean/max/hit | Encoder CG mean/max/hit | Wall (s) |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.478127 | 0.550799 | 50.0/50/32 | 50.0/50/8 | 1.67 |
| 2 | 0.467392 | 0.548767 | 50.0/50/32 | 50.0/50/8 | 1.55 |
| 3 | 0.46527 | 0.548977 | 50.0/50/32 | 50.0/50/8 | 1.55 |
| 4 | 0.464377 | 0.549057 | 50.0/50/32 | 50.0/50/8 | 1.55 |
| 5 | 0.463917 | 0.549021 | 50.0/50/32 | 50.0/50/8 | 1.55 |
| 6 | 0.463649 | 0.549002 | 50.0/50/32 | 50.0/50/8 | 1.55 |
| 7 | 0.463481 | 0.548988 | 50.0/50/32 | 50.0/50/8 | 1.55 |
| 8 | 0.46337 | 0.548993 | 50.0/50/32 | 50.0/50/8 | 1.55 |

The objective is evaluated exactly from fixed-teacher Mean-128 Fisher sufficient statistics. Flat-1024 is not captured or fitted.
