# Layer 29 Mean-128 Chunk8 Fisher Fit

| Split | Fisher NMSE init | Fisher NMSE final |
|---|---:|---:|
| fit | 0.853814 | 0.330225 |
| heldout | 0.859715 | 0.43067 |

| Sweep | Train Fisher NMSE | Held-out Fisher NMSE | Query CG mean/max/hit | Encoder CG mean/max/hit | Wall (s) |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.348881 | 0.436022 | 50.0/50/32 | 50.0/50/8 | 1.66 |
| 2 | 0.334019 | 0.430557 | 50.0/50/32 | 50.0/50/8 | 1.53 |
| 3 | 0.331881 | 0.430508 | 50.0/50/32 | 50.0/50/8 | 1.53 |
| 4 | 0.33109 | 0.430539 | 50.0/50/32 | 50.0/50/8 | 1.53 |
| 5 | 0.330718 | 0.43056 | 50.0/50/32 | 50.0/50/8 | 1.53 |
| 6 | 0.330506 | 0.430562 | 50.0/50/32 | 50.0/50/8 | 1.53 |
| 7 | 0.330367 | 0.430565 | 50.0/50/32 | 50.0/50/8 | 1.53 |
| 8 | 0.330266 | 0.430598 | 50.0/50/32 | 50.0/50/8 | 1.53 |

The objective is evaluated exactly from fixed-teacher Mean-128 Fisher sufficient statistics. Flat-1024 is not captured or fitted.
