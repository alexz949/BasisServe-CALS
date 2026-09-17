# Layer 11 Mean-128 Chunk8 Fisher Fit

| Split | Fisher NMSE init | Fisher NMSE final |
|---|---:|---:|
| fit | 0.865042 | 0.577805 |
| heldout | 0.868032 | 0.650531 |

| Sweep | Train Fisher NMSE | Held-out Fisher NMSE | Query CG mean/max/hit | Encoder CG mean/max/hit | Wall (s) |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.593437 | 0.654884 | 50.0/50/32 | 50.0/50/8 | 1.67 |
| 2 | 0.581748 | 0.650415 | 50.0/50/32 | 50.0/50/8 | 1.55 |
| 3 | 0.579644 | 0.650204 | 50.0/50/32 | 50.0/50/8 | 1.55 |
| 4 | 0.578817 | 0.650194 | 50.0/50/32 | 50.0/50/8 | 1.55 |
| 5 | 0.578394 | 0.650229 | 50.0/50/32 | 50.0/50/8 | 1.55 |
| 6 | 0.578142 | 0.650308 | 50.0/50/32 | 50.0/50/8 | 1.55 |
| 7 | 0.577975 | 0.650394 | 50.0/50/32 | 50.0/50/8 | 1.55 |
| 8 | 0.577855 | 0.650482 | 50.0/50/32 | 50.0/50/8 | 1.55 |

The objective is evaluated exactly from fixed-teacher Mean-128 Fisher sufficient statistics. Flat-1024 is not captured or fitted.
