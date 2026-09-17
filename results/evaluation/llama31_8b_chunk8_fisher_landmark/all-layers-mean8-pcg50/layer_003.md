# Layer 3 Mean-128 Chunk8 Fisher Fit

| Split | Fisher NMSE init | Fisher NMSE final |
|---|---:|---:|
| fit | 0.587721 | 0.533093 |
| heldout | 0.587305 | 0.544542 |

| Sweep | Train Fisher NMSE | Held-out Fisher NMSE | Query CG mean/max/hit | Encoder CG mean/max/hit | Wall (s) |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.539458 | 0.548823 | 50.0/50/32 | 50.0/50/8 | 1.70 |
| 2 | 0.535414 | 0.546117 | 50.0/50/32 | 50.0/50/8 | 1.55 |
| 3 | 0.534342 | 0.545565 | 50.0/50/32 | 50.0/50/8 | 1.55 |
| 4 | 0.533875 | 0.545289 | 50.0/50/32 | 50.0/50/8 | 1.55 |
| 5 | 0.53361 | 0.545113 | 50.0/50/32 | 50.0/50/8 | 1.55 |
| 6 | 0.533425 | 0.544934 | 50.0/50/32 | 50.0/50/8 | 1.55 |
| 7 | 0.533278 | 0.54478 | 50.0/50/32 | 50.0/50/8 | 1.55 |
| 8 | 0.533151 | 0.544618 | 50.0/50/32 | 50.0/50/8 | 1.55 |

The objective is evaluated exactly from fixed-teacher Mean-128 Fisher sufficient statistics. Flat-1024 is not captured or fitted.
