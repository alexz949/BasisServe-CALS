# Layer 2 Mean-128 Chunk8 Fisher Fit

| Split | Fisher NMSE init | Fisher NMSE final |
|---|---:|---:|
| fit | 0.816958 | 0.700121 |
| heldout | 0.823389 | 0.783024 |

| Sweep | Train Fisher NMSE | Held-out Fisher NMSE | Query CG mean/max/hit | Encoder CG mean/max/hit | Wall (s) |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.720396 | 0.790758 | 50.0/50/32 | 48.8/50/7 | 1.65 |
| 2 | 0.708168 | 0.783703 | 50.0/50/32 | 50.0/50/8 | 1.55 |
| 3 | 0.704298 | 0.7824 | 50.0/50/32 | 50.0/50/8 | 1.55 |
| 4 | 0.702495 | 0.782445 | 50.0/50/32 | 50.0/50/8 | 1.55 |
| 5 | 0.70149 | 0.782782 | 50.0/50/32 | 50.0/50/8 | 1.55 |
| 6 | 0.700878 | 0.783059 | 50.0/50/32 | 50.0/50/8 | 1.55 |
| 7 | 0.700484 | 0.783178 | 50.0/50/32 | 50.0/50/8 | 1.55 |
| 8 | 0.700222 | 0.783197 | 50.0/50/32 | 50.0/50/8 | 1.55 |

The objective is evaluated exactly from fixed-teacher Mean-128 Fisher sufficient statistics. Flat-1024 is not captured or fitted.
