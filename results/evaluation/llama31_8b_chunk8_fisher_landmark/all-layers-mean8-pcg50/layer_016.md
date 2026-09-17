# Layer 16 Mean-128 Chunk8 Fisher Fit

| Split | Fisher NMSE init | Fisher NMSE final |
|---|---:|---:|
| fit | 0.847912 | 0.348232 |
| heldout | 0.864564 | 0.405616 |

| Sweep | Train Fisher NMSE | Held-out Fisher NMSE | Query CG mean/max/hit | Encoder CG mean/max/hit | Wall (s) |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.365021 | 0.412274 | 50.0/50/32 | 50.0/50/8 | 1.65 |
| 2 | 0.352273 | 0.406367 | 50.0/50/32 | 50.0/50/8 | 1.53 |
| 3 | 0.350116 | 0.405799 | 50.0/50/32 | 50.0/50/8 | 1.53 |
| 4 | 0.349237 | 0.405621 | 50.0/50/32 | 50.0/50/8 | 1.53 |
| 5 | 0.348789 | 0.405575 | 50.0/50/32 | 50.0/50/8 | 1.53 |
| 6 | 0.348534 | 0.405607 | 50.0/50/32 | 50.0/50/8 | 1.53 |
| 7 | 0.348377 | 0.405626 | 50.0/50/32 | 50.0/50/8 | 1.53 |
| 8 | 0.348273 | 0.405642 | 50.0/50/32 | 50.0/50/8 | 1.53 |

The objective is evaluated exactly from fixed-teacher Mean-128 Fisher sufficient statistics. Flat-1024 is not captured or fitted.
