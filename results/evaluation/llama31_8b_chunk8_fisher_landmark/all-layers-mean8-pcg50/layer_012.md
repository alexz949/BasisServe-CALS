# Layer 12 Mean-128 Chunk8 Fisher Fit

| Split | Fisher NMSE init | Fisher NMSE final |
|---|---:|---:|
| fit | 0.836991 | 0.321683 |
| heldout | 0.824819 | 0.358042 |

| Sweep | Train Fisher NMSE | Held-out Fisher NMSE | Query CG mean/max/hit | Encoder CG mean/max/hit | Wall (s) |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.335388 | 0.361342 | 50.0/50/32 | 50.0/50/8 | 1.91 |
| 2 | 0.324945 | 0.356872 | 50.0/50/32 | 50.0/50/8 | 1.55 |
| 3 | 0.323217 | 0.357183 | 50.0/50/32 | 50.0/50/8 | 1.55 |
| 4 | 0.32253 | 0.357485 | 50.0/50/32 | 50.0/50/8 | 1.54 |
| 5 | 0.322178 | 0.357722 | 50.0/50/32 | 50.0/50/8 | 1.55 |
| 6 | 0.321966 | 0.357882 | 50.0/50/32 | 50.0/50/8 | 1.54 |
| 7 | 0.321824 | 0.357967 | 50.0/50/32 | 50.0/50/8 | 1.55 |
| 8 | 0.321723 | 0.358012 | 50.0/50/32 | 50.0/50/8 | 1.55 |

The objective is evaluated exactly from fixed-teacher Mean-128 Fisher sufficient statistics. Flat-1024 is not captured or fitted.
