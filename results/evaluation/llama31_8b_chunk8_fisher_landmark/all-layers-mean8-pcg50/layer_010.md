# Layer 10 Mean-128 Chunk8 Fisher Fit

| Split | Fisher NMSE init | Fisher NMSE final |
|---|---:|---:|
| fit | 0.836686 | 0.620545 |
| heldout | 0.861842 | 0.690441 |

| Sweep | Train Fisher NMSE | Held-out Fisher NMSE | Query CG mean/max/hit | Encoder CG mean/max/hit | Wall (s) |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.640377 | 0.701378 | 50.0/50/32 | 50.0/50/8 | 1.71 |
| 2 | 0.62629 | 0.692795 | 50.0/50/32 | 50.0/50/8 | 1.55 |
| 3 | 0.623401 | 0.691536 | 50.0/50/32 | 50.0/50/8 | 1.55 |
| 4 | 0.622102 | 0.690974 | 50.0/50/32 | 50.0/50/8 | 1.55 |
| 5 | 0.621404 | 0.690734 | 50.0/50/32 | 50.0/50/8 | 1.55 |
| 6 | 0.621007 | 0.690593 | 50.0/50/32 | 50.0/50/8 | 1.55 |
| 7 | 0.620764 | 0.690512 | 50.0/50/32 | 50.0/50/8 | 1.55 |
| 8 | 0.620605 | 0.69046 | 50.0/50/32 | 50.0/50/8 | 1.55 |

The objective is evaluated exactly from fixed-teacher Mean-128 Fisher sufficient statistics. Flat-1024 is not captured or fitted.
