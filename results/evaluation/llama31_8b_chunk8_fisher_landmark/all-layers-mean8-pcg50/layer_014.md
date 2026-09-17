# Layer 14 Mean-128 Chunk8 Fisher Fit

| Split | Fisher NMSE init | Fisher NMSE final |
|---|---:|---:|
| fit | 0.791172 | 0.59506 |
| heldout | 0.819325 | 0.656372 |

| Sweep | Train Fisher NMSE | Held-out Fisher NMSE | Query CG mean/max/hit | Encoder CG mean/max/hit | Wall (s) |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.608605 | 0.663005 | 50.0/50/32 | 50.0/50/8 | 1.69 |
| 2 | 0.600006 | 0.658801 | 50.0/50/32 | 50.0/50/8 | 1.52 |
| 3 | 0.597618 | 0.657909 | 50.0/50/32 | 50.0/50/8 | 1.53 |
| 4 | 0.596505 | 0.657455 | 50.0/50/32 | 50.0/50/8 | 1.52 |
| 5 | 0.595923 | 0.65715 | 50.0/50/32 | 50.0/50/8 | 1.52 |
| 6 | 0.595572 | 0.656912 | 50.0/50/32 | 50.0/50/8 | 1.53 |
| 7 | 0.595331 | 0.656695 | 50.0/50/32 | 50.0/50/8 | 1.52 |
| 8 | 0.595143 | 0.656518 | 50.0/50/32 | 50.0/50/8 | 1.52 |

The objective is evaluated exactly from fixed-teacher Mean-128 Fisher sufficient statistics. Flat-1024 is not captured or fitted.
