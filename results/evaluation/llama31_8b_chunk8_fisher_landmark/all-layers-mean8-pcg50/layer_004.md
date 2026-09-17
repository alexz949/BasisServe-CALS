# Layer 4 Mean-128 Chunk8 Fisher Fit

| Split | Fisher NMSE init | Fisher NMSE final |
|---|---:|---:|
| fit | 0.683739 | 0.600275 |
| heldout | 0.681484 | 0.627901 |

| Sweep | Train Fisher NMSE | Held-out Fisher NMSE | Query CG mean/max/hit | Encoder CG mean/max/hit | Wall (s) |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.610755 | 0.632011 | 50.0/50/32 | 48.9/50/7 | 1.63 |
| 2 | 0.604116 | 0.628707 | 50.0/50/32 | 50.0/50/8 | 1.53 |
| 3 | 0.602336 | 0.628166 | 50.0/50/32 | 50.0/50/8 | 1.53 |
| 4 | 0.601487 | 0.627954 | 50.0/50/32 | 50.0/50/8 | 1.53 |
| 5 | 0.601 | 0.627876 | 50.0/50/32 | 50.0/50/8 | 1.53 |
| 6 | 0.60069 | 0.627844 | 50.0/50/32 | 50.0/50/8 | 1.53 |
| 7 | 0.60048 | 0.627859 | 50.0/50/32 | 50.0/50/8 | 1.53 |
| 8 | 0.600333 | 0.627879 | 50.0/50/32 | 50.0/50/8 | 1.53 |

The objective is evaluated exactly from fixed-teacher Mean-128 Fisher sufficient statistics. Flat-1024 is not captured or fitted.
