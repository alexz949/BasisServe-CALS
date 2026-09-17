# Layer 17 Mean-128 Chunk8 Fisher Fit

| Split | Fisher NMSE init | Fisher NMSE final |
|---|---:|---:|
| fit | 0.915997 | 0.429491 |
| heldout | 0.920002 | 0.507516 |

| Sweep | Train Fisher NMSE | Held-out Fisher NMSE | Query CG mean/max/hit | Encoder CG mean/max/hit | Wall (s) |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.451206 | 0.515879 | 50.0/50/32 | 50.0/50/8 | 1.73 |
| 2 | 0.434945 | 0.508739 | 50.0/50/32 | 50.0/50/8 | 1.53 |
| 3 | 0.431917 | 0.507878 | 50.0/50/32 | 50.0/50/8 | 1.53 |
| 4 | 0.430768 | 0.507558 | 50.0/50/32 | 50.0/50/8 | 1.53 |
| 5 | 0.430203 | 0.50744 | 50.0/50/32 | 50.0/50/8 | 1.53 |
| 6 | 0.429878 | 0.507436 | 50.0/50/32 | 50.0/50/8 | 1.53 |
| 7 | 0.429675 | 0.507448 | 50.0/50/32 | 50.0/50/8 | 1.53 |
| 8 | 0.429542 | 0.507462 | 50.0/50/32 | 50.0/50/8 | 1.53 |

The objective is evaluated exactly from fixed-teacher Mean-128 Fisher sufficient statistics. Flat-1024 is not captured or fitted.
