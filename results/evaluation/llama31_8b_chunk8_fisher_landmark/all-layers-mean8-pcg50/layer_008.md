# Layer 8 Mean-128 Chunk8 Fisher Fit

| Split | Fisher NMSE init | Fisher NMSE final |
|---|---:|---:|
| fit | 0.794222 | 0.6431 |
| heldout | 0.796789 | 0.680377 |

| Sweep | Train Fisher NMSE | Held-out Fisher NMSE | Query CG mean/max/hit | Encoder CG mean/max/hit | Wall (s) |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.658114 | 0.688346 | 50.0/50/32 | 50.0/50/8 | 1.67 |
| 2 | 0.648624 | 0.682653 | 50.0/50/32 | 50.0/50/8 | 1.53 |
| 3 | 0.645963 | 0.681485 | 50.0/50/32 | 50.0/50/8 | 1.53 |
| 4 | 0.644717 | 0.680956 | 50.0/50/32 | 50.0/50/8 | 1.53 |
| 5 | 0.644031 | 0.680702 | 50.0/50/32 | 50.0/50/8 | 1.53 |
| 6 | 0.64362 | 0.68058 | 50.0/50/32 | 50.0/50/8 | 1.53 |
| 7 | 0.643355 | 0.680511 | 50.0/50/32 | 50.0/50/8 | 1.53 |
| 8 | 0.643172 | 0.680455 | 50.0/50/32 | 50.0/50/8 | 1.53 |

The objective is evaluated exactly from fixed-teacher Mean-128 Fisher sufficient statistics. Flat-1024 is not captured or fitted.
