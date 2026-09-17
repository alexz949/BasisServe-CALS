# Layer 1 Mean-128 Chunk8 Fisher Fit

| Split | Fisher NMSE init | Fisher NMSE final |
|---|---:|---:|
| fit | 0.860714 | 0.533798 |
| heldout | 0.867353 | 1.23952 |

| Sweep | Train Fisher NMSE | Held-out Fisher NMSE | Query CG mean/max/hit | Encoder CG mean/max/hit | Wall (s) |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.599807 | 1.04163 | 50.0/50/32 | 50.0/50/8 | 0.64 |
| 2 | 0.562905 | 1.09475 | 50.0/50/32 | 50.0/50/8 | 0.52 |
| 3 | 0.55105 | 1.1248 | 50.0/50/32 | 50.0/50/8 | 0.53 |
| 4 | 0.544852 | 1.15033 | 50.0/50/32 | 50.0/50/8 | 0.53 |
| 5 | 0.540935 | 1.1733 | 50.0/50/32 | 50.0/50/8 | 0.53 |
| 6 | 0.538148 | 1.19457 | 50.0/50/32 | 50.0/50/8 | 0.53 |
| 7 | 0.53605 | 1.21391 | 50.0/50/32 | 50.0/50/8 | 0.53 |
| 8 | 0.53445 | 1.23144 | 50.0/50/32 | 50.0/50/8 | 0.53 |

The objective is evaluated exactly from fixed-teacher Mean-128 Fisher sufficient statistics. Flat-1024 is not captured or fitted.
