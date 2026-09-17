# Layer 13 Mean-128 Chunk8 Fisher Fit

| Split | Fisher NMSE init | Fisher NMSE final |
|---|---:|---:|
| fit | 0.789817 | 0.550979 |
| heldout | 0.83645 | 0.627759 |

| Sweep | Train Fisher NMSE | Held-out Fisher NMSE | Query CG mean/max/hit | Encoder CG mean/max/hit | Wall (s) |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.568827 | 0.636539 | 50.0/50/32 | 50.0/50/8 | 1.65 |
| 2 | 0.55688 | 0.629326 | 50.0/50/32 | 50.0/50/8 | 1.52 |
| 3 | 0.55413 | 0.628531 | 50.0/50/32 | 50.0/50/8 | 1.53 |
| 4 | 0.552844 | 0.628335 | 50.0/50/32 | 50.0/50/8 | 1.53 |
| 5 | 0.55206 | 0.6282 | 50.0/50/32 | 50.0/50/8 | 1.53 |
| 6 | 0.55156 | 0.628048 | 50.0/50/32 | 50.0/50/8 | 1.52 |
| 7 | 0.551247 | 0.627916 | 50.0/50/32 | 50.0/50/8 | 1.52 |
| 8 | 0.55105 | 0.627832 | 50.0/50/32 | 50.0/50/8 | 1.52 |

The objective is evaluated exactly from fixed-teacher Mean-128 Fisher sufficient statistics. Flat-1024 is not captured or fitted.
