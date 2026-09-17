# Layer 23 Mean-128 Chunk8 Fisher Fit

| Split | Fisher NMSE init | Fisher NMSE final |
|---|---:|---:|
| fit | 0.888022 | 0.479464 |
| heldout | 0.919078 | 0.575012 |

| Sweep | Train Fisher NMSE | Held-out Fisher NMSE | Query CG mean/max/hit | Encoder CG mean/max/hit | Wall (s) |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.499796 | 0.582336 | 50.0/50/32 | 50.0/50/8 | 1.70 |
| 2 | 0.484693 | 0.575585 | 50.0/50/32 | 50.0/50/8 | 1.55 |
| 3 | 0.48196 | 0.575126 | 50.0/50/32 | 50.0/50/8 | 1.55 |
| 4 | 0.480854 | 0.57505 | 50.0/50/32 | 50.0/50/8 | 1.55 |
| 5 | 0.480263 | 0.574951 | 50.0/50/32 | 50.0/50/8 | 1.55 |
| 6 | 0.479897 | 0.574934 | 50.0/50/32 | 50.0/50/8 | 1.55 |
| 7 | 0.479665 | 0.574979 | 50.0/50/32 | 50.0/50/8 | 1.55 |
| 8 | 0.479517 | 0.575034 | 50.0/50/32 | 50.0/50/8 | 1.55 |

The objective is evaluated exactly from fixed-teacher Mean-128 Fisher sufficient statistics. Flat-1024 is not captured or fitted.
