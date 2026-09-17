# Layer 26 Mean-128 Chunk8 Fisher Fit

| Split | Fisher NMSE init | Fisher NMSE final |
|---|---:|---:|
| fit | 0.96755 | 0.427753 |
| heldout | 0.992689 | 0.518836 |

| Sweep | Train Fisher NMSE | Held-out Fisher NMSE | Query CG mean/max/hit | Encoder CG mean/max/hit | Wall (s) |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.453613 | 0.527129 | 50.0/50/32 | 50.0/50/8 | 1.64 |
| 2 | 0.432933 | 0.518272 | 50.0/50/32 | 50.0/50/8 | 1.53 |
| 3 | 0.430145 | 0.518185 | 50.0/50/32 | 50.0/50/8 | 1.52 |
| 4 | 0.429035 | 0.518411 | 50.0/50/32 | 50.0/50/8 | 1.53 |
| 5 | 0.428477 | 0.518601 | 50.0/50/32 | 50.0/50/8 | 1.53 |
| 6 | 0.428153 | 0.51874 | 50.0/50/32 | 50.0/50/8 | 1.52 |
| 7 | 0.427947 | 0.518805 | 50.0/50/32 | 50.0/50/8 | 1.53 |
| 8 | 0.427807 | 0.51886 | 50.0/50/32 | 50.0/50/8 | 1.53 |

The objective is evaluated exactly from fixed-teacher Mean-128 Fisher sufficient statistics. Flat-1024 is not captured or fitted.
