# Layer 24 Mean-128 Chunk8 Fisher Fit

| Split | Fisher NMSE init | Fisher NMSE final |
|---|---:|---:|
| fit | 0.895571 | 0.453265 |
| heldout | 0.919144 | 0.549591 |

| Sweep | Train Fisher NMSE | Held-out Fisher NMSE | Query CG mean/max/hit | Encoder CG mean/max/hit | Wall (s) |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.473405 | 0.555232 | 50.0/50/32 | 50.0/50/8 | 1.71 |
| 2 | 0.457689 | 0.54856 | 50.0/50/32 | 50.0/50/8 | 1.55 |
| 3 | 0.455335 | 0.548737 | 50.0/50/32 | 50.0/50/8 | 1.55 |
| 4 | 0.454406 | 0.54901 | 50.0/50/32 | 50.0/50/8 | 1.55 |
| 5 | 0.453925 | 0.549239 | 50.0/50/32 | 50.0/50/8 | 1.55 |
| 6 | 0.453633 | 0.54944 | 50.0/50/32 | 50.0/50/8 | 1.55 |
| 7 | 0.453444 | 0.549553 | 50.0/50/32 | 50.0/50/8 | 1.55 |
| 8 | 0.453316 | 0.549616 | 50.0/50/32 | 50.0/50/8 | 1.55 |

The objective is evaluated exactly from fixed-teacher Mean-128 Fisher sufficient statistics. Flat-1024 is not captured or fitted.
