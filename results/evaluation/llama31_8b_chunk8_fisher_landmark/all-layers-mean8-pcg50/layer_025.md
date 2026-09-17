# Layer 25 Mean-128 Chunk8 Fisher Fit

| Split | Fisher NMSE init | Fisher NMSE final |
|---|---:|---:|
| fit | 0.867693 | 0.373393 |
| heldout | 0.885005 | 0.490964 |

| Sweep | Train Fisher NMSE | Held-out Fisher NMSE | Query CG mean/max/hit | Encoder CG mean/max/hit | Wall (s) |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.392405 | 0.49433 | 50.0/50/32 | 50.0/50/8 | 1.67 |
| 2 | 0.377471 | 0.489453 | 50.0/50/32 | 50.0/50/8 | 1.53 |
| 3 | 0.375128 | 0.489866 | 50.0/50/32 | 50.0/50/8 | 1.53 |
| 4 | 0.374284 | 0.490312 | 50.0/50/32 | 50.0/50/8 | 1.53 |
| 5 | 0.373884 | 0.490611 | 50.0/50/32 | 50.0/50/8 | 1.53 |
| 6 | 0.37366 | 0.490788 | 50.0/50/32 | 50.0/50/8 | 1.53 |
| 7 | 0.373522 | 0.490878 | 50.0/50/32 | 50.0/50/8 | 1.53 |
| 8 | 0.373429 | 0.49091 | 50.0/50/32 | 50.0/50/8 | 1.53 |

The objective is evaluated exactly from fixed-teacher Mean-128 Fisher sufficient statistics. Flat-1024 is not captured or fitted.
