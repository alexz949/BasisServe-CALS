# Layer 31 Mean-128 Chunk8 Fisher Fit

| Split | Fisher NMSE init | Fisher NMSE final |
|---|---:|---:|
| fit | 0.774516 | 0.538718 |
| heldout | 0.782462 | 0.599976 |

| Sweep | Train Fisher NMSE | Held-out Fisher NMSE | Query CG mean/max/hit | Encoder CG mean/max/hit | Wall (s) |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.5537 | 0.609477 | 50.0/50/32 | 50.0/50/8 | 1.67 |
| 2 | 0.542677 | 0.602787 | 50.0/50/32 | 50.0/50/8 | 1.53 |
| 3 | 0.540644 | 0.601734 | 50.0/50/32 | 50.0/50/8 | 1.53 |
| 4 | 0.539802 | 0.6012 | 50.0/50/32 | 50.0/50/8 | 1.53 |
| 5 | 0.539365 | 0.600763 | 50.0/50/32 | 50.0/50/8 | 1.53 |
| 6 | 0.5391 | 0.600448 | 50.0/50/32 | 50.0/50/8 | 1.53 |
| 7 | 0.538915 | 0.600211 | 50.0/50/32 | 50.0/50/8 | 1.53 |
| 8 | 0.538776 | 0.600063 | 50.0/50/32 | 50.0/50/8 | 1.53 |

The objective is evaluated exactly from fixed-teacher Mean-128 Fisher sufficient statistics. Flat-1024 is not captured or fitted.
