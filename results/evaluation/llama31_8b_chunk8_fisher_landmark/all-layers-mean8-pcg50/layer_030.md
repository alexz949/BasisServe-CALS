# Layer 30 Mean-128 Chunk8 Fisher Fit

| Split | Fisher NMSE init | Fisher NMSE final |
|---|---:|---:|
| fit | 0.889748 | 0.519738 |
| heldout | 0.931272 | 0.606404 |

| Sweep | Train Fisher NMSE | Held-out Fisher NMSE | Query CG mean/max/hit | Encoder CG mean/max/hit | Wall (s) |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.542322 | 0.618757 | 50.0/50/32 | 50.0/50/8 | 1.71 |
| 2 | 0.525547 | 0.608251 | 50.0/50/32 | 50.0/50/8 | 1.55 |
| 3 | 0.522669 | 0.607221 | 50.0/50/32 | 50.0/50/8 | 1.55 |
| 4 | 0.521386 | 0.606788 | 50.0/50/32 | 50.0/50/8 | 1.55 |
| 5 | 0.520692 | 0.606604 | 50.0/50/32 | 50.0/50/8 | 1.55 |
| 6 | 0.520264 | 0.60648 | 50.0/50/32 | 50.0/50/8 | 1.55 |
| 7 | 0.519986 | 0.606422 | 50.0/50/32 | 50.0/50/8 | 1.55 |
| 8 | 0.519805 | 0.606404 | 50.0/50/32 | 50.0/50/8 | 1.55 |

The objective is evaluated exactly from fixed-teacher Mean-128 Fisher sufficient statistics. Flat-1024 is not captured or fitted.
