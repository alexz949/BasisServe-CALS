# Layer 7 Mean-128 Chunk8 Fisher Fit

| Split | Fisher NMSE init | Fisher NMSE final |
|---|---:|---:|
| fit | 0.82927 | 0.494386 |
| heldout | 0.827487 | 0.504172 |

| Sweep | Train Fisher NMSE | Held-out Fisher NMSE | Query CG mean/max/hit | Encoder CG mean/max/hit | Wall (s) |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.508229 | 0.511378 | 50.0/50/32 | 50.0/50/8 | 1.67 |
| 2 | 0.497933 | 0.50521 | 50.0/50/32 | 50.0/50/8 | 1.55 |
| 3 | 0.495996 | 0.504353 | 50.0/50/32 | 50.0/50/8 | 1.55 |
| 4 | 0.495227 | 0.504143 | 50.0/50/32 | 50.0/50/8 | 1.55 |
| 5 | 0.494854 | 0.504141 | 50.0/50/32 | 50.0/50/8 | 1.55 |
| 6 | 0.494642 | 0.504156 | 50.0/50/32 | 50.0/50/8 | 1.55 |
| 7 | 0.494509 | 0.504187 | 50.0/50/32 | 50.0/50/8 | 1.55 |
| 8 | 0.49442 | 0.504195 | 50.0/50/32 | 50.0/50/8 | 1.55 |

The objective is evaluated exactly from fixed-teacher Mean-128 Fisher sufficient statistics. Flat-1024 is not captured or fitted.
