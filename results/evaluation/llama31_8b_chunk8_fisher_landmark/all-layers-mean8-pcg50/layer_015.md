# Layer 15 Mean-128 Chunk8 Fisher Fit

| Split | Fisher NMSE init | Fisher NMSE final |
|---|---:|---:|
| fit | 0.817127 | 0.510073 |
| heldout | 0.837499 | 0.564678 |

| Sweep | Train Fisher NMSE | Held-out Fisher NMSE | Query CG mean/max/hit | Encoder CG mean/max/hit | Wall (s) |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.527139 | 0.576058 | 50.0/50/32 | 50.0/50/8 | 1.91 |
| 2 | 0.515339 | 0.568599 | 50.0/50/32 | 50.0/50/8 | 1.55 |
| 3 | 0.512649 | 0.566841 | 50.0/50/32 | 50.0/50/8 | 1.55 |
| 4 | 0.511498 | 0.565939 | 50.0/50/32 | 50.0/50/8 | 1.55 |
| 5 | 0.510896 | 0.565426 | 50.0/50/32 | 50.0/50/8 | 1.55 |
| 6 | 0.510542 | 0.565103 | 50.0/50/32 | 50.0/50/8 | 1.55 |
| 7 | 0.51031 | 0.564893 | 50.0/50/32 | 50.0/50/8 | 1.55 |
| 8 | 0.510143 | 0.564752 | 50.0/50/32 | 50.0/50/8 | 1.55 |

The objective is evaluated exactly from fixed-teacher Mean-128 Fisher sufficient statistics. Flat-1024 is not captured or fitted.
