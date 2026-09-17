# Layer 1 Mean-128 Chunk8 Fisher Fit

| Split | Fisher NMSE init | Fisher NMSE final |
|---|---:|---:|
| fit | 0.664215 | 0.0544066 |
| heldout | 0.676829 | 0.0991181 |

| Sweep | Train Fisher NMSE | Held-out Fisher NMSE | Query CG mean/max/hit | Encoder CG mean/max/hit | Wall (s) |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.0592403 | 0.102476 | 50.0/50/32 | 50.0/50/8 | 1.90 |
| 2 | 0.0557322 | 0.0993703 | 50.0/50/32 | 50.0/50/8 | 1.53 |
| 3 | 0.0551503 | 0.0993555 | 50.0/50/32 | 50.0/50/8 | 1.53 |
| 4 | 0.0548716 | 0.0992156 | 50.0/50/32 | 50.0/50/8 | 1.53 |
| 5 | 0.0547021 | 0.0992592 | 50.0/50/32 | 50.0/50/8 | 1.53 |
| 6 | 0.0545837 | 0.0992001 | 50.0/50/32 | 50.0/50/8 | 1.53 |
| 7 | 0.0544991 | 0.0992354 | 50.0/50/32 | 50.0/50/8 | 1.53 |
| 8 | 0.0544358 | 0.0991821 | 50.0/50/32 | 50.0/50/8 | 1.53 |

The objective is evaluated exactly from fixed-teacher Mean-128 Fisher sufficient statistics. Flat-1024 is not captured or fitted.
