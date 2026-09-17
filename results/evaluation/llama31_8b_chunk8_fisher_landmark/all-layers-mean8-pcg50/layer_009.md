# Layer 9 Mean-128 Chunk8 Fisher Fit

| Split | Fisher NMSE init | Fisher NMSE final |
|---|---:|---:|
| fit | 0.768103 | 0.5593 |
| heldout | 0.778746 | 0.590369 |

| Sweep | Train Fisher NMSE | Held-out Fisher NMSE | Query CG mean/max/hit | Encoder CG mean/max/hit | Wall (s) |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.571935 | 0.597114 | 50.0/50/32 | 50.0/50/8 | 1.65 |
| 2 | 0.563425 | 0.592709 | 50.0/50/32 | 50.0/50/8 | 1.52 |
| 3 | 0.561381 | 0.591439 | 50.0/50/32 | 50.0/50/8 | 1.52 |
| 4 | 0.560531 | 0.590797 | 50.0/50/32 | 50.0/50/8 | 1.52 |
| 5 | 0.56004 | 0.590474 | 50.0/50/32 | 50.0/50/8 | 1.52 |
| 6 | 0.559715 | 0.590334 | 50.0/50/32 | 50.0/50/8 | 1.52 |
| 7 | 0.559499 | 0.5903 | 50.0/50/32 | 50.0/50/8 | 1.52 |
| 8 | 0.559355 | 0.590325 | 50.0/50/32 | 50.0/50/8 | 1.53 |

The objective is evaluated exactly from fixed-teacher Mean-128 Fisher sufficient statistics. Flat-1024 is not captured or fitted.
