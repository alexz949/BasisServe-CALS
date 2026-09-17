# Layer 21 Mean-128 Chunk8 Fisher Fit

| Split | Fisher NMSE init | Fisher NMSE final |
|---|---:|---:|
| fit | 0.847426 | 0.426742 |
| heldout | 0.873937 | 0.543987 |

| Sweep | Train Fisher NMSE | Held-out Fisher NMSE | Query CG mean/max/hit | Encoder CG mean/max/hit | Wall (s) |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.442646 | 0.548554 | 50.0/50/32 | 50.0/50/8 | 1.67 |
| 2 | 0.430457 | 0.543952 | 50.0/50/32 | 50.0/50/8 | 1.55 |
| 3 | 0.428487 | 0.543852 | 50.0/50/32 | 50.0/50/8 | 1.55 |
| 4 | 0.427684 | 0.543948 | 50.0/50/32 | 50.0/50/8 | 1.55 |
| 5 | 0.427272 | 0.543985 | 50.0/50/32 | 50.0/50/8 | 1.55 |
| 6 | 0.427034 | 0.543967 | 50.0/50/32 | 50.0/50/8 | 1.55 |
| 7 | 0.426884 | 0.543964 | 50.0/50/32 | 50.0/50/8 | 1.55 |
| 8 | 0.426782 | 0.543944 | 50.0/50/32 | 50.0/50/8 | 1.55 |

The objective is evaluated exactly from fixed-teacher Mean-128 Fisher sufficient statistics. Flat-1024 is not captured or fitted.
