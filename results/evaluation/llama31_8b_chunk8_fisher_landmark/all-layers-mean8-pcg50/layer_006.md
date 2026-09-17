# Layer 6 Mean-128 Chunk8 Fisher Fit

| Split | Fisher NMSE init | Fisher NMSE final |
|---|---:|---:|
| fit | 0.708713 | 0.612992 |
| heldout | 0.716267 | 0.639413 |

| Sweep | Train Fisher NMSE | Held-out Fisher NMSE | Query CG mean/max/hit | Encoder CG mean/max/hit | Wall (s) |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.623473 | 0.647376 | 50.0/50/32 | 50.0/50/8 | 1.69 |
| 2 | 0.616839 | 0.64243 | 50.0/50/32 | 50.0/50/8 | 1.55 |
| 3 | 0.615119 | 0.641162 | 50.0/50/32 | 50.0/50/8 | 1.55 |
| 4 | 0.614299 | 0.640526 | 50.0/50/32 | 50.0/50/8 | 1.55 |
| 5 | 0.613815 | 0.640145 | 50.0/50/32 | 50.0/50/8 | 1.55 |
| 6 | 0.613492 | 0.639887 | 50.0/50/32 | 50.0/50/8 | 1.55 |
| 7 | 0.613257 | 0.639679 | 50.0/50/32 | 50.0/50/8 | 1.55 |
| 8 | 0.613072 | 0.639505 | 50.0/50/32 | 50.0/50/8 | 1.55 |

The objective is evaluated exactly from fixed-teacher Mean-128 Fisher sufficient statistics. Flat-1024 is not captured or fitted.
