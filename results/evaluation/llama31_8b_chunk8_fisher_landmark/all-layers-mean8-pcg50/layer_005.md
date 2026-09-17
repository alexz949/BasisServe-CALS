# Layer 5 Mean-128 Chunk8 Fisher Fit

| Split | Fisher NMSE init | Fisher NMSE final |
|---|---:|---:|
| fit | 0.761183 | 0.628979 |
| heldout | 0.76438 | 0.687884 |

| Sweep | Train Fisher NMSE | Held-out Fisher NMSE | Query CG mean/max/hit | Encoder CG mean/max/hit | Wall (s) |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.640892 | 0.690198 | 50.0/50/32 | 49.8/50/7 | 1.71 |
| 2 | 0.633454 | 0.687359 | 50.0/50/32 | 50.0/50/8 | 1.52 |
| 3 | 0.631474 | 0.687192 | 50.0/50/32 | 50.0/50/8 | 1.53 |
| 4 | 0.630488 | 0.687246 | 50.0/50/32 | 50.0/50/8 | 1.53 |
| 5 | 0.6299 | 0.687364 | 50.0/50/32 | 50.0/50/8 | 1.52 |
| 6 | 0.629517 | 0.687506 | 50.0/50/32 | 50.0/50/8 | 1.53 |
| 7 | 0.629251 | 0.687642 | 50.0/50/32 | 50.0/50/8 | 1.53 |
| 8 | 0.629058 | 0.687779 | 50.0/50/32 | 50.0/50/8 | 1.52 |

The objective is evaluated exactly from fixed-teacher Mean-128 Fisher sufficient statistics. Flat-1024 is not captured or fitted.
