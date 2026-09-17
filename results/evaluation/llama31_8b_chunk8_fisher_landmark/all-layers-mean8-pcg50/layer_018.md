# Layer 18 Mean-128 Chunk8 Fisher Fit

| Split | Fisher NMSE init | Fisher NMSE final |
|---|---:|---:|
| fit | 0.872264 | 0.424935 |
| heldout | 0.876531 | 0.511104 |

| Sweep | Train Fisher NMSE | Held-out Fisher NMSE | Query CG mean/max/hit | Encoder CG mean/max/hit | Wall (s) |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.44432 | 0.515306 | 50.0/50/32 | 50.0/50/8 | 1.69 |
| 2 | 0.429595 | 0.511058 | 50.0/50/32 | 50.0/50/8 | 1.55 |
| 3 | 0.427118 | 0.511041 | 50.0/50/32 | 50.0/50/8 | 1.55 |
| 4 | 0.426102 | 0.511168 | 50.0/50/32 | 50.0/50/8 | 1.55 |
| 5 | 0.425577 | 0.511201 | 50.0/50/32 | 50.0/50/8 | 1.55 |
| 6 | 0.425278 | 0.511198 | 50.0/50/32 | 50.0/50/8 | 1.55 |
| 7 | 0.425097 | 0.511171 | 50.0/50/32 | 50.0/50/8 | 1.55 |
| 8 | 0.42498 | 0.511159 | 50.0/50/32 | 50.0/50/8 | 1.55 |

The objective is evaluated exactly from fixed-teacher Mean-128 Fisher sufficient statistics. Flat-1024 is not captured or fitted.
