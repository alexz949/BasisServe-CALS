# Layer 0 Direct Chunk8 Fisher Fit

| Feature | Split | Fisher NMSE init | Fisher NMSE final | Chunk rel-MSE final | Exact support recall | Routed-candidate mass |
|---|---|---:|---:|---:|---:|---:|
| mean | fit | 0.768285 | 0.426283 | 0.256073 | 0.7346 | 0.4047 |
| mean | heldout | 0.892611 | 0.828176 | 0.233465 | 0.7030 | 0.4228 |
| flat | fit | 0.768285 | 0.226474 | 0.286102 | 0.8149 | 0.4134 |
| flat | heldout | 0.89261 | 1.16875 | 0.263366 | 0.6689 | 0.4133 |

Fisher candidates exclude fixed sink32 and exact recent64. The saved deployment tensors contain only the direct chunk encoder and the per-query-head query factor.
