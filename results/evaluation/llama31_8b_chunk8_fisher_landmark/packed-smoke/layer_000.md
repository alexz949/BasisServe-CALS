# Layer 0 Direct Chunk8 Fisher Fit

| Feature | Split | Fisher NMSE init | Fisher NMSE final | Chunk rel-MSE final | Exact support recall | Routed-candidate mass |
|---|---|---:|---:|---:|---:|---:|
| mean | fit | 0.768285 | 0.426222 | 0.256515 | 0.7347 | 0.4048 |
| mean | heldout | 0.892611 | 0.837576 | 0.231728 | 0.7002 | 0.4222 |
| flat | fit | 0.768285 | 0.229645 | 0.286114 | 0.8135 | 0.4137 |
| flat | heldout | 0.89261 | 1.15953 | 0.266738 | 0.6716 | 0.4140 |

Fisher candidates exclude fixed sink32 and exact recent64. The saved deployment tensors contain only the direct chunk encoder and the per-query-head query factor.
