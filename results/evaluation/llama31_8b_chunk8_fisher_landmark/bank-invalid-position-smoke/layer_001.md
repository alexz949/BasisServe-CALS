# Layer 1 Direct Chunk8 Fisher Fit

| Feature | Split | Fisher NMSE init | Fisher NMSE final | Chunk rel-MSE final | Exact support recall | Routed-candidate mass |
|---|---|---:|---:|---:|---:|---:|
| mean | fit | 0.860714 | 0.533799 | 0.0123049 | 0.8450 | 0.6680 |
| mean | heldout | 0.867353 | 1.23952 | 0.0125611 | 0.8207 | 0.6610 |
| flat | fit | 0.860714 | 0.343888 | 0.0126159 | 0.8551 | 0.6738 |
| flat | heldout | 0.867353 | 1.63581 | 0.0137025 | 0.7961 | 0.6506 |

Fisher candidates exclude fixed sink32 and exact recent64. The saved deployment tensors contain only the direct chunk encoder and the per-query-head query factor.
