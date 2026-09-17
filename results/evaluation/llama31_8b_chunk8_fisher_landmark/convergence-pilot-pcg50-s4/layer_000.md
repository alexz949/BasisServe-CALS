# Layer 0 Direct Chunk8 Fisher Fit

| Feature | Split | Fisher NMSE init | Fisher NMSE final | Chunk rel-MSE final | Exact support recall | Routed-candidate mass |
|---|---|---:|---:|---:|---:|---:|
| mean | fit | 0.874159 | 0.655725 | 0.272184 | 0.8091 | 0.5788 |
| mean | heldout | 0.876847 | 0.678296 | 0.267666 | 0.8130 | 0.5812 |
| flat | fit | 0.874159 | 0.623054 | 0.27576 | 0.8082 | 0.5788 |
| flat | heldout | 0.876846 | 0.699176 | 0.270965 | 0.8117 | 0.5809 |

Fisher candidates exclude fixed sink32 and exact recent64. The saved deployment tensors contain only the direct chunk encoder and the per-query-head query factor.
