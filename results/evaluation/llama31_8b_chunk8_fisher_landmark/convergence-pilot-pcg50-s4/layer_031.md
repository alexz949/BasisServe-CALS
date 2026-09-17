# Layer 31 Direct Chunk8 Fisher Fit

| Feature | Split | Fisher NMSE init | Fisher NMSE final | Chunk rel-MSE final | Exact support recall | Routed-candidate mass |
|---|---|---:|---:|---:|---:|---:|
| mean | fit | 0.774516 | 0.539547 | 0.0171885 | 0.7644 | 0.7445 |
| mean | heldout | 0.782462 | 0.600902 | 0.0175349 | 0.7667 | 0.7415 |
| flat | fit | 0.774516 | 0.475532 | 0.0171211 | 0.7614 | 0.7468 |
| flat | heldout | 0.782462 | 0.661438 | 0.0174841 | 0.7630 | 0.7386 |

Fisher candidates exclude fixed sink32 and exact recent64. The saved deployment tensors contain only the direct chunk encoder and the per-query-head query factor.
