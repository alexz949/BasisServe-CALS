# Layer 15 Direct Chunk8 Fisher Fit

| Feature | Split | Fisher NMSE init | Fisher NMSE final | Chunk rel-MSE final | Exact support recall | Routed-candidate mass |
|---|---|---:|---:|---:|---:|---:|
| mean | fit | 0.817127 | 0.511152 | 0.027043 | 0.7326 | 0.7860 |
| mean | heldout | 0.837499 | 0.565528 | 0.0273161 | 0.7156 | 0.7541 |
| flat | fit | 0.817127 | 0.423277 | 0.0261578 | 0.7311 | 0.7906 |
| flat | heldout | 0.837499 | 0.644863 | 0.0262322 | 0.7138 | 0.7502 |

Fisher candidates exclude fixed sink32 and exact recent64. The saved deployment tensors contain only the direct chunk encoder and the per-query-head query factor.
