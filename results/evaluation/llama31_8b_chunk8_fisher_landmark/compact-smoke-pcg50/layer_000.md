# Layer 0 Direct Chunk8 Fisher Fit

| Feature | Split | Fisher NMSE init | Fisher NMSE final | Chunk rel-MSE final | Exact support recall | Routed-candidate mass |
|---|---|---:|---:|---:|---:|---:|
| mean | fit | 0.768285 | 0.427011 | 0.255441 | 0.7341 | 0.4047 |
| mean | heldout | 0.892611 | 0.831647 | 0.232602 | 0.7014 | 0.4225 |
| flat | fit | 0.768285 | 0.236634 | 0.291455 | 0.8096 | 0.4135 |
| flat | heldout | 0.89261 | 1.15942 | 0.269761 | 0.6723 | 0.4148 |

Fisher candidates exclude fixed sink32 and exact recent64. The saved deployment tensors contain only the direct chunk encoder and the per-query-head query factor.
