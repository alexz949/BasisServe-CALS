# Llama-3.1-8B-Instruct Direct Chunk8 Fisher ALS Smoke

This smoke uses Dense V128, frozen Base16, audited 64K calibration windows, four ALS sweeps, and a final query closure solve.

| Feature | Split | Fisher NMSE init | Fisher NMSE final | Chunk rel-MSE final | Exact support recall | Routed-candidate mass |
|---|---|---:|---:|---:|---:|---:|
| mean | fit | 0.780609 | 0.464131 | 0.242106 | 0.7477 | 0.3730 |
| mean | heldout | 0.887347 | 0.916387 | 0.282015 | 0.6942 | 0.4233 |
| flat | fit | 0.780609 | 0.277661 | 0.299431 | 0.8128 | 0.3800 |
| flat | heldout | 0.887345 | 1.07233 | 0.339124 | 0.6756 | 0.4177 |

The objective masks the four fixed sink chunks and excludes exact recent64 tokens. Saved deployment factors contain only the direct chunk encoder and per-query-head factor; the old token R16 is initialization-only.
