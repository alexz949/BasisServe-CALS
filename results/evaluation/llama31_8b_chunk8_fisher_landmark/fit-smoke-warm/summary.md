# Llama-3.1-8B-Instruct Direct Chunk8 Fisher ALS Smoke

This smoke uses Dense V128, frozen Base16, audited 64K calibration windows, four ALS sweeps, and a final query closure solve.

| Feature | Split | Fisher NMSE init | Fisher NMSE final | Chunk rel-MSE final | Exact support recall | Routed-candidate mass |
|---|---|---:|---:|---:|---:|---:|
| mean | fit | 0.780609 | 0.461954 | 0.242406 | 0.7483 | 0.3733 |
| mean | heldout | 0.887347 | 0.91834 | 0.337639 | 0.7007 | 0.4245 |
| flat | fit | 0.780609 | 0.263213 | 0.267317 | 0.8169 | 0.3804 |
| flat | heldout | 0.887345 | 1.07618 | 0.336493 | 0.6769 | 0.4170 |

The objective masks the four fixed sink chunks and excludes exact recent64 tokens. Saved deployment factors contain only the direct chunk encoder and per-query-head factor; the old token R16 is initialization-only.
