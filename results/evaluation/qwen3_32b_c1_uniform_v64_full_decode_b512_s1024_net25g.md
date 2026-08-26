# Qwen3-32B C1 full-decode latency simulation

Batch `512`, context `1024`, TP`8`. Effective network: `alpha=20.0 us`, `beta=3.125 GB/s`.

| Path | p50 step ms | p95-component-sum ms | p50 simulated tok/s | Speedup vs dense |
|:---|---:|---:|---:|---:|
| dense_tp8 | 1597.09 | 1945.22 | 320.583 | 1x |
| barrier_big | 1271.6 | 1602.24 | 402.642 | 1.256x |
| barrier_wave_2 | 1272.33 | 1602.92 | 402.41 | 1.255x |
| barrier_wave_3 | 1272.93 | 1603.54 | 402.221 | 1.255x |
| barrier_partial_8 | 1275.98 | 1728.5 | 401.261 | 1.252x |
| comm_overlap_wave_2 | 1207.91 | 1538.5 | 423.873 | 1.322x |
| comm_overlap_wave_3 | 1208.51 | 1539.11 | 423.663 | 1.322x |
| comm_overlap_partial_8 | 1211.55 | 1664.07 | 422.598 | 1.318x |
| ideal_overlap_wave_2 | 1207.91 | 1538.5 | 423.873 | 1.322x |
| ideal_overlap_wave_3 | 1208.51 | 1539.11 | 423.663 | 1.322x |
| ideal_overlap_partial_8 | 1211.55 | 1664.07 | 422.598 | 1.318x |

GPU operator terms are measured; TP communication is alpha-beta modeled. P95 is a sum of component p95 values, not an empirical end-to-end latency quantile. Pipeline paths assume independent source progress; `ideal_overlap` additionally allows decoder GEMMs to overlap the receiver's local attention.
