# Qwen3-32B C1 full-decode latency simulation

Batch `512`, context `1024`, TP`8`. Effective network: `alpha=20.0 us`, `beta=3.125 GB/s`.

| Path | p50 step ms | p95-component-sum ms | p50 simulated tok/s | Speedup vs dense |
|:---|---:|---:|---:|---:|
| dense_tp8 | 1571.88 | 1937.56 | 325.724 | 1x |
| barrier_big | 1458.45 | 1734.5 | 351.058 | 1.078x |
| barrier_wave_2 | 1459.21 | 1755.29 | 350.875 | 1.077x |
| barrier_wave_3 | 1459.79 | 1756.47 | 350.735 | 1.077x |
| barrier_partial_8 | 1462.66 | 1845.61 | 350.047 | 1.075x |
| comm_overlap_wave_2 | 1391.67 | 1683.85 | 367.903 | 1.129x |
| comm_overlap_wave_3 | 1390.92 | 1682.8 | 368.101 | 1.13x |
| comm_overlap_partial_8 | 1391.11 | 1764.5 | 368.051 | 1.13x |
| ideal_overlap_wave_2 | 1391.67 | 1673.79 | 367.903 | 1.129x |
| ideal_overlap_wave_3 | 1390.92 | 1666.62 | 368.101 | 1.13x |
| ideal_overlap_partial_8 | 1391.11 | 1702.29 | 368.051 | 1.13x |

GPU operator terms are measured; TP communication is alpha-beta modeled. P95 is a sum of component p95 values, not an empirical end-to-end latency quantile. Pipeline paths assume independent source progress; `ideal_overlap` additionally allows decoder GEMMs to overlap the receiver's local attention.
