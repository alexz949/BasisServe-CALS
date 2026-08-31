# Qwen3-32B C1 full-decode latency simulation

Batch `256`, context `4096`, TP`8`. Effective network: `alpha=20.0 us`, `beta=3.125 GB/s`.

| Path | p50 step ms | p95-component-sum ms | p50 simulated tok/s | Speedup vs dense |
|:---|---:|---:|---:|---:|
| dense_tp8 | 1442.4 | 1459.42 | 177.482 | 1x |
| barrier_big | 1304.21 | 1319.2 | 196.287 | 1.106x |
| barrier_wave_2 | 1305.08 | 1320.05 | 196.156 | 1.105x |
| barrier_wave_3 | 1305.87 | 1320.86 | 196.037 | 1.105x |
| barrier_partial_8 | 1309.8 | 1324.79 | 195.45 | 1.101x |
| comm_overlap_wave_2 | 1270.87 | 1285.76 | 201.437 | 1.135x |
| comm_overlap_wave_3 | 1270.13 | 1284.98 | 201.555 | 1.136x |
| comm_overlap_partial_8 | 1270.18 | 1285.05 | 201.546 | 1.136x |
| ideal_overlap_wave_2 | 1270.87 | 1285.76 | 201.437 | 1.135x |
| ideal_overlap_wave_3 | 1270.13 | 1284.98 | 201.555 | 1.136x |
| ideal_overlap_partial_8 | 1270.01 | 1284.84 | 201.573 | 1.136x |

GPU operator terms are measured; TP communication is alpha-beta modeled. P95 is a sum of component p95 values, not an empirical end-to-end latency quantile. Pipeline paths assume independent source progress; `ideal_overlap` additionally allows decoder GEMMs to overlap the receiver's local attention.
