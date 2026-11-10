# Decode-step GPU time by category (mean over TP ranks)

Profiled run: one trial, 32 decode steps after 16 conditioning steps, eager. The torch profiler adds host overhead, so the profiled step is slower than the unprofiled grid cell. NCCL time includes waiting for peer ranks.

| Arm | Batch | Profiled step p50 ms | GPU busy ms/step | NCCL ms/step (mean) | NCCL per rank | Attention ms/step | GEMM ms/step | Other ms/step |
|---|---|---|---|---|---|---|---|---|
| `dense` | 32 | 46.0 | 38.7 | 25.1 | 20.2 / 27.1 / 24.1 / 29.2 | 6.6 | 6.2 | 0.7 |
| `palu_g4_r96` | 32 | 42.6 | 39.4 | 22.4 | 22.9 / 21.6 / 22.3 / 22.7 | 9.4 | 7.0 | 0.6 |
| `palu_g4_sharded_r96` | 32 | 110.4 | 106.5 | 87.0 | 87.5 / 86.2 / 86.9 / 87.5 | 11.4 | 7.0 | 1.1 |
