# Decode-step GPU time by category (mean over TP ranks), L40S node 2

Profiled run: one trial, 32 decode steps after 16 conditioning steps, eager, `NCCL_P2P_DISABLE=1`. The torch
profiler adds host overhead, so the profiled step is slower than the unprofiled grid cell (dense batch 32:
34.7 ms profiled, 22.10 ms in the grid). NCCL time includes waiting for peer ranks, which is why rank 0 (the
slowest rank to reach each collective) shows the least NCCL time.

Categories are kernel names from the Chrome traces in `*/p4096_b32/profiles/rank*.json.gz`:
NCCL = `ncclDev*`; attention = FlashAttention split-KV and combine, PaLU `kernel_unified_attention_diffkv` and
`kernel_reduce_segments_diffkv`, sharded `_decode_scores`, `_apply_weights` and its softmax; GEMM = CUTLASS and
cuBLAS GEMMs and split-K reduction; other = everything else. Per-step values are trace totals divided by 32.

| Arm | Batch | Profiled step p50 ms | GPU busy ms/step | NCCL ms/step (mean) | NCCL per rank | Attention ms/step | GEMM ms/step | Other ms/step |
|---|---|---|---|---|---|---|---|---|
| `dense` | 32 | 34.7 | 28.5 | 15.9 | 7.9 / 17.0 / 20.1 / 18.7 | 6.4 | 5.7 | 0.5 |
| `palu_g4_r96` | 32 | 34.7 | 29.7 | 14.5 | 7.4 / 14.7 / 17.0 / 18.6 | 8.0 | 6.7 | 0.4 |
| `palu_g4_sharded_r96` | 32 | 58.7 | 48.0 | 32.6 | 19.2 / 40.1 / 35.6 / 35.6 | 7.6 | 7.1 | 0.7 |

Reference (A5000, same protocol): dense 46.0 / 38.7 / 25.1 NCCL; `palu_g4_r96` 42.6 / 39.4 / 22.4;
`palu_g4_sharded_r96` 110.4 / 106.5 / 87.0 (profiled p50 / GPU busy / NCCL, ms per step).
