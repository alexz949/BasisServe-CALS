# TP1 Llama-3.1-8B BasisKV full-model benchmark

B1 teacher-forced decode with exact Flash-SDPA prefill, Dense V128 and original W_O. BasisKV uses B16R16, Page32 and a 2,048-token physical support (62 routed pages + the exact recent 64 tokens). Values are full-model decode latency, not an isolated attention-kernel latency.

Each cell aggregates two 128-token measured runs. The first 16 decode tokens in each run are warmup, with ABBA mode ordering on a dedicated GPU lane.

- Environment: `basis` conda environment, PyTorch 2.13.0, 8x NVIDIA L40S
- Command: `conda run -n basis python benchmarks/system/run_tp1_sparse_full_v6.py --warmup-steps 16 --measure-steps 128 --rerun`
- Failures: none (32/32 formal trials completed; all logits finite)

| Context | Storage | Legacy ms/token | Optimized ms/token | Reduction | Optimized tok/s |
|---:|:---|---:|---:|---:|---:|
| 16K | local | 31.225 | 29.705 | 4.87% | 33.61 |
| 16K | offload | 34.850 | 32.689 | 6.20% | 30.57 |
| 32K | local | 33.548 | 31.156 | 7.13% | 32.08 |
| 32K | offload | 37.152 | 34.260 | 7.78% | 29.17 |
| 64K | local | 38.049 | 34.320 | 9.80% | 29.14 |
| 64K | offload | 41.564 | 37.149 | 10.62% | 26.90 |
| 128K | local | 47.485 | 40.347 | 15.03% | 24.79 |
| 128K | offload | 51.572 | 43.592 | 15.47% | 22.93 |

## Correctness and scope

The preceding 8K smoke validated all 32 layer outputs against a PyTorch exact attention calculation on the identical selected support. Formal runs use fixed teacher-forced inputs; every run records finiteness and output argmax tokens. This matrix is the matched BasisKV before/after experiment. Dense-local, naive dense-offload, ShadowKV and LRQK remain separate systems baselines.

Both repeats produced identical argmax sequences within each mode. Legacy and optimized argmax differed on 0/144 tokens at 16K and 128K, and 1/144 tokens at 32K and 64K; local and offload showed the same pattern. This is a minor BF16 difference between the reference append and fused append paths, while fixed teacher forcing keeps benchmark inputs identical.
