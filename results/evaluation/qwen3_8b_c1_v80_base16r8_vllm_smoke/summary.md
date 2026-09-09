# Qwen3-8B C1-V80 conditional RULER-v1 32K

| Task | Samples | Accuracy |
|:---|---:|---:|
| niah_single_1 | 1 | 100.00% |
| **Task-balanced mean** | 1 | **100.00%** |

- Mean physical exact-K tokens per layer/query: `16333.92`.
- Mean physical exact-K traffic per layer/query: `3.988 MiB` at BF16 K128.
- Mean nominal per-Q-head token visits per layer/query: `65536.00`.

Protocol: BF16 Qwen3-8B-Base, uniform C1-V80 calibrated on 32 C4 documents x 32768 positions, dense exact-K FlashAttention prefill, then native vLLM paged sparse exact-K decode. The first generated token comes from dense prefill. Exact K remains GPU-resident in this quality runtime.
