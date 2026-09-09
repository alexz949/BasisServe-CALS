# Qwen3-8B C1-V80 loki RULER-v1 32K

| Task | Samples | Accuracy |
|:---|---:|---:|
| niah_single_1 | 8 | 100.00% |
| niah_single_2 | 8 | 100.00% |
| niah_single_3 | 8 | 100.00% |
| niah_multikey_1 | 8 | 87.50% |
| niah_multikey_2 | 8 | 37.50% |
| niah_multiquery | 8 | 81.25% |
| niah_multivalue | 8 | 87.50% |
| vt | 8 | 87.50% |
| fwe | 8 | 83.33% |
| qa_1 | 8 | 37.50% |
| qa_2 | 8 | 37.50% |
| **Task-balanced mean** | 88 | **76.33%** |

- Mean physical exact-K tokens per layer/query: `30950.83`.
- Mean physical exact-K traffic per layer/query: `7.556 MiB` at BF16 K128.
- Mean nominal per-Q-head token visits per layer/query: `65536.00`.

Protocol: BF16 Qwen3-8B-Base, uniform C1-V80 calibrated on 32 C4 documents x 32768 positions, dense exact-K FlashAttention prefill, then native vLLM paged sparse exact-K decode. The first generated token comes from dense prefill. Exact K remains GPU-resident in this quality runtime.
