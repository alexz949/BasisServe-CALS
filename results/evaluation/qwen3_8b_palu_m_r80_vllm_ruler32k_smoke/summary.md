# Qwen3-8B Dense-K + PaLU M-LRD Fisher R80 RULER-v1 32K

| Task | Samples | Accuracy |
|:---|---:|---:|
| niah_single_1 | 1 | 100.00% |
| **Task-balanced mean** | 1 | **100.00%** |

Protocol: BF16 Qwen3-8B-Base, dense exact K, native vLLM FlashAttention, greedy decoding, and the same 11-task x 8-sample RULER-32K dataset used by the sparse-Key arms.

PaLU is reconstructed into standard dense V projection slots for a quality-only control; this run does not use a compact V cache.
