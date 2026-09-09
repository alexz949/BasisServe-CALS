# Qwen3-8B Dense-K + C1-V80 RULER-v1 32K

| Task | Samples | Accuracy |
|:---|---:|---:|
| niah_single_1 | 8 | 100.00% |
| niah_single_2 | 8 | 100.00% |
| niah_single_3 | 8 | 100.00% |
| niah_multikey_1 | 8 | 87.50% |
| niah_multikey_2 | 8 | 87.50% |
| niah_multiquery | 8 | 96.88% |
| niah_multivalue | 8 | 93.75% |
| vt | 8 | 92.50% |
| fwe | 8 | 87.50% |
| qa_1 | 8 | 50.00% |
| qa_2 | 8 | 37.50% |
| **Task-balanced mean** | 88 | **84.83%** |

Protocol: BF16 Qwen3-8B-Base, dense exact K, native vLLM FlashAttention, greedy decoding, and the same 11-task x 8-sample RULER-32K dataset used by the sparse-Key arms.

C1-V80 is folded into standard dense V/O slots for a quality-only control; this run does not use a physically compact V cache.
