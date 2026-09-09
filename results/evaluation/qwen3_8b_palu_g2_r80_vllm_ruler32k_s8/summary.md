# Qwen3-8B Dense-K + PaLU G2-LRD Fisher R80 RULER-v1 32K

| Task | Samples | Accuracy |
|:---|---:|---:|
| niah_single_1 | 8 | 100.00% |
| niah_single_2 | 8 | 87.50% |
| niah_single_3 | 8 | 100.00% |
| niah_multikey_1 | 8 | 62.50% |
| niah_multikey_2 | 8 | 37.50% |
| niah_multiquery | 8 | 87.50% |
| niah_multivalue | 8 | 81.25% |
| vt | 8 | 90.00% |
| fwe | 8 | 95.83% |
| qa_1 | 8 | 12.50% |
| qa_2 | 8 | 37.50% |
| **Task-balanced mean** | 88 | **72.01%** |

Protocol: BF16 Qwen3-8B-Base, dense exact K, native vLLM FlashAttention, greedy decoding, and the same 11-task x 8-sample RULER-32K dataset used by the sparse-Key arms.

PaLU is reconstructed into standard dense V projection slots for a quality-only control; this run does not use a compact V cache.
