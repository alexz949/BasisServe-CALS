# Qwen3-8B-Base dense RULER-v1 32K

Unmodified dense K and dense V with greedy decoding.

| Task | Samples | Dense accuracy |
|:---|---:|---:|
| niah_single_1 | 8 | 100.00% |
| niah_single_2 | 8 | 100.00% |
| niah_single_3 | 8 | 100.00% |
| niah_multikey_1 | 8 | 87.50% |
| niah_multikey_2 | 8 | 100.00% |
| niah_multiquery | 8 | 93.75% |
| niah_multivalue | 8 | 100.00% |
| vt | 8 | 95.00% |
| fwe | 8 | 87.50% |
| qa_1 | 8 | 50.00% |
| qa_2 | 8 | 37.50% |
| **Task-balanced mean** | 88 | **86.48%** |

Protocol: official RULER-v1 base completion prompts and substring scorers; 8 examples per task; greedy decoding; BF16 dense SDPA.
