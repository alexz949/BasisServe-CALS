# Qwen3-8B-Base dense RULER-v1 64K

Unmodified dense K and dense V with greedy decoding.

| Task | Samples | Dense accuracy |
|:---|---:|---:|
| niah_multikey_2 | 1 | 100.00% |
| niah_multivalue | 1 | 75.00% |
| niah_single_2 | 1 | 100.00% |
| niah_single_3 | 1 | 100.00% |
| fwe | 1 | 100.00% |
| **Task-balanced mean** | 5 | **95.00%** |

Protocol: official RULER-v1 base completion prompts and substring scorers; 8 examples per task; greedy decoding; BF16 dense SDPA.
