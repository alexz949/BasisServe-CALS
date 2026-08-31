# Qwen3-8B-Base dense RULER-v1 4K

Unmodified dense K and dense V with greedy decoding.

| Task | Samples | Dense accuracy |
|:---|---:|---:|
| niah_single_1 | 1 | 100.00% |
| **Task-balanced mean** | 1 | **100.00%** |

Protocol: official RULER-v1 base completion prompts and substring scorers; 100 examples per task; greedy decoding; BF16 dense SDPA.
