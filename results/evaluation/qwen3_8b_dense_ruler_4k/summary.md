# Qwen3-8B-Base dense RULER-v1 4K

Unmodified dense K and dense V with greedy decoding.

| Task | Samples | Dense accuracy |
|:---|---:|---:|
| niah_single_1 | 100 | 100.00% |
| niah_single_2 | 100 | 100.00% |
| niah_single_3 | 100 | 100.00% |
| niah_multikey_1 | 100 | 99.00% |
| niah_multikey_2 | 100 | 100.00% |
| niah_multikey_3 | 100 | 100.00% |
| niah_multivalue | 100 | 98.00% |
| niah_multiquery | 100 | 100.00% |
| vt | 100 | 100.00% |
| cwe | 100 | 100.00% |
| fwe | 100 | 94.00% |
| qa_1 | 100 | 89.00% |
| qa_2 | 100 | 60.00% |
| **Task-balanced mean** | 1300 | **95.38%** |

Protocol: official RULER-v1 base completion prompts and substring scorers; 100 examples per task; greedy decoding; BF16 dense SDPA.
