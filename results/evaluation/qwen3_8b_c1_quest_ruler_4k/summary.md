# Qwen3-8B-Base C1-V64 + QUEST RULER-v1 4K

Dense-K and physical-shared QUEST use the same C1-V64 ALS5 Value checkpoint and identical greedy prompts.

| Task | Samples | Dense-K + C1-V64 | Physical-shared-1024 | Delta | Regressions | Improvements |
|:---|---:|---:|---:|---:|---:|---:|
| niah_single_1 | 100 | 100.00% | 100.00% | +0.00 pp | 0 | 0 |
| niah_single_2 | 100 | 100.00% | 100.00% | +0.00 pp | 0 | 0 |
| niah_single_3 | 100 | 100.00% | 98.00% | -2.00 pp | 2 | 0 |
| niah_multikey_1 | 100 | 98.00% | 98.00% | +0.00 pp | 0 | 0 |
| niah_multikey_2 | 100 | 99.00% | 99.00% | +0.00 pp | 0 | 0 |
| niah_multikey_3 | 100 | 92.00% | 81.00% | -11.00 pp | 11 | 0 |
| niah_multivalue | 100 | 96.25% | 98.25% | +2.00 pp | 1 | 6 |
| niah_multiquery | 100 | 97.50% | 97.25% | -0.25 pp | 3 | 2 |
| vt | 100 | 98.60% | 98.60% | +0.00 pp | 2 | 2 |
| cwe | 100 | 88.40% | 73.50% | -14.90 pp | 74 | 9 |
| fwe | 100 | 85.33% | 79.33% | -6.00 pp | 21 | 3 |
| qa_1 | 100 | 78.00% | 77.00% | -1.00 pp | 1 | 0 |
| qa_2 | 100 | 58.00% | 58.00% | +0.00 pp | 0 | 0 |
| **Task-balanced mean** | 1300 | **91.62%** | **89.07%** | **-2.55 pp** | **115** | **22** |

Protocol: official RULER-v1 base completion prompts and substring scorers; 100 examples per task; greedy decoding; dense prompt prefill; QUEST enabled for decode after the first generated token; page size 16; fixed 1024-token budget; layers 0 and 1 exact.

This is a quality oracle. Exact K remains GPU resident and QUEST metadata is rebuilt in Python, so elapsed time is not an offload or serving-throughput result.
