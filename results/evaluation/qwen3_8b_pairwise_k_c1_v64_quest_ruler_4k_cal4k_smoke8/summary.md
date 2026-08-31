# Qwen3-8B-Base Pairwise-KQ-QUEST + C1-V64 RULER-v1 4K

Prompt prefill and generation both use the 128-token transactional block schedule.

| Task | Samples | Dense-K+C1 | Pairwise full | Pairwise B512 | B512 vs full | Regressions | Improvements |
|:---|---:|---:|---:|---:|---:|---:|---:|
| niah_single_1 | 8 | 100.00% | 100.00% | 87.50% | -12.50 pp | 1 | 0 |
| niah_single_2 | 8 | 100.00% | 100.00% | 87.50% | -12.50 pp | 1 | 0 |
| niah_single_3 | 8 | 100.00% | 100.00% | 0.00% | -100.00 pp | 8 | 0 |
| niah_multikey_1 | 8 | 100.00% | 100.00% | 25.00% | -75.00 pp | 6 | 0 |
| niah_multikey_2 | 8 | 87.50% | 100.00% | 0.00% | -100.00 pp | 8 | 0 |
| niah_multikey_3 | 8 | 75.00% | 37.50% | 0.00% | -37.50 pp | 3 | 0 |
| niah_multivalue | 8 | 93.75% | 96.88% | 31.25% | -65.62 pp | 8 | 0 |
| niah_multiquery | 8 | 96.88% | 100.00% | 31.25% | -68.75 pp | 8 | 0 |
| vt | 8 | 100.00% | 90.00% | 67.50% | -22.50 pp | 6 | 1 |
| cwe | 8 | 92.50% | 26.25% | 12.50% | -13.75 pp | 5 | 0 |
| fwe | 8 | 79.17% | 62.50% | 33.33% | -29.17 pp | 4 | 1 |
| qa_1 | 8 | 75.00% | 50.00% | 37.50% | -12.50 pp | 1 | 0 |
| qa_2 | 8 | 62.50% | 62.50% | 62.50% | +0.00 pp | 0 | 0 |
| **Task-balanced mean** | 104 | **89.41%** | **78.89%** | **36.60%** | **-42.29 pp** | **59** | **2** |

Protocol: official RULER-v1 base prompts and substring scorers; greedy decoding; block length 128; page size 16; Pairwise-QUEST historical budget 512; layers 0 and 1 use full compressed-K support; resident C1-V64.

Dense-K+C1 is reused from the frozen matching RULER baseline. This remains a quality oracle: exact K is retained by DynamicCache and eager gather time is not serving latency.
