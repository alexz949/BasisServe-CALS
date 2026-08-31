# Qwen3-8B-Base Pairwise-KQ-QUEST + C1-V64 RULER-v1 4K

Prompt prefill and generation both use the 128-token transactional block schedule.

| Task | Samples | Dense-K+C1 | Pairwise full | Pairwise B512 | B512 vs full | Regressions | Improvements |
|:---|---:|---:|---:|---:|---:|---:|---:|
| niah_single_1 | 8 | 100.00% | 100.00% | 75.00% | -25.00 pp | 2 | 0 |
| niah_single_2 | 8 | 100.00% | 100.00% | 75.00% | -25.00 pp | 2 | 0 |
| niah_single_3 | 8 | 100.00% | 87.50% | 0.00% | -87.50 pp | 7 | 0 |
| niah_multikey_1 | 8 | 100.00% | 37.50% | 0.00% | -37.50 pp | 3 | 0 |
| niah_multikey_2 | 8 | 87.50% | 50.00% | 0.00% | -50.00 pp | 4 | 0 |
| niah_multikey_3 | 8 | 75.00% | 0.00% | 0.00% | +0.00 pp | 0 | 0 |
| niah_multivalue | 8 | 93.75% | 81.25% | 18.75% | -62.50 pp | 8 | 0 |
| niah_multiquery | 8 | 96.88% | 68.75% | 9.38% | -59.38 pp | 8 | 0 |
| vt | 8 | 100.00% | 57.50% | 17.50% | -40.00 pp | 7 | 0 |
| cwe | 8 | 92.50% | 16.25% | 16.25% | -0.00 pp | 3 | 5 |
| fwe | 8 | 79.17% | 41.67% | 25.00% | -16.67 pp | 2 | 1 |
| qa_1 | 8 | 75.00% | 50.00% | 12.50% | -37.50 pp | 3 | 0 |
| qa_2 | 8 | 62.50% | 37.50% | 37.50% | +0.00 pp | 0 | 0 |
| **Task-balanced mean** | 104 | **89.41%** | **55.99%** | **22.07%** | **-33.93 pp** | **49** | **6** |

Protocol: official RULER-v1 base prompts and substring scorers; greedy decoding; block length 128; page size 16; Pairwise-QUEST historical budget 512; layers 0 and 1 use full compressed-K support; resident C1-V64.

Dense-K+C1 is reused from the frozen matching RULER baseline. This remains a quality oracle: exact K is retained by DynamicCache and eager gather time is not serving latency.
