# Dense-K + C1-V64 RULER block-schedule control

The frozen one-shot baseline is compared with exact Dense-K+C1-V64 using 128-token prompt blocks and the same greedy decode.

| Task | Samples | One-shot | Block-128 | Delta | Regressions | Improvements |
|:---|---:|---:|---:|---:|---:|---:|
| niah_single_1 | 8 | 100.00% | 100.00% | +0.00 pp | 0 | 0 |
| niah_single_2 | 8 | 100.00% | 100.00% | +0.00 pp | 0 | 0 |
| niah_single_3 | 8 | 100.00% | 100.00% | +0.00 pp | 0 | 0 |
| niah_multikey_1 | 8 | 100.00% | 100.00% | +0.00 pp | 0 | 0 |
| niah_multikey_2 | 8 | 87.50% | 87.50% | +0.00 pp | 0 | 0 |
| niah_multikey_3 | 8 | 75.00% | 75.00% | +0.00 pp | 0 | 0 |
| niah_multivalue | 8 | 93.75% | 93.75% | +0.00 pp | 0 | 0 |
| niah_multiquery | 8 | 96.88% | 96.88% | +0.00 pp | 0 | 0 |
| vt | 8 | 100.00% | 100.00% | +0.00 pp | 0 | 0 |
| cwe | 8 | 92.50% | 92.50% | +0.00 pp | 0 | 0 |
| fwe | 8 | 79.17% | 79.17% | +0.00 pp | 0 | 0 |
| qa_1 | 8 | 75.00% | 75.00% | +0.00 pp | 0 | 0 |
| qa_2 | 8 | 62.50% | 62.50% | +0.00 pp | 0 | 0 |
| **Task-balanced mean** | 104 | **89.41%** | **89.41%** | **+0.00 pp** | **0** | **0** |

This isolates transaction scheduling: both arms use exact K and the same resident C1-V64 checkpoint.
