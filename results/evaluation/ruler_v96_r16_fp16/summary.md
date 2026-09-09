# RULER32K: C1-V96/Base16/R16 versus exact K and LRQK

Same88 prompts,11 tasks, V100 FP16 C1-V96 full prefill and decode. Ours: full-window Query-Gram Q32, Page32/B2048, pinned page0.
LRQK: FP32 routing state, R32/k1152/recent64. Frozen offline R16 bank reused; no fitting in this experiment.

| Task | Full K | LRQK | Ours R16 |
|---|---:|---:|---:|
| niah_single_1 | 100.0000 | 100.0000 | 100.0000 |
| niah_single_2 | 100.0000 | 100.0000 | 100.0000 |
| niah_single_3 | 100.0000 | 100.0000 | 100.0000 |
| niah_multikey_1 | 87.5000 | 87.5000 | 87.5000 |
| niah_multikey_2 | 100.0000 | 100.0000 | 100.0000 |
| niah_multiquery | 96.8750 | 100.0000 | 96.8750 |
| niah_multivalue | 100.0000 | 100.0000 | 100.0000 |
| vt | 97.5000 | 95.0000 | 92.5000 |
| fwe | 91.6667 | 91.6667 | 83.3333 |
| qa_1 | 50.0000 | 50.0000 | 50.0000 |
| qa_2 | 37.5000 | 37.5000 | 37.5000 |
| Mean | 87.3674 | 87.4242 | 86.1553 |

Paired changes: {"full": {"improved": 1, "regressed": 5}, "k1152": {"improved": 0, "regressed": 4}}

Environment: basis. Commands: docs/ruler_v96_r16_protocol.md.
