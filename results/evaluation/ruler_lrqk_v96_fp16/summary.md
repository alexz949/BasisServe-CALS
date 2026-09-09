# LRQK RULER32K: V100 FP16 C1-V96

11 tasks x8 reused prompts; model/K/V FP16, LRQK routing FP32. Full C1 prefill in both arms. No V128 baseline.
LRQK R32, per-head k1152 plus recent64; not Page32 and not a hard physical B2048 cap.

| Task | Exact K | LRQK k1152 |
|---|---:|---:|
| niah_single_1 | 100.0000 | 100.0000 |
| niah_single_2 | 100.0000 | 100.0000 |
| niah_single_3 | 100.0000 | 100.0000 |
| niah_multikey_1 | 87.5000 | 87.5000 |
| niah_multikey_2 | 100.0000 | 100.0000 |
| niah_multiquery | 96.8750 | 100.0000 |
| niah_multivalue | 100.0000 | 100.0000 |
| vt | 97.5000 | 95.0000 |
| fwe | 91.6667 | 91.6667 |
| qa_1 | 50.0000 | 50.0000 |
| qa_2 | 37.5000 | 37.5000 |
| Mean | 87.3674 | 87.4242 |

Physical union: {"scope": "last decode step per prompt; all layers/groups, not all-step mean", "mean": 2661.90289614899, "minimum": 1545, "maximum": 4464}
Paired: {"improved": 1, "regressed": 1}

Environment and commands: docs/lrqk_ruler_v96_protocol.md. Per-record commands preserved.
