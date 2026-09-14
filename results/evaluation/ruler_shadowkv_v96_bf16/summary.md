# ShadowKV K + resident C1-V96: L40S BF16 RULER32K

11 tasks x8 reused prompts; model/K/V and stored SVD factors BF16; SVD solve FP32. Full C1 Triton prefill in both arms.
ShadowKV rank160 across concatenated KV heads, chunk8, selected2048 plus exact48 outlier chunks, prompt local tail and all generated tokens.

| Task | Exact K | ShadowKV |
|---|---:|---:|
| niah_single_1 | 100.0000 | 100.0000 |
| niah_single_2 | 100.0000 | 100.0000 |
| niah_single_3 | 100.0000 | 100.0000 |
| niah_multikey_1 | 87.5000 | 87.5000 |
| niah_multikey_2 | 100.0000 | 100.0000 |
| niah_multiquery | 100.0000 | 96.8750 |
| niah_multivalue | 100.0000 | 96.8750 |
| vt | 100.0000 | 92.5000 |
| fwe | 91.6667 | 91.6667 |
| qa_1 | 50.0000 | 50.0000 |
| qa_2 | 37.5000 | 37.5000 |
| Mean | 87.8788 | 86.6288 |

Physical union: {"scope": "last decode step per prompt with at least one decode; zero-decode prompts excluded; not all-step mean", "mean": 2499.5402298850577, "minimum": 2467, "maximum": 2597, "zero_decode_prompts": 1}
Paired: {"improved": 0, "regressed": 5}

Environment and commands: docs/shadowkv_v96_protocol.md. Per-record commands preserved.
