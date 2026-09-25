# LongBench-v1 ShadowKV 9-task (>4K): allocated C1 V

9 tasks (narrativeqa 200, multifieldqa_en 110, hotpotqa 195, musique 200, dureader 200, gov_report 180, samsum 165, passage_retrieval_en 200, lcc 93); all 1543 included; shared V setting across arms.

| Task | full | ours | lrqk | shadowkv |
|---|---:|---:|---:|---:|
| narrativeqa | 28.0578 | 29.1597 | 27.9046 | 29.2199 |
| multifieldqa_en | 50.2504 | 51.6161 | 49.9961 | 51.0635 |
| hotpotqa | 52.2305 | 51.6608 | 52.1309 | 52.6392 |
| musique | 31.3150 | 31.1467 | 31.4589 | 31.4876 |
| dureader | 32.3360 | 29.5832 | 30.6322 | 31.1208 |
| gov_report | 32.7553 | 29.0970 | 30.9983 | 30.9910 |
| samsum | 44.4953 | 43.3446 | 45.3685 | 43.7421 |
| passage_retrieval_en | 99.5000 | 99.5000 | 99.5000 | 99.0000 |
| lcc | 52.2151 | 48.9462 | 51.6129 | 51.1828 |
| Mean (samples) | 46.6934 | 45.7361 | 46.2927 | 46.4051 |
| Mean (tasks) | 47.0173 | 46.0060 | 46.6225 | 46.7163 |
| LRQK 6-task mean | 51.2123 | 50.2773 | 50.8967 | 50.8665 |
