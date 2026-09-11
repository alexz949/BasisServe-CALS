# Qwen3-8B C1 two-sided KL average V96: full LongBench-v1

All 21 tasks and all samples. Shared BF16 C1 prefill. Base16/R16: 64 x 32K fit, 16 x 32K validation, fit-only Query-Gram Q32.
Scores use official task metrics (0–100); overall mean is the arithmetic mean of 21 task scores.
LRQK uses FP32 routing state and per-query top-k + recent64; its physical GQA union is not a hard 2048-token budget.

| Task | N | Full K | Base16 + R16 | LRQK |
|---|---:|---:|---:|---:|
| narrativeqa | 200 | 24.7449 | 24.4655 | 24.6413 |
| qasper | 200 | 30.4261 | 30.6685 | 30.7706 |
| multifieldqa_en | 150 | 50.6087 | 51.6211 | 50.8868 |
| multifieldqa_zh | 200 | 54.0958 | 55.2956 | 54.4523 |
| hotpotqa | 200 | 49.4278 | 49.2709 | 49.7611 |
| 2wikimqa | 200 | 40.0845 | 40.3143 | 40.0845 |
| musique | 200 | 28.1228 | 27.1272 | 28.2062 |
| dureader | 200 | 30.5586 | 29.2310 | 29.9312 |
| gov_report | 200 | 32.1002 | 31.5012 | 31.6008 |
| qmsum | 200 | 25.1655 | 25.1858 | 25.1092 |
| multi_news | 200 | 26.0912 | 26.0344 | 25.9913 |
| vcsum | 200 | 16.2598 | 16.0581 | 16.0047 |
| trec | 200 | 75.5000 | 75.5000 | 75.5000 |
| triviaqa | 200 | 91.4897 | 91.4897 | 91.6563 |
| samsum | 200 | 45.4749 | 45.6101 | 46.1230 |
| lsht | 200 | 44.0000 | 43.5000 | 44.0000 |
| passage_count | 200 | 4.0000 | 3.0000 | 3.5000 |
| passage_retrieval_en | 200 | 83.9167 | 84.2500 | 83.0000 |
| passage_retrieval_zh | 200 | 81.2500 | 79.7500 | 80.5000 |
| lcc | 500 | 67.3740 | 67.4720 | 67.0880 |
| repobench-p | 500 | 57.2520 | 57.9140 | 57.5860 |
| Mean | | 45.6163 | 45.4885 | 45.5425 |
