# Llama-3.1-8B-Instruct RULER 128k: original V128

11 tasks × 100 identical prompts per arm; greedy generation with native EOS and official caps.
Dense uses TP2. Routing arms use four independent GPUs. Native single-GPU Full-K checks cover all observed prefill first-token differences against TP2.

| Task | Dense | B16R16 | ShadowKV | LRQK | Loki |
|---|---:|---:|---:|---:|---:|
| niah_single_1 | 100.0000 | 100.0000 | 100.0000 | 100.0000 | 84.0000 |
| niah_single_2 | 100.0000 | 100.0000 | 100.0000 | 100.0000 | 97.0000 |
| niah_single_3 | 99.0000 | 100.0000 | 100.0000 | 100.0000 | 0.0000 |
| niah_multikey_1 | 99.0000 | 99.0000 | 99.0000 | 99.0000 | 98.0000 |
| niah_multikey_2 | 89.0000 | 80.0000 | 76.0000 | 83.0000 | 31.0000 |
| niah_multiquery | 98.7500 | 97.7500 | 97.2500 | 98.5000 | 45.2500 |
| niah_multivalue | 94.2500 | 88.5000 | 89.5000 | 92.2500 | 48.0000 |
| vt | 76.2000 | 73.0000 | 70.2000 | 64.2000 | 39.0000 |
| fwe | 74.6667 | 69.6667 | 69.3333 | 58.0000 | 33.6667 |
| qa_1 | 78.0000 | 80.0000 | 79.0000 | 80.0000 | 64.0000 |
| qa_2 | 46.0000 | 47.0000 | 46.0000 | 46.0000 | 34.0000 |
| 11-task mean | 86.8061 | 84.9924 | 84.2076 | 83.7227 | 52.1742 |
| 10-task mean excluding single_3 | 85.5867 | 83.4917 | 82.6283 | 82.0950 | 57.3917 |

B16R16: physical budget2048 including sink32/recent64. LRQK: top832+recent64. Loki: top856/recent0, Wikipedia PCA32.
ShadowKV: official CPU-offload, routed2048 plus outliers/local/generated. Per-query top is not a hard physical KV-group cap.
Prefill audit: all 27 TP2/single-GPU first-token differences plus two controls checked; native Full-K matched routing prefill in all 29.
Environment: basis. Formal command: `python -m evaluation.run_llama_ruler100 evaluate`. Full commands are in logs/.
