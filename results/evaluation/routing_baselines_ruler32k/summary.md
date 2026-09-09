# C1-V80 routing comparison: RULER 32K

Same 88 previously used prompts, full C1 prefill, shared first token, 36 sparse layers, BF16 on L40S.
Uniform R8 uses the existing Q8 Base16 and Q16 Page-Fisher residual. Loki uses the existing C4 64x32K PCA checkpoint.
QUEST uses raw-bound GQA max. Loki-page uses normalized Page-LSE GQA max and is an adaptation of Loki.
The first three sparse methods use 64 Page32 including page0. Loki-token selects 2048 tokens independently per query head; its physical budget is larger.
These are C1 selector controls, not full reproductions of the original papers. Exact K stays on GPU; elapsed time is not an optimized offload benchmark.

| Task | Exact K | Base16+R8 | QUEST-page | Loki-page R32 | Loki-token R32 |
|---|---:|---:|---:|---:|---:|
| niah_single_1 | 100.0000% | 100.0000% | 100.0000% | 100.0000% | 100.0000% |
| niah_single_2 | 100.0000% | 100.0000% | 50.0000% | 87.5000% | 100.0000% |
| niah_single_3 | 100.0000% | 100.0000% | 0.0000% | 25.0000% | 100.0000% |
| niah_multikey_1 | 87.5000% | 87.5000% | 25.0000% | 75.0000% | 87.5000% |
| niah_multikey_2 | 87.5000% | 50.0000% | 0.0000% | 37.5000% | 50.0000% |
| niah_multiquery | 96.8750% | 96.8750% | 9.3750% | 31.2500% | 81.2500% |
| niah_multivalue | 93.7500% | 93.7500% | 12.5000% | 25.0000% | 96.8750% |
| vt | 92.5000% | 90.0000% | 70.0000% | 90.0000% | 92.5000% |
| fwe | 91.6667% | 79.1667% | 12.5000% | 75.0000% | 83.3333% |
| qa_1 | 50.0000% | 50.0000% | 50.0000% | 50.0000% | 50.0000% |
| qa_2 | 37.5000% | 37.5000% | 50.0000% | 37.5000% | 37.5000% |
| Task-balanced mean | 85.2083% | 80.4356% | 34.4886% | 57.6136% | 79.9053% |

| Method | Mean selected physical tokens / KV group | Peak routing metadata, MiB |
|---|---:|---:|
| uniform_r8 | 2033.688 | 2447.851 |
| quest_page | 2045.969 | 144.000 |
| loki_page_r32 | 2038.433 | 575.965 |
| loki_token_r32 | 3865.413 | 575.965 |

Routing storage above reports actual materialized caches: our Base128+R8, Loki R32, or QUEST min/max. It does not equate the deployed residual-only R8 storage with this oracle's materialized Base.
Eight examples per task do not establish small accuracy differences reliably. No configuration was selected using these new results.
