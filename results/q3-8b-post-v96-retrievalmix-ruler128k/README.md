# q3-8b-post-v96-retrievalmix-ruler128k

Compact per-sample predictions (prompt token ids stripped; the prompt hash, prediction, score, routing statistics and timing are kept). Full protocol per arm is in `protocols/`; paired comparison tables in `summaries/`.

2026-09-24/25 additions: page-1 and page-4 Page-Fisher refits (no sink, 2048 + recent 64 = 2112) on all 1100 prompts; page-8 diagnostics (page-32 bank decoded at page 8 with one 8-token sink page on niah_multikey_2 + fwe; page-8 Page-Fisher refit, no sink, on niah_multikey_2 + fwe and on vt + qa_1 + qa_2); page-1 refit decoded with a 32-token pinned sink on fwe. Original arms: page 32, sink 32, recent 64 inside 2048.

| arm | samples | tasks | task-balanced mean |
|---|---:|---:|---:|
| b16r16_page1_fisher | 1100 | 11 | 81.43 |
| b16r16_page1_fisher_sink32_diag_fwe | 100 | 1 | 80.00 |
| b16r16_page4_fisher | 1100 | 11 | 80.58 |
| b16r16_page8_decode_page32bank_diag_mk2_fwe | 200 | 2 | 65.17 |
| b16r16_page8_fisher_diag_mk2_fwe | 200 | 2 | 67.83 |
| b16r16_page8_fisher_diag_vt_qa | 300 | 3 | 58.47 |
| b16r16_page_fisher | 1100 | 11 | 79.27 |
| b16r16_score_mse | 1100 | 11 | 81.49 |
| b8r24_page_fisher | 1100 | 11 | 81.38 |
| b8r24_score_mse | 1100 | 11 | 81.51 |
| full | 1100 | 11 | 82.58 |
| loki_ab200_post_rope | 200 | 2 | 61.00 |
| loki_ab200_pre_rope | 200 | 2 | 4.50 |
| loki_post_rope | 1100 | 11 | 73.59 |
| lrqk | 1100 | 11 | 80.98 |
| shadowkv | 1100 | 11 | 76.65 |

Protocol excerpt (first arm):

```json
{
 "sequence_length": 131072,
 "ours": "Base16/Residual16 Page1 mass, GQA max; sink0 (0 pinned page) + recent64 inside hard B2112",
 "samples_per_task": {
  "niah_single_1": 100,
  "niah_single_2": 100,
  "niah_single_3": 100,
  "niah_multikey_1": 100,
  "niah_multikey_2": 100,
  "niah_multiquery": 100,
  "niah_multivalue": 100,
  "vt": 100,
  "fwe": 100,
  "qa_1": 100,
  "qa_2": 100
 },
 "generation": "greedy, native EOS, official caps",
 "benchmark": "ruler",
 "ours_budget": 2112
}
```
