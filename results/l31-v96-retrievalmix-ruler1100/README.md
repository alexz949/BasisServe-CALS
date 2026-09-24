# l31-v96-retrievalmix-ruler1100

Compact per-sample predictions (prompt token ids stripped; the prompt hash, prediction, score, routing statistics and timing are kept). Full protocol per arm is in `protocols/`; summaries copied from the run directory.

| arm | samples | tasks | task-balanced mean |
|---|---:|---:|---:|
| b0r32_page_fisher | 1100 | 11 | 81.84 |
| b16r16_page_fisher | 1100 | 11 | 82.01 |
| b16r16_score_mse_partial634 | 634 | 7 | 92.84 |
| full | 1100 | 11 | 84.32 |
| loki_post_rope_ablation | 1100 | 11 | 18.51 |
| loki_pre_rope | 1100 | 11 | 50.19 |
| lrqk | 1100 | 11 | 81.05 |
| shadowkv | 1100 | 11 | 77.47 |

Protocol excerpt (first arm):

```json
{
 "sequence_length": 131072,
 "ours": {
  "base": 0,
  "residual": 32,
  "page_size": 32,
  "physical_group_budget": 2048,
  "sink": 32,
  "recent": 64
 },
 "samples_per_task": 100,
 "generation": "greedy; native EOS; official task caps; same compressed V in every arm"
}
```
