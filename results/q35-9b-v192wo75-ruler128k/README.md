# q35-9b-v192wo75-ruler128k

Compact per-sample predictions (prompt token ids stripped; the prompt hash, prediction, score, routing statistics and timing are kept). Full protocol per arm is in `protocols/`; summaries copied from the run directory.

| arm | samples | tasks | task-balanced mean |
|---|---:|---:|---:|
| b16r16_score_mse | 1100 | 11 | 90.84 |
| b24r8_page_fisher | 1100 | 11 | 88.45 |

Protocol excerpt (first arm):

```json
{
 "sequence_length": 131072,
 "ours": {
  "base": 16,
  "residual": 16,
  "page_size": 32,
  "physical_group_budget": 2048,
  "sink": 0,
  "recent": 64,
  "recent_inside_budget": true,
  "maximum_support": 2048
 },
 "prompt_format": "official RULER base completion",
 "samples_per_task": 100
}
```
