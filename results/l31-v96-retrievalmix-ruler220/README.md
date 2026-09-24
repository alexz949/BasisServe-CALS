# l31-v96-retrievalmix-ruler220

Compact per-sample predictions (prompt token ids stripped; the prompt hash, prediction, score, routing statistics and timing are kept). Full protocol per arm is in `protocols/`; summaries copied from the run directory.

| arm | samples | tasks | task-balanced mean |
|---|---:|---:|---:|
| full_conditionA_c4only | 220 | 11 | 81.31 |
| full_conditionB_retrievalmix | 220 | 11 | 84.02 |
| full_hf_c4only_partial118 | 118 | 6 | 97.64 |

Protocol excerpt (first arm):

```json
{
 "sequence_length": 131072,
 "ours": {
  "base": 16,
  "residual": 16,
  "page_size": 32,
  "physical_group_budget": 2048,
  "sink": 32,
  "recent": 64
 },
 "samples_per_task": 20,
 "generation": "greedy; native EOS; official task caps; same compressed V in every arm"
}
```
