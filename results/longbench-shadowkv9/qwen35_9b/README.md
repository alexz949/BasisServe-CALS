# longbench-shadowkv9 / Qwen3.5-9B (V192 + GDN Wo75, C4-32 calibration, released HF Page-Fisher router)

Compact per-sample predictions (prompt token ids stripped; the prompt hash, prediction, score, routing statistics and timing are kept). Full protocol per arm is in `protocols/`; paired comparison tables in `summaries/`. The per-sample prediction files (`predictions/<arm>.jsonl`, 9–15 MB each) are on the Hugging Face repo `alexz949/BasisServe-CALS` under `results/longbench-shadowkv9/qwen35_9b/predictions/`.

1564 samples; ours = 256 + recent 64 (no sink on Qwen3.5). passage_retrieval_en ~20 under compression vs 100 dense is a chat-format behaviour change (the compressed model explains instead of answering 'Paragraph N'); samsum dense 8.5 is a scoring artifact of a leading empty think block (see summaries).

| arm | samples | tasks | task-balanced mean |
|---|---:|---:|---:|
| b16r16_hf_page_fisher | 1564 | 9 | 39.20 |
| dense | 1564 | 9 | 47.37 |
| full | 1564 | 9 | 39.45 |
| lrqk | 1564 | 9 | 39.73 |
| shadowkv | 1564 | 9 | 39.70 |

Protocol excerpt (first arm):

```json
{
 "sequence_length": 131072,
 "ours": {
  "base": 16,
  "residual": 16,
  "page_size": 32,
  "physical_group_budget": 320,
  "sink": 0,
  "recent": 64,
  "recent_inside_budget": true,
  "maximum_support": 320
 },
 "samples_per_task": {
  "narrativeqa": 200,
  "multifieldqa_en": 110,
  "hotpotqa": 195,
  "musique": 200,
  "dureader": 200,
  "gov_report": 182,
  "samsum": 167,
  "passage_retrieval_en": 200,
  "lcc": 110
 },
 "benchmark": "longbench"
}
```
