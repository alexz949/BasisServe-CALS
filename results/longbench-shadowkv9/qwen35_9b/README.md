# longbench-shadowkv9 / Qwen3.5-9B (V192 + GDN Wo75, C4-32 calibration, released HF Page-Fisher router; recal-mix48 arms = recalibrated on the 16 C4 + 16 retrieval mix)

Compact per-sample predictions (prompt token ids stripped; the prompt hash, prediction, score, routing statistics and timing are kept). Full protocol per arm is in `protocols/`; paired comparison tables in `summaries/`.

1564 samples; ours = 256 + recent 64 (no sink on Qwen3.5). Arms without the recal-mix48 suffix use the C4-32-calibrated V192+Wo75 and the released page-32 Page-Fisher router; the recal-mix48 arms use the V192 + GDN Wo75 recalibrated on the 16 C4 + 16 synthetic-retrieval 128K windows (HF checkpoints/qwen35-9b-128k/recal-mix48) with page-4 / page-1 Page-Fisher routers (no sink), LRQK rank 32 top-k 256 + recent 64, ShadowKV rank 160 routed 256 + 48 outlier chunks, and Loki (centered pre-RoPE Key PCA r32 of the dense model on the 32 mix48 fit windows, HF checkpoints/qwen35-9b-128k/recal-mix48/loki_pca32_pre_rope; top-k 320 per query head, no recent window, 320 physical like ours and LRQK) on the same recalibrated model — compare recal arms only with each other. passage_retrieval_en ~20-40 under compression vs 100 dense is a chat-format behaviour change (sparse arms score higher than compressed Full on it); samsum dense 8.5 is a scoring artifact of a leading empty think block (see summaries).

| arm | samples | tasks | task-balanced mean |
|---|---:|---:|---:|
| b16r16_hf_page_fisher | 1564 | 9 | 39.20 |
| b16r16_page1_fisher_recal_mix48 | 1564 | 9 | 41.62 |
| b16r16_page4_fisher_recal_mix48 | 1564 | 9 | 41.32 |
| dense | 1564 | 9 | 47.37 |
| full | 1564 | 9 | 39.45 |
| full_recal_mix48 | 1564 | 9 | 41.30 |
| loki_recal_mix48 | 1564 | 9 | 40.92 |
| lrqk | 1564 | 9 | 39.73 |
| lrqk_recal_mix48 | 1564 | 9 | 41.08 |
| shadowkv | 1564 | 9 | 39.70 |
| shadowkv_recal_mix48 | 1564 | 9 | 41.83 |

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
