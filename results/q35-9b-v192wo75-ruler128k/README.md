# q35-9b-v192wo75-ruler128k

Compact per-sample predictions (prompt token ids stripped; the prompt hash, prediction, score, routing statistics and timing are kept). Full protocol per arm is in `protocols/`; paired comparison tables in `summaries/`.

RULER-128K, 1100 prompts (11 tasks x 100), official base-completion prompts, thinking off. b16r16_score_mse / b24r8_page_fisher: C4-32-calibrated V192 + GDN Wo75 with page-32 routers (compared with the released HF five-arm table in summary_*_vs_hf_release.txt). recal_mix48 arms (2026-09-25): V192 + GDN Wo75 recalibrated on the 16 C4 + 16 synthetic-retrieval 128K windows (HF checkpoints/qwen35-9b-128k/recal-mix48); full = exact attention on that model; b16r16_page4_fisher = page-4 Page-Fisher router, no sink, 2048 routed + recent 64 = 2112; lrqk = rank 32, top-k 832 + recent 64; shadowkv = rank 160, routed 2048 + 48 outlier chunks; loki = centered pre-RoPE Key PCA r32 of the dense model on the 32 mix48 fit windows (HF .../recal-mix48/loki_pca32_pre_rope), top-k 856 per query head, no recent window — the released baseline settings. Compare arms within one calibration only; paired bootstrap CIs in summaries/eval1100_p4_mix_summary.txt.

| arm | samples | tasks | task-balanced mean |
|---|---:|---:|---:|
| b16r16_page4_fisher_recal_mix48 | 1100 | 11 | 94.27 |
| b16r16_score_mse | 1100 | 11 | 90.84 |
| b24r8_page_fisher | 1100 | 11 | 88.45 |
| full_recal_mix48 | 1100 | 11 | 94.73 |
| loki_recal_mix48 | 1100 | 11 | 93.67 |
| lrqk_recal_mix48 | 1100 | 11 | 93.81 |
| shadowkv_recal_mix48 | 1100 | 11 | 93.13 |

Protocol excerpt (first arm):

```json
{
 "sequence_length": 131072,
 "ours": {
  "base": 16,
  "residual": 16,
  "page_size": 4,
  "physical_group_budget": 2112,
  "sink": 0,
  "recent": 64,
  "recent_inside_budget": true,
  "maximum_support": 2112
 },
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
 "benchmark": "ruler"
}
```
