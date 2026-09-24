# nh8b-reasoning-128k-v96wo75-ruler128k

Compact per-sample predictions (prompt token ids stripped; the prompt hash, prediction, score, routing statistics and timing are kept). Full protocol per arm is in `protocols/`; summaries copied from the run directory.

| arm | samples | tasks | task-balanced mean |
|---|---:|---:|---:|
| ablation400_b0r32 | 400 | 4 | 89.50 |
| ablation400_b8r24 | 400 | 4 | 87.85 |
| b16r16_page_fisher | 1100 | 11 | 72.83 |
| b16r16_score_mse | 1100 | 11 | 55.74 |
| b8r24_page_fisher | 1100 | 11 | 75.91 |
| full | 1100 | 11 | 74.58 |
| loki | 1100 | 11 | 73.36 |
| lrqk | 1100 | 11 | 73.73 |
| shadowkv | 1100 | 11 | 70.56 |

Protocol excerpt (first arm):

```json
{
 "sequence_length": 131072,
 "rope": "native",
 "value_mode": "allocated C1 V",
 "ours": "Base16/Residual16 Page32 mass, GQA max; sink32 + recent64 inside hard B2048",
 "lrqk": {
  "rank": 32,
  "topk_per_query_head": 832,
  "recent": 64,
  "prefill_iterations": 2,
  "decode_iterations": 2,
  "tolerance": 0.01,
  "seed": 0,
  "state_dtype": "bfloat16",
  "solve_dtype": "float32"
 },
 "shadowkv": {
  "rank": 160,
  "chunk": 8,
  "routed": 2048,
  "outlier_chunks": 48,
  "extra_support": "native local and generated tokens"
 },
 "loki": {
  "rank": 32,
  "topk_per_query_head": 856,
  "recent": 0,
  "bank_manifest_sha256": "c073ed0dc7b9ffdcd8a8cc627a6b50a479ad2a698042bced2deb37344aeea426",
  "coordinate": "raw Q/K projection without mean subtraction",
  "calibration": "dense model"
 },
 "input_template": "chat template, user turn = RULER input without answer prefix, generation prompt + blank line",
 "samples_per_task": 100,
 "generation": "greedy, native EOS, official caps"
}
```
