# q3-8b-base-v96-retrievalmix-ruler128k-partial

Compact per-sample predictions (prompt token ids stripped; the prompt hash, prediction, score, routing statistics and timing are kept). Full protocol per arm is in `protocols/`; summaries copied from the run directory.

| arm | samples | tasks | task-balanced mean |
|---|---:|---:|---:|
| b16r16_page_fisher_partial633 | 633 | 7 | 64.87 |
| loki_ab200_post_rope | 200 | 2 | 37.00 |
| loki_ab200_pre_rope | 200 | 2 | 1.50 |
| mk2_100_b8r24 | 100 | 1 | 16.00 |
| mk2_100_full | 100 | 1 | 22.00 |

Protocol excerpt (first arm):

```json
{
 "sequence_length": 131072,
 "rope": "yarn4",
 "value_mode": "allocated C1 V",
 "ours": "Base16/Residual16 Page32 mass, GQA max; sink32 + recent64 inside hard B2048",
 "lrqk": {
  "rank": 32,
  "topk_per_query_head": 704,
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
  "bank_manifest_sha256": "bcd2f246d6d485d734ec58af9eaf168427cd2f01a8e47309b479fba3b8f118c8",
  "coordinate": "post-RoPE Q/K projection without mean subtraction",
  "calibration": "dense model"
 },
 "samples_per_task": 100,
 "generation": "greedy, native EOS, official caps"
}
```
