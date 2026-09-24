# q3-8b-post-v96-retrievalmix-ruler128k

Compact per-sample predictions (prompt token ids stripped; the prompt hash, prediction, score, routing statistics and timing are kept). Full protocol per arm is in `protocols/`; summaries copied from the run directory.

| arm | samples | tasks | task-balanced mean |
|---|---:|---:|---:|
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
  "bank_manifest_sha256": "a03a5f325e0bc6cb416ba0329b66e605f844bda6c9a02cc53e8b264c65f0e31c",
  "coordinate": "post-RoPE Q/K projection without mean subtraction",
  "calibration": "dense model"
 },
 "input_template": "tokenizer.apply_chat_template user message (input), add_generation_prompt=True, then the official answer prefix as assistant text",
 "samples_per_task": 100,
 "generation": "greedy, native EOS, official caps"
}
```
