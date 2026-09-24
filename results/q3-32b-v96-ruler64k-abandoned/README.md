# q3-32b-v96-ruler64k-abandoned

Compact per-sample predictions (prompt token ids stripped; the prompt hash, prediction, score, routing statistics and timing are kept). Full protocol per arm is in `protocols/`; summaries copied from the run directory.

| arm | samples | tasks | task-balanced mean |
|---|---:|---:|---:|
| b16r16_page_fisher_partial104 | 104 | 2 | 100.00 |
| loki_ab200_post_rope | 200 | 2 | 97.00 |
| loki_ab200_pre_rope | 200 | 2 | 3.50 |

Protocol excerpt (first arm):

```json
{
 "sequence_length": 65536,
 "rope": "yarn2",
 "value_mode": "allocated C1 V",
 "ours": "Base16/Residual16 Page32 mass, GQA max; sink32 + recent64 inside hard B2048",
 "lrqk": {
  "rank": 32,
  "topk_per_query_head": 416,
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
  "bank_manifest_sha256": "f61d7b363ee9f881b52d13130d93796f3d9f9e534071fa683a45b763089323e7",
  "coordinate": "post-RoPE Q/K projection without mean subtraction",
  "calibration": "dense model"
 },
 "samples_per_task": 100,
 "generation": "greedy, native EOS, official caps"
}
```
