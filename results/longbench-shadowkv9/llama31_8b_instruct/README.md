# longbench-shadowkv9 / Llama-3.1-8B-Instruct (V96 identity B, C4/retrieval 50-50 calibration)

Compact per-sample predictions (prompt token ids stripped; the prompt hash, prediction, score, routing statistics and timing are kept). Full protocol per arm is in `protocols/`; paired comparison tables in `summaries/`.

ShadowKV-aligned LongBench-v1 protocol: 9 tasks, samples > 4096 tokens under the model tokenizer (1543), sparse budget 256; page-32 arm = released Page-Fisher bank with sink 32 + recent 64 (352 physical); page-8/4/1 arms = Page-Fisher refits at that page size, no sink, 256 + recent 64 (320); dense_full = uncompressed model on the same prompts; lrqk = rank 32, top-k 256 + recent 64; shadowkv = rank 160, routed 256 + 48 outlier chunks; loki_prerope_pca32 = centered pre-RoPE K PCA r32 of the dense model, top-k 256 + recent 64 per query head (own audited run, paired in summaries/eval_llama31_8b_instruct_loki_summary.txt). Quantized-cache arms (2026-09-25, page-4 B16R16 and Full-K, simulated quantize-dequantize with BF16 storage, prefill on BF16 K/V, decode on the dequantized cache, routing sidecar BF16 with the Base term from the stored V latent): *_fp8kv = float8_e4m3fn, K per-token per-KV-head absmax scale, V96 latent per-page-4 per-head scale; *_nuq4kv = KVQuant NUQ4 (SqueezeAILab/KVQuant 57a2383), pre-RoPE K static per-channel + V96 latent dynamic per-token, 16 Fisher-weighted signposts, 1% dense-and-sparse outliers, calibrated on 16 x 2048 WikiText-2 train tokens of the deployed V96 model (HF .../kvquant_nuq4_B); summaries/eval_llama31_8b_instruct_p4fit_{fp8kv,nuq4kv}_summary.txt.

| arm | samples | tasks | task-balanced mean |
|---|---:|---:|---:|
| b16r16_page1_decode_page32bank | 1543 | 9 | 46.89 |
| b16r16_page1_fisher | 1543 | 9 | 46.66 |
| b16r16_page32_page_fisher_sink32 | 1543 | 9 | 46.01 |
| b16r16_page4_fisher | 1543 | 9 | 46.91 |
| b16r16_page4_fisher_fp8kv | 1543 | 9 | 46.85 |
| b16r16_page4_fisher_nuq4kv | 1543 | 9 | 46.50 |
| b16r16_page8_fisher | 1543 | 9 | 46.61 |
| dense_full | 1543 | 9 | 50.21 |
| full | 1543 | 9 | 47.02 |
| full_fp8kv | 1543 | 9 | 47.14 |
| full_nuq4kv | 1543 | 9 | 47.24 |
| loki_prerope_pca32 | 1543 | 9 | 46.54 |
| lrqk | 1543 | 9 | 46.62 |
| shadowkv | 1543 | 9 | 46.72 |

Protocol excerpt (first arm):

```json
{
 "sequence_length": 131072,
 "ours": "Base16/Residual16 Page1 mass, GQA max; sink1 (one page) + recent64 inside hard B321",
 "samples_per_task": {
  "narrativeqa": 200,
  "multifieldqa_en": 110,
  "hotpotqa": 195,
  "musique": 200,
  "dureader": 200,
  "gov_report": 180,
  "samsum": 165,
  "passage_retrieval_en": 200,
  "lcc": 93
 },
 "generation": "greedy, native EOS, official caps",
 "benchmark": "longbench",
 "ours_budget": 321
}
```
