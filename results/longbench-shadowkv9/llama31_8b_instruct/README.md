# longbench-shadowkv9 / Llama-3.1-8B-Instruct (V96 identity B, C4/retrieval 50-50 calibration)

Compact per-sample predictions (prompt token ids stripped; the prompt hash, prediction, score, routing statistics and timing are kept). Full protocol per arm is in `protocols/`; paired comparison tables in `summaries/`. The per-sample prediction files (`predictions/<arm>.jsonl`, 9–15 MB each) are on the Hugging Face repo `alexz949/BasisServe-CALS` under `results/longbench-shadowkv9/llama31_8b_instruct/predictions/`.

ShadowKV-aligned LongBench-v1 protocol: 9 tasks, samples > 4096 tokens under the model tokenizer (1543), sparse budget 256; page-32 arm = released Page-Fisher bank with sink 32 + recent 64 (352 physical); page-8/4/1 arms = Page-Fisher refits at that page size, no sink, 256 + recent 64 (320); dense_full = uncompressed model on the same prompts.

| arm | samples | tasks | task-balanced mean |
|---|---:|---:|---:|
| b16r16_page1_decode_page32bank | 1543 | 9 | 46.89 |
| b16r16_page1_fisher | 1543 | 9 | 46.66 |
| b16r16_page32_page_fisher_sink32 | 1543 | 9 | 46.01 |
| b16r16_page4_fisher | 1543 | 9 | 46.91 |
| b16r16_page8_fisher | 1543 | 9 | 46.61 |
| dense_full | 1543 | 9 | 50.21 |
| full | 1543 | 9 | 47.02 |
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
