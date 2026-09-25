# longbench-shadowkv9 / Qwen3-8B post-trained (V96, retrieval-mix calibration, yarn4)

Compact per-sample predictions (prompt token ids stripped; the prompt hash, prediction, score, routing statistics and timing are kept). Full protocol per arm is in `protocols/`; paired comparison tables in `summaries/`. The per-sample prediction files (`predictions/<arm>.jsonl`, 9–15 MB each) are on the Hugging Face repo `alexz949/BasisServe-CALS` under `results/longbench-shadowkv9/qwen3_8b/predictions/`.

Same protocol (1549 samples). LRQK covers 1546 samples: three gov_report samples (982, 995, 1018) diverge in the LRQK decode solve (BF16 factor storage in our port), excluded; paired comparisons use the 1546 common samples. page-16/8 decode arms reuse the page-32 bank with one sink page (336/328 physical).

| arm | samples | tasks | task-balanced mean |
|---|---:|---:|---:|
| b16r16_page16_decode_page32bank | 1549 | 9 | 44.15 |
| b16r16_page1_fisher | 1549 | 9 | 45.42 |
| b16r16_page32_page_fisher_sink32 | 1549 | 9 | 43.15 |
| b16r16_page4_fisher | 1549 | 9 | 45.27 |
| b16r16_page8_decode_page32bank | 1549 | 9 | 45.08 |
| b16r16_page8_fisher | 1549 | 9 | 44.91 |
| full | 1549 | 9 | 46.20 |
| lrqk_1546 | 1546 | 9 | 45.79 |
| shadowkv | 1549 | 9 | 44.49 |

Protocol excerpt (first arm):

```json
{
 "sequence_length": 131072,
 "ours": "Base16/Residual16 Page16 mass, GQA max; sink16 (one page) + recent64 inside hard B336",
 "samples_per_task": {
  "narrativeqa": 200,
  "multifieldqa_en": 111,
  "hotpotqa": 195,
  "musique": 200,
  "dureader": 200,
  "gov_report": 182,
  "samsum": 165,
  "passage_retrieval_en": 200,
  "lcc": 96
 },
 "generation": "greedy, native EOS, official caps",
 "benchmark": "longbench",
 "ours_budget": 336
}
```
