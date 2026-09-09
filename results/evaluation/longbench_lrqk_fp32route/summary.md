# FP16 C1 with FP32 LRQK routing

Same192 LongBench prompts. FP16 model and exact K/C1-V96 cache; FP32 routing factors/codes/score scan.
C1-V96 memory-efficient SDPA prefill. Existing completed FP16 full-K control reused with per-record hashes.
No clipping or optimizer change. Neither LRQK budget is a hard shared B2048 cap.

| Task | Full K | k1152 | k1280 |
|---|---:|---:|---:|
| qasper | 32.4596 | 31.7511 | 31.7077 |
| multifieldqa_en | 40.8995 | 40.9939 | 40.9502 |
| hotpotqa | 56.2213 | 56.2213 | 56.2213 |
| 2wikimqa | 43.8690 | 43.8690 | 43.8690 |
| gov_report | 30.3007 | 29.8472 | 31.0826 |
| qmsum | 27.2612 | 26.5405 | 26.5271 |
| Mean | 38.5019 | 38.2038 | 38.3930 |

## Physical union

```json
{
  "k1152": {
    "scope": "Final decode step per prompt, all layers/groups; not all-step mean",
    "mean": 2292.789044415509,
    "minimum": 1194.0,
    "maximum": 4448.0,
    "p50": 2224.0,
    "p90": 2934.0,
    "p95": 3196.0,
    "fraction_above_2048": 0.6651837384259259
  },
  "k1280": {
    "scope": "Final decode step per prompt, all layers/groups; not all-step mean",
    "mean": 2491.0851779513887,
    "minimum": 1194.0,
    "maximum": 4890.0,
    "p50": 2413.0,
    "p90": 3202.0,
    "p95": 3509.0,
    "fraction_above_2048": 0.8103660300925926
  }
}
```
