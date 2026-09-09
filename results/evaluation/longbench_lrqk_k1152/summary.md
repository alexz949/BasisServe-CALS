# LRQK-equation routing + C1-V96 LongBench

Same192 prompts, six tasks x32; C1-V96 prefill; exact selected K / resident C1-V96.
Adapted resident-cache reference, not a bitwise reproduction of upstream CPU/ring-cache behavior.
Per-query-head token Top-1152 + recent64; NOT a GQA-shared2048-token or Page32 budget.
QA F1 / summary ROUGE-L; six-task arithmetic mean. Not full LongBench or an offload speed benchmark.

| Task | V96 full K | LRQK + V96 |
|---|---:|---:|
| qasper | 32.4074 | 32.5441 |
| multifieldqa_en | 42.8418 | 40.5185 |
| hotpotqa | 57.9676 | 57.7838 |
| 2wikimqa | 40.2530 | 42.1280 |
| gov_report | 30.5532 | 29.5903 |
| qmsum | 27.6797 | 27.9716 |
| Mean | 38.6171 | 38.4227 |

## Physical token union

Final decode step per sample, all layers and KV groups; not all-step average

```json
{
  "scope": "Final decode step per sample, all layers and KV groups; not all-step average",
  "count": 55296,
  "mean": 2280.501338252315,
  "minimum": 1194.0,
  "maximum": 4416.0,
  "p50": 2216.0,
  "p90": 2903.0,
  "p95": 3154.0,
  "fraction_above_2048": 0.6570638020833334
}
```

Settings and caveats: docs/c1_lrqk_integration.md. Commands are preserved per sample.
