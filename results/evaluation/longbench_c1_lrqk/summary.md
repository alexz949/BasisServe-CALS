# LRQK-equation routing + C1-V96 LongBench

Same192 prompts, six tasks x32; C1-V96 prefill; exact selected K / resident C1-V96.
Adapted resident-cache reference, not a bitwise reproduction of upstream CPU/ring-cache behavior.
Per-query-head token Top-2048 + recent64 by default; NOT a GQA-shared2048-token or Page32 budget.
QA F1 / summary ROUGE-L; six-task arithmetic mean. Not full LongBench or an offload speed benchmark.

| Task | V96 full K | LRQK + V96 |
|---|---:|---:|
| qasper | 32.4074 | 31.5713 |
| multifieldqa_en | 42.8418 | 39.3042 |
| hotpotqa | 57.9676 | 56.4051 |
| 2wikimqa | 40.2530 | 42.1280 |
| gov_report | 30.5532 | 31.4825 |
| qmsum | 27.6797 | 27.6080 |
| Mean | 38.6171 | 38.0832 |

Settings and caveats: docs/c1_lrqk_integration.md. Commands are preserved per sample.
