# R16 Q32 vs Q64: matched V100 FP16 LongBench

Same192 prompts, fixed C1-V96/Base16, Page32/B2048,40 BCD sweeps,PCG100. Q32 factors were fitted on L40S and Q64 on V100 using FP32; generation is matched V100 FP16. Diagnostic Q32 stays fixed.

| Task | Q32 | Q64 |
|---|---:|---:|
| qasper | 31.8998 | 31.7077 |
| multifieldqa_en | 42.6738 | 42.3154 |
| hotpotqa | 56.4416 | 56.4416 |
| 2wikimqa | 43.8690 | 43.8690 |
| gov_report | 29.6477 | 30.9495 |
| qmsum | 28.3561 | 26.3186 |

```json
{
  "means": {
    "q32": 38.81467579869914,
    "q64": 38.60029856401917
  },
  "delta_pp": -0.2143772346799753,
  "improved": 23,
  "regressed": 25,
  "tied": 144
}
```
