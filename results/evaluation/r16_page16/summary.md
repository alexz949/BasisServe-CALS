# Page32 vs Page16: matched V100 FP16 LongBench

Same192 prompts, fixed C1-V96/Base16, Page32/B2048. B2048 and pinned32 tokens; Q32,40 BCD sweeps,PCG100. Page32 factors fitted on L40S and Page16 on V100 in FP32. Generation matched V100 FP16.

| Task | Page32 | Page16 |
|---|---:|---:|
| qasper | 31.8998 | 31.7511 |
| multifieldqa_en | 42.6738 | 41.1400 |
| hotpotqa | 56.4416 | 56.4416 |
| 2wikimqa | 43.8690 | 43.8690 |
| gov_report | 29.6477 | 29.7670 |
| qmsum | 28.3561 | 26.8901 |

```json
{
  "means": {
    "page32": 38.81467579869914,
    "page16": 38.30980868444618
  },
  "delta_pp": -0.5048671142529663,
  "improved": 27,
  "regressed": 25,
  "tied": 140
}
```
