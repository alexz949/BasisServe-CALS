# R16 solver budget: matched V100 FP16 LongBench

Same192 prompts, fixed C1-V96/Base16, Page32/B2048. Old40/100 factors were fitted on L40S; new50/150 factors on V100, both using FP32 fitting. Generation is matched V100 FP16.

| Task | 40 sweeps / PCG100 | 50 sweeps / PCG150 |
|---|---:|---:|
| qasper | 31.8998 | 31.9779 |
| multifieldqa_en | 42.6738 | 42.8779 |
| hotpotqa | 56.4416 | 56.4416 |
| 2wikimqa | 43.8690 | 43.8690 |
| gov_report | 29.6477 | 30.1082 |
| qmsum | 28.3561 | 27.6125 |

```json
{
  "means": {
    "s40p100": 38.81467579869914,
    "s50p150": 38.81452351752841
  },
  "delta_pp": -0.00015228117074550518,
  "improved": 21,
  "regressed": 17,
  "tied": 154
}
```
