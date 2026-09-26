# Qwen3-8B-Base KV4 + FP8 PPL

Phase: **smoke**. Environment: `basis`; single L40S.

Smoke numbers are diagnostics, not full-test PPL.

| Nominal rank | Arm | PPL | Delta PPL | Change |
|---:|---|---:|---:|---:|
| 64 | bf16 | 26.406772 | +0.000000 | +0.000% |
| 64 | kv4 | 10.666617 | -15.740155 | -59.607% |
| 64 | fp8 | 10.403244 | -16.003527 | -60.604% |
| 64 | kv4_fp8 | 10.720442 | -15.686330 | -59.403% |
| 96 | bf16 | 10.183958 | +0.000000 | +0.000% |
| 96 | kv4 | 10.343896 | +0.159938 | +1.570% |
| 96 | fp8 | 10.210873 | +0.026916 | +0.264% |
| 96 | kv4_fp8 | 10.185898 | +0.001940 | +0.019% |

R64/R96 are equivalent average ranks of the original adaptive checkpoints, not uniform per-layer ranks.
K/V NUQ4 uses the official quantize/dequantize quality simulation with outliers; this is not a packed-cache speed test.
FP8 uses actual E4M3 GEMMs for folded V encoder and output decoder only; other components remain BF16.
All fitted quantities use WT2 train, never test. Activation scales are frozen for evaluation, not inferred from future test tokens.
Structure-only factor validation; no SHA256. No TP8 verification.

Command:
```bash
evaluation/eval_qwen3_kv4_fp8_ppl.py --phase smoke --ranks 64 96 --output results/q3-kv4-fp8
```
