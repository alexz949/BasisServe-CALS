# Llama-3.1-8B-Instruct Dense K4V2: C4 PPL

Phase: **smoke**; environment: `basis`, one L40S, B1/TP1.

Smoke calibration and evaluation only; not a formal quality result.

| Arm | C4 PPL |
|---|---:|
| Dense BF16 | 12.090564 |
| Dense K4V2 | 13.436510 |

Quantization increment: **+1.345946 PPL (+11.132%)**.
Historical dense BF16 C4 reference: 9.773998; not used to calculate the increment.

Nominal KV compression 5.33x from bit width alone (32 / (4 + 2)); outliers, scales and metadata are excluded.
Dense full-rank K/V; no C1 factors, low-rank V, A8, FP8 GEMM, sparse routing or MLP change.
Fresh calibration: WT2 train, 1 x 128 tokens, seed 0; no C4 validation used for fitting.
K: pre-RoPE per-channel NUQ4; V: per-token NUQ2 over all 1024 channels.
Official Fisher-weighted NUQ plus 0.99 outlier rule, no rotation or first-token exclusion.
Quantize/dequantize quality simulation, not packed-cache serving performance.
No SHA256 checks. Codebook checks and final source-byte comparisons passed.
No GitHub/HF upload or old result overwrite.

Command (basis):
```bash
evaluation/eval_llama_dense_nuq_c4.py --phase smoke --arm k4v2 --output results/l31-dense-nuq-c4
```
