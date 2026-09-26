# Llama-3.1-8B-Instruct Two-Sided KL V64: C4 KV4 PPL

Phase: **smoke**; environment: `basis`, one L40S, B1/TP1.

Smoke calibration and evaluation only; not a formal quality result.

| Arm | C4 PPL |
|---|---:|
| Same-checkpoint BF16 | 15.296446 |
| Same-checkpoint KV4 | 15.518771 |

KV4 increment: **+0.222325 PPL (+1.453%)**.
Historical BF16 full-C4 reference: 12.453613; not used to calculate the increment.

Checkpoint: `ICLR-results/llama31-8b-instruct/c1/two-sided-kl/R64-D0-S8`.
Ranks are adaptive, averaging 64; this is not a uniform-rank experiment.
Fresh calibration: WT2 train, 1 x 128 tokens, seed 0; no C4 validation used for fitting.
K: pre-RoPE per-channel NUQ4; V: active latent coordinates, per-token over all KV heads.
Official Fisher-weighted NUQ4 plus 0.99 outlier rule, no rotation or first-token exclusion.
Encoder/decoder and all other modules remain BF16; no A8 or FP8 GEMM.
Quantize/dequantize quality simulation, not packed-cache serving performance.
No SHA256 checks. Codebook checks and final source-byte comparisons passed.
No GitHub/HF upload or old result overwrite.

Command (basis):
```bash
evaluation/eval_llama_adaptive_nuq4_c4.py --phase smoke --rank 64 --output results/l31-adaptive-kv4-c4-r64
```
