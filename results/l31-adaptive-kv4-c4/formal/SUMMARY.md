# Llama-3.1-8B-Instruct Two-Sided KL V96: C4 KV4 PPL

Phase: **formal**; environment: `basis`, one L40S, B1/TP1.

Frozen C4 validation 128 x 2048 windows; 262,016 scored tokens, no cross-document transitions.

| Arm | C4 PPL |
|---|---:|
| Same-checkpoint BF16 | 10.662169 |
| Same-checkpoint KV4 | 10.726644 |

KV4 increment: **+0.064475 PPL (+0.605%)**.
Historical BF16 full-C4 reference: 10.662169; not used to calculate the increment.

Checkpoint: `ICLR-results/llama31-8b-instruct/c1/two-sided-kl/R96-D0-S8`.
Ranks are adaptive, averaging 96; this is not the preceding uniform-V96 experiment.
Fresh calibration: WT2 train, 16 x 2048 tokens, seed 0; no C4 validation used for fitting.
K: pre-RoPE per-channel NUQ4; V: active latent coordinates, per-token over all KV heads.
Official Fisher-weighted NUQ4 plus 0.99 outlier rule, no rotation or first-token exclusion.
Encoder/decoder and all other modules remain BF16; no A8 or FP8 GEMM.
Quantize/dequantize quality simulation, not packed-cache serving performance.
No SHA256 checks. Codebook checks and final source-byte comparisons passed.
No GitHub/HF upload or old result overwrite.

Command (basis):
```bash
evaluation/eval_llama_adaptive_nuq4_c4.py --phase formal --output results/l31-adaptive-kv4-c4
```
