# Llama-3.1-8B-Instruct V96 KV4 PPL

Phase: **smoke**. Environment: `basis`, one L40S, B1/TP1.

Smoke only; not a formal quality result.

- PPL: **8.749013**.
- Windows: 2; scored tokens: 4094.
- Unused test tokens: 284981.
- Evaluation wall time: 1.26 s (not serving latency).

Frozen formal R96 codebooks fitted on 16 x 2048 WT2 train tokens; no recalibration.
NUQ4 with outliers quantizes full K before RoPE and active V96 latent coordinates.
Encoder/decoder and other projections stay BF16; no A8, FP8 GEMM, sparse routing or MLP changes.
Quality uses quantize/dequantize simulation, not packed-cache serving or TP8 performance.
No matched BF16 baseline was run here; do not infer delta PPL from unrelated evaluations.
Structure validation and source-byte comparisons only; no SHA256. No GitHub/HF upload.

Command (basis):
```bash
evaluation/eval_llama_nuq4_ppl.py --phase smoke --output results/l31-nuq4-ppl
```
