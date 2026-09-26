# BF16 Encoder, A8 Latent and W8 Decoder

Phase: formal. Environment: basis. Qwen3-8B-Base, adaptive R64/R96.
KV4 NUQ4 is fixed in all arms. This is single-GPU PPL, not a TP8 communication benchmark.

| Rank | KV4 + BF16 | KV4 + A8 | KV4 + A8/W8 | A8 delta | W8 increment over A8 |
|---:|---:|---:|---:|---:|---:|
| 64 | 8.277446 | 8.280403 | 8.285759 | +0.002957 | +0.005356 |
| 96 | 7.338056 | 7.342300 | 7.344549 | +0.004244 | +0.002249 |

Encoder remains BF16. A8: E4M3 quantize/dequantize then BF16 decoder GEMM.
A8/W8: the same latent scale, E4M3 decoder weights, actual FP8 GEMM with BF16 output.
Static scales come from the previous full train-only calibration under BF16 projections + KV4.
PPL includes quantized upstream layers. W8 increment is an end-to-end difference, not independent additive error.
No SHA256. NUQ4 remains quantize/dequantize quality simulation, not packed INT4 storage.
Smoke covers only two 2048-token windows; formal covers all 146 windows / 298862 targets.
Per-worker commands and settings: r64/manifest.json and r96/manifest.json.
