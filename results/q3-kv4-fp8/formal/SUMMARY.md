# Qwen3-8B-Base KV4 + FP8 Full PPL

Environment: `basis`. Eight independent single-GPU arms; no TP8 collectives.
Full WT2 test: 146 non-overlapping 2048-token windows, B1, 298862 scored tokens per arm.

| Nominal rank | Arm | PPL | Delta PPL | Change |
|---:|---|---:|---:|---:|
| 64 | bf16 | 8.258728 | +0.000000 | +0.000% |
| 64 | kv4 | 8.277446 | +0.018718 | +0.227% |
| 64 | fp8 | 8.388993 | +0.130265 | +1.577% |
| 64 | kv4_fp8 | 8.300389 | +0.041661 | +0.504% |
| 96 | bf16 | 7.283357 | +0.000000 | +0.000% |
| 96 | kv4 | 7.338056 | +0.054699 | +0.751% |
| 96 | fp8 | 7.294730 | +0.011373 | +0.156% |
| 96 | kv4_fp8 | 7.354980 | +0.071623 | +0.983% |

R64/R96 are adaptive equivalent average ranks. Actual E4M3 W8A8 encoder/decoder GEMMs return BF16; other modules remain BF16.
KV4 is official NUQ4 with outliers, quantize/dequantize quality simulation, not a packed-cache speed measurement.
NUQ4 and frozen FP8 scales are calibrated only on WT2 train (16 x 2048 tokens). No SHA256 checks or TP8 validation.
R64 BF16 and its NUQ4 calibration are reused from the original serial process; other arms use the same evaluator in separate processes.
The anomalous short R64 smoke is not used in this table; see the README and order audit.
Parallel workers may contend for CPU/PCIe; wall_seconds is not a serving latency result.

Commands and protocols: `../README.md`, `manifest.json`, `parallel_manifest.json` and `parallel_outcomes.json`.
