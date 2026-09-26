# TP8 Joint Batch Sweep Results

Completed 28 attempts: 26 successful, 2 GPU OOM.
Validated 208 successful rank records and matched inputs for each successful pair.

Environment: `basis`, 8 x L40S, BF16, TP8. One trial per point, not repeated estimates.
16 conditioning forwards + 128 measured forwards; full-model steady decode, not request E2E.
All points are newly measured in this sweep. No historical results are substituted.
Dense is GPU-resident Flash SDPA; Basis is Joint V96 full-scan with historical K offload.
Latency is mean per-step rank-max CUDA-event time; throughput is measured wall tokens/s.
Memory is maximum single-rank PyTorch allocated GiB, not total device or KV-only memory.
**Primary metric: peak decode allocated GPU memory versus batch.**
Prefill OOM means no measured decode peak is available; it does not establish a decode-memory limit.
Negative memory savings mean Basis uses more allocated GPU memory at that point.
See [README](README.md) for the exact command, prompt sources and metric definitions.

## Observed Memory Crossover

Smallest jointly successful tested batch where Basis has lower allocated memory.
These are observed grid points, not exact crossover thresholds; OOM pairs are excluded.

| Model | Context | Decode peak | Prefill peak |
| --- | ---: | ---: | ---: |
| qwen8 | 65536 | 2 | 2 |
| qwen8 | 130048 | 1 | 1 |

## qwen8 / 65536

| Batch | Dense decode peak GiB | Basis decode peak GiB | Saved GiB | Saved percent |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 4.059 | 4.536 | -0.478 | -11.767 |
| 2 | 5.187 | 5.073 | 0.114 | 2.199 |
| 4 | 7.445 | 6.178 | 1.267 | 17.023 |
| 6 | 9.701 | 7.223 | 2.477 | 25.539 |
| 8 | 11.959 | 8.292 | 3.667 | 30.662 |
| 10 | 14.215 | 9.367 | 4.848 | 34.103 |
| 12 | 16.472 | 10.429 | 6.042 | 36.682 |
| 14 | 18.729 | 11.507 | 7.222 | 38.560 |
| 16 | - | 12.574 | - | - |

| Batch | Dense status | Basis status | Dense ms | Basis ms | Speedup | Dense tokens/s | Basis tokens/s |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | complete | complete | 40.658 | 32.821 | 1.239 | 24.710 | 30.640 |
| 2 | complete | complete | 40.940 | 32.778 | 1.249 | 48.999 | 61.214 |
| 4 | complete | complete | 40.998 | 32.437 | 1.264 | 97.926 | 123.660 |
| 6 | complete | complete | 41.104 | 32.422 | 1.268 | 146.334 | 185.397 |
| 8 | complete | complete | 41.171 | 32.370 | 1.272 | 194.710 | 247.648 |
| 10 | complete | complete | 41.166 | 36.155 | 1.139 | 243.230 | 277.170 |
| 12 | complete | complete | 41.480 | 36.108 | 1.149 | 289.420 | 332.781 |
| 14 | complete | complete | 41.052 | 36.198 | 1.134 | 341.134 | 387.206 |
| 16 | gpu_oom | complete | - | 36.553 | - | - | 438.028 |

Auxiliary prefill peak allocated memory (GiB):

| Batch | Dense prefill peak | Basis prefill peak |
| ---: | ---: | ---: |
| 1 | 5.659 | 6.137 |
| 2 | 8.356 | 8.243 |
| 4 | 13.749 | 12.485 |
| 6 | 19.142 | 16.670 |
| 8 | 24.536 | 20.876 |
| 10 | 29.930 | 25.089 |
| 12 | 35.323 | 29.291 |
| 14 | 40.718 | 33.507 |
| 16 | - | 37.713 |

## qwen8 / 130048

| Batch | Dense decode peak GiB | Basis decode peak GiB | Saved GiB | Saved percent |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 5.182 | 5.060 | 0.122 | 2.351 |
| 2 | 7.432 | 6.102 | 1.330 | 17.896 |
| 4 | 11.934 | 8.189 | 3.745 | 31.377 |
| 6 | 16.345 | 10.268 | 6.077 | 37.179 |
| 8 | - | 12.313 | - | - |

| Batch | Dense status | Basis status | Dense ms | Basis ms | Speedup | Dense tokens/s | Basis tokens/s |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | complete | complete | 40.619 | 33.950 | 1.196 | 24.737 | 29.590 |
| 2 | complete | complete | 40.727 | 32.296 | 1.261 | 49.331 | 62.115 |
| 4 | complete | complete | 40.630 | 32.492 | 1.250 | 98.841 | 123.341 |
| 6 | complete | complete | 40.772 | 32.465 | 1.256 | 147.295 | 185.132 |
| 8 | gpu_oom | complete | - | 32.560 | - | - | 245.920 |

Auxiliary prefill peak allocated memory (GiB):

| Batch | Dense prefill peak | Basis prefill peak |
| ---: | ---: | ---: |
| 1 | 8.321 | 8.200 |
| 2 | 13.648 | 12.319 |
| 4 | 24.303 | 20.560 |
| 6 | 34.864 | 28.790 |
| 8 | - | 36.987 |

## Whole-Run Completion

Largest successful tested batch through both prefill and decode, not a decode capacity bound.
These points are constrained by this prefill implementation; do not infer that decode cannot fit.

| Model | Context | Dense | Basis |
| --- | ---: | ---: | ---: |
| qwen8 | 65536 | 14 | 16 |
| qwen8 | 130048 | 6 | 8 |

## OOM Evidence

OOM is not a decode latency or successful capacity point. Stages below are bounded
by the last emitted rank states; conditioning/decode are not separately instrumented.

| Model | Context | Batch | Arm | Possible failure phases |
| --- | ---: | ---: | --- | --- |
| qwen8 | 65536 | 16 | dense | prefill |
| qwen8 | 130048 | 8 | dense | prefill |

## Caveats and Artifacts

- Single measurements do not provide run-to-run variance or quality equivalence.
- Fixed serial order Dense then Basis; no graph/MLP/quantization changes.
- Host NUMA placement is unverified; container set_mempolicy restrictions remain.
- Peak RSS sums are historical process maxima, not simultaneous host-memory usage.
- Source snapshot: `source.tar.gz`; existing experiment code was not changed during the sweep.
- Per-trial commands/status/metrics: `formal/manifest.json` and `formal/summary.csv`.
- Paired machine-readable table: `comparison.csv`; per-trial raw data and logs under `formal/`.
- No SHA256 validation or generated-token equivalence check was performed.
- Raw JSON/logs and the actual tested runtime are published in the [HF archive](https://huggingface.co/alexz949/BasisServe-CALS/resolve/8ee91f2b2f88a90284b2ef004b8107f06ec03c97/results/system_benchmarks/qwen3_8b_tp8_joint/raw.tar.gz); see [README](README.md#published-raw-data-and-reproduction) for reproduction requirements.
