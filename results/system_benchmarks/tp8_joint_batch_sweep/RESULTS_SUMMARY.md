# TP8 Joint Batch Sweep Results

Completed 72 attempts: 44 successful, 28 GPU OOM.
Validated 352 successful rank records and matched inputs for each successful pair.

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
| llama | 65536 | 2 | 2 |
| llama | 130048 | 1 | 1 |
| qwen | 65536 | 6 | 6 |
| qwen | 130048 | 4 | 4 |

## llama / 65536

| Batch | Dense decode peak GiB | Basis decode peak GiB | Saved GiB | Saved percent |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 3.738 | 4.165 | -0.427 | -11.431 |
| 2 | 4.741 | 4.644 | 0.097 | 2.041 |
| 4 | 6.747 | 5.626 | 1.120 | 16.604 |
| 6 | 8.753 | 6.561 | 2.193 | 25.048 |
| 8 | 10.759 | 7.512 | 3.247 | 30.179 |
| 10 | 12.766 | 8.472 | 4.294 | 33.635 |
| 12 | 14.771 | 9.418 | 5.354 | 36.243 |
| 14 | 16.777 | 10.378 | 6.399 | 38.140 |
| 16 | - | 11.330 | - | - |

| Batch | Dense status | Basis status | Dense ms | Basis ms | Speedup | Dense tokens/s | Basis tokens/s |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | complete | complete | 31.775 | 24.302 | 1.307 | 31.543 | 41.409 |
| 2 | complete | complete | 31.464 | 25.142 | 1.251 | 63.884 | 79.797 |
| 4 | complete | complete | 31.648 | 24.547 | 1.289 | 126.998 | 163.904 |
| 6 | complete | complete | 31.420 | 24.603 | 1.277 | 191.653 | 244.941 |
| 8 | complete | complete | 31.482 | 24.809 | 1.269 | 254.813 | 322.948 |
| 10 | complete | complete | 32.819 | 27.368 | 1.199 | 305.237 | 366.393 |
| 12 | complete | complete | 31.206 | 26.261 | 1.188 | 384.719 | 457.034 |
| 14 | complete | complete | 32.139 | 28.461 | 1.129 | 435.652 | 491.987 |
| 16 | gpu_oom | complete | - | 30.450 | - | - | 525.537 |

Auxiliary prefill peak allocated memory (GiB):

| Batch | Dense prefill peak | Basis prefill peak |
| ---: | ---: | ---: |
| 1 | 5.338 | 5.766 |
| 2 | 7.910 | 7.814 |
| 4 | 13.052 | 11.936 |
| 6 | 18.195 | 16.007 |
| 8 | 23.338 | 20.099 |
| 10 | 28.481 | 24.196 |
| 12 | 33.624 | 28.281 |
| 14 | 38.767 | 32.380 |
| 16 | - | 36.470 |

## llama / 130048

| Batch | Dense decode peak GiB | Basis decode peak GiB | Saved GiB | Saved percent |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 4.735 | 4.632 | 0.103 | 2.167 |
| 2 | 6.736 | 5.561 | 1.175 | 17.449 |
| 4 | 10.738 | 7.420 | 3.318 | 30.896 |
| 6 | 14.659 | 9.272 | 5.387 | 36.746 |
| 8 | 18.634 | 11.093 | 7.540 | 40.467 |
| 10 | - | - | - | - |
| 12 | - | - | - | - |
| 14 | - | - | - | - |
| 16 | - | - | - | - |

| Batch | Dense status | Basis status | Dense ms | Basis ms | Speedup | Dense tokens/s | Basis tokens/s |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | complete | complete | 30.630 | 24.067 | 1.273 | 32.864 | 41.898 |
| 2 | complete | complete | 31.079 | 24.592 | 1.264 | 64.759 | 81.614 |
| 4 | complete | complete | 31.139 | 24.479 | 1.272 | 129.067 | 163.982 |
| 6 | complete | complete | 31.151 | 24.536 | 1.270 | 192.764 | 244.687 |
| 8 | complete | complete | 33.614 | 27.369 | 1.228 | 238.013 | 292.333 |
| 10 | gpu_oom | gpu_oom | - | - | - | - | - |
| 12 | gpu_oom | gpu_oom | - | - | - | - | - |
| 14 | gpu_oom | gpu_oom | - | - | - | - | - |
| 16 | gpu_oom | gpu_oom | - | - | - | - | - |

Auxiliary prefill peak allocated memory (GiB):

| Batch | Dense prefill peak | Basis prefill peak |
| ---: | ---: | ---: |
| 1 | 7.875 | 7.773 |
| 2 | 12.953 | 11.778 |
| 4 | 23.107 | 19.791 |
| 6 | 33.179 | 27.794 |
| 8 | 43.305 | 35.769 |
| 10 | - | - |
| 12 | - | - |
| 14 | - | - |
| 16 | - | - |

## qwen / 65536

| Batch | Dense decode peak GiB | Basis decode peak GiB | Saved GiB | Saved percent |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 10.911 | 14.398 | -3.487 | -31.963 |
| 2 | 12.916 | 15.359 | -2.444 | -18.919 |
| 4 | 16.928 | 17.335 | -0.407 | -2.404 |
| 6 | 20.938 | 19.210 | 1.728 | 8.251 |
| 8 | 24.950 | 21.108 | 3.842 | 15.399 |
| 10 | - | 23.043 | - | - |
| 12 | - | - | - | - |
| 14 | - | - | - | - |
| 16 | - | - | - | - |

| Batch | Dense status | Basis status | Dense ms | Basis ms | Speedup | Dense tokens/s | Basis tokens/s |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | complete | complete | 71.096 | 56.436 | 1.260 | 14.094 | 17.779 |
| 2 | complete | complete | 75.090 | 59.579 | 1.260 | 26.666 | 33.628 |
| 4 | complete | complete | 74.014 | 57.208 | 1.294 | 54.066 | 70.029 |
| 6 | complete | complete | 73.307 | 56.837 | 1.290 | 81.939 | 105.693 |
| 8 | complete | complete | 71.681 | 59.646 | 1.202 | 111.743 | 134.202 |
| 10 | gpu_oom | complete | - | 63.575 | - | - | 157.305 |
| 12 | gpu_oom | gpu_oom | - | - | - | - | - |
| 14 | gpu_oom | gpu_oom | - | - | - | - | - |
| 16 | gpu_oom | gpu_oom | - | - | - | - | - |

Auxiliary prefill peak allocated memory (GiB):

| Batch | Dense prefill peak | Basis prefill peak |
| ---: | ---: | ---: |
| 1 | 12.899 | 16.387 |
| 2 | 16.861 | 19.307 |
| 4 | 24.787 | 25.197 |
| 6 | 32.711 | 30.989 |
| 8 | 40.636 | 36.802 |
| 10 | - | 42.653 |
| 12 | - | - |
| 14 | - | - |
| 16 | - | - |

## qwen / 130048

| Batch | Dense decode peak GiB | Basis decode peak GiB | Saved GiB | Saved percent |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 12.906 | 15.316 | -2.410 | -18.674 |
| 2 | 16.907 | 17.180 | -0.273 | -1.614 |
| 4 | 24.910 | 20.916 | 3.993 | 16.031 |
| 6 | - | - | - | - |
| 8 | - | - | - | - |
| 10 | - | - | - | - |
| 12 | - | - | - | - |
| 14 | - | - | - | - |
| 16 | - | - | - | - |

| Batch | Dense status | Basis status | Dense ms | Basis ms | Speedup | Dense tokens/s | Basis tokens/s |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | complete | complete | 70.124 | 55.640 | 1.260 | 14.277 | 18.004 |
| 2 | complete | complete | 71.728 | 57.017 | 1.258 | 27.931 | 35.148 |
| 4 | complete | complete | 72.013 | 57.077 | 1.262 | 55.614 | 70.152 |
| 6 | gpu_oom | gpu_oom | - | - | - | - | - |
| 8 | gpu_oom | gpu_oom | - | - | - | - | - |
| 10 | gpu_oom | gpu_oom | - | - | - | - | - |
| 12 | gpu_oom | gpu_oom | - | - | - | - | - |
| 14 | gpu_oom | gpu_oom | - | - | - | - | - |
| 16 | gpu_oom | gpu_oom | - | - | - | - | - |

Auxiliary prefill peak allocated memory (GiB):

| Batch | Dense prefill peak | Basis prefill peak |
| ---: | ---: | ---: |
| 1 | 16.804 | 19.214 |
| 2 | 24.640 | 24.912 |
| 4 | 40.308 | 36.316 |
| 6 | - | - |
| 8 | - | - |
| 10 | - | - |
| 12 | - | - |
| 14 | - | - |
| 16 | - | - |

## Whole-Run Completion

Largest successful tested batch through both prefill and decode, not a decode capacity bound.
These points are constrained by this prefill implementation; do not infer that decode cannot fit.

| Model | Context | Dense | Basis |
| --- | ---: | ---: | ---: |
| llama | 65536 | 14 | 16 |
| llama | 130048 | 8 | 8 |
| qwen | 65536 | 8 | 10 |
| qwen | 130048 | 4 | 4 |

## OOM Evidence

OOM is not a decode latency or successful capacity point. Stages below are bounded
by the last emitted rank states; conditioning/decode are not separately instrumented.

| Model | Context | Batch | Arm | Possible failure phases |
| --- | ---: | ---: | --- | --- |
| llama | 65536 | 16 | dense | prefill |
| llama | 130048 | 10 | dense | prefill |
| llama | 130048 | 10 | basis_joint | prefill |
| llama | 130048 | 12 | dense | prefill |
| llama | 130048 | 12 | basis_joint | prefill |
| llama | 130048 | 14 | dense | prefill |
| llama | 130048 | 14 | basis_joint | prefill |
| llama | 130048 | 16 | dense | prefill |
| llama | 130048 | 16 | basis_joint | prefill |
| qwen | 65536 | 10 | dense | prefill |
| qwen | 65536 | 12 | dense | prefill |
| qwen | 65536 | 12 | basis_joint | prefill |
| qwen | 65536 | 14 | dense | prefill |
| qwen | 65536 | 14 | basis_joint | prefill |
| qwen | 65536 | 16 | dense | prefill |
| qwen | 65536 | 16 | basis_joint | prefill |
| qwen | 130048 | 6 | dense | prefill |
| qwen | 130048 | 6 | basis_joint | prefill |
| qwen | 130048 | 8 | dense | prefill |
| qwen | 130048 | 8 | basis_joint | prefill |
| qwen | 130048 | 10 | dense | model_loading_or_cache_allocation |
| qwen | 130048 | 10 | basis_joint | prefill |
| qwen | 130048 | 12 | dense | model_loading_or_cache_allocation |
| qwen | 130048 | 12 | basis_joint | prefill |
| qwen | 130048 | 14 | dense | model_loading_or_cache_allocation |
| qwen | 130048 | 14 | basis_joint | prefill |
| qwen | 130048 | 16 | dense | model_loading_or_cache_allocation |
| qwen | 130048 | 16 | basis_joint | prefill |

## Caveats and Artifacts

- Single measurements do not provide run-to-run variance or quality equivalence.
- Fixed serial order Dense then Basis; no graph/MLP/quantization changes.
- Host NUMA placement is unverified; container set_mempolicy restrictions remain.
- Peak RSS sums are historical process maxima, not simultaneous host-memory usage.
- Source snapshot: `source.tar.gz`; existing experiment code was not changed during the sweep.
- Per-trial commands/status/metrics: `formal/manifest.json` and `formal/summary.csv`.
- Paired machine-readable table: `comparison.csv`; per-trial raw data and logs under `formal/`.
- No SHA256 validation or generated-token equivalence check was performed.
- Raw JSON, logs, manifests and source snapshot: [fixed HF revision and inventory](README.md#published-data).
