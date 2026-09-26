# TP8 Dense Flash SDPA results

Environment: `basis`, 8 L40S, TP8, BF16. Commands and setup: [README](README.md).
Exact child commands and progress: `queue.json`; original Dense data remain untouched.

Published raw data: [HF archive](https://huggingface.co/alexz949/BasisServe-CALS/resolve/61958dd0106b32802a628c1596c54d95e6283a85/results/system_benchmarks/tp8_dense_flash_rerun/raw.tar.gz)
and [955-file inventory](https://huggingface.co/alexz949/BasisServe-CALS/blob/61958dd0106b32802a628c1596c54d95e6283a85/results/system_benchmarks/tp8_dense_flash_rerun/raw_manifest.json).
Extract the archive to restore repository-relative paths, including raw rank
JSON, queue manifests and logs referenced here. Retained Qwen Basis data are
linked from [the Qwen artifact note](../qwen3_32b_tp8_joint/SMOKE_SUMMARY.md#published-artifacts).
See also [peak GPU memory](MEMORY_SUMMARY.md).

16 conditioning forwards + 128 measured forwards. Full-model steady decode, not E2E or vLLM.
Llama values are medians of three cohort-level measurements; Qwen is a single-trial pilot.
Latency uses the harness's per-step maximum across ranks. Throughput is the recorded aggregate throughput.
Basis/ALS were not rerun in the Dense grid. A separate Qwen 130048/B4 Basis pilot was subsequently added to the comparison below.
No generated-token equivalence checks were performed.

| Model | Context | Batch | Success/attempts | Status | ms/step | tokens/s |
|---|---:|---:|---:|---|---:|---:|
| Llama-3.1-8B-Instruct | 4096 | 1 | 3/3 | complete | 31.198 | 32.27 |
| Llama-3.1-8B-Instruct | 4096 | 8 | 3/3 | complete | 31.820 | 252.32 |
| Llama-3.1-8B-Instruct | 4096 | 32 | 3/3 | complete | 32.264 | 993.16 |
| Llama-3.1-8B-Instruct | 4096 | 128 | 3/3 | complete | 35.594 | 3597.11 |
| Llama-3.1-8B-Instruct | 16384 | 1 | 3/3 | complete | 31.014 | 32.51 |
| Llama-3.1-8B-Instruct | 16384 | 4 | 3/3 | complete | 31.560 | 127.36 |
| Llama-3.1-8B-Instruct | 16384 | 8 | 3/3 | complete | 31.670 | 253.46 |
| Llama-3.1-8B-Instruct | 16384 | 16 | 3/3 | complete | 31.643 | 506.33 |
| Llama-3.1-8B-Instruct | 65536 | 1 | 3/3 | complete | 30.709 | 32.77 |
| Llama-3.1-8B-Instruct | 65536 | 4 | 3/3 | complete | 32.352 | 124.01 |
| Llama-3.1-8B-Instruct | 65536 | 8 | 3/3 | complete | 31.540 | 254.55 |
| Llama-3.1-8B-Instruct | 65536 | 16 | 0/3 | gpu_oom | - | - |
| Llama-3.1-8B-Instruct | 130048 | 1 | 3/3 | complete | 30.999 | 32.54 |
| Llama-3.1-8B-Instruct | 130048 | 4 | 3/3 | complete | 31.223 | 128.57 |
| Llama-3.1-8B-Instruct | 130048 | 8 | 3/3 | complete | 33.641 | 237.82 |
| Llama-3.1-8B-Instruct | 130048 | 16 | 0/3 | gpu_oom | - | - |
| Qwen3-32B | 65536 | 1 | 1/1 | complete | 69.787 | 14.38 |
| Qwen3-32B | 65536 | 2 | 1/1 | complete | 73.451 | 27.24 |
| Qwen3-32B | 65536 | 4 | 1/1 | complete | 71.900 | 55.71 |
| Qwen3-32B | 65536 | 8 | 1/1 | complete | 72.084 | 111.02 |
| Qwen3-32B | 65536 | 16 | 0/1 | gpu_oom | - | - |
| Qwen3-32B | 130048 | 1 | 1/1 | complete | 70.235 | 14.26 |
| Qwen3-32B | 130048 | 2 | 1/1 | complete | 71.000 | 28.21 |
| Qwen3-32B | 130048 | 4 | 1/1 | complete | 71.391 | 56.09 |
| Qwen3-32B | 130048 | 8 | 0/1 | gpu_oom | - | - |
| Qwen3-32B | 130048 | 16 | 0/0 | not_attempted_after_oom | - | - |

OOM trials are not decode latency measurements. See trial logs for failure stages.
Qwen higher batches skipped after OOM are not measured failures.
Historical speedups against the custom paged Dense kernel must not be labeled as speedups against Flash SDPA.

## Comparison with retained Basis results

These are cross-run comparisons, not newly paired Dense/Basis reruns.
Llama uses the historical three-cohort median; Qwen uses single-trial Joint V96 full-scan pilots, including the new 130048/B4 supplement.
Sources: `../llama31_8b_tp8_full_scan/summary.csv` and
`../qwen3_32b_tp8_joint/{64k_b1,64k_capacity,128k_capacity,128k_b4_basis}`.

| Model | Context | Batch | New Flash Dense ms | Retained Basis ms | Dense/Basis |
|---|---:|---:|---:|---:|---:|
| Llama-3.1-8B-Instruct | 4096 | 1 | 31.198 | 24.205 | 1.289x |
| Llama-3.1-8B-Instruct | 4096 | 8 | 31.820 | 24.639 | 1.291x |
| Llama-3.1-8B-Instruct | 4096 | 32 | 32.264 | 24.696 | 1.306x |
| Llama-3.1-8B-Instruct | 4096 | 128 | 35.594 | 51.364 | 0.693x |
| Llama-3.1-8B-Instruct | 16384 | 1 | 31.014 | 24.482 | 1.267x |
| Llama-3.1-8B-Instruct | 16384 | 4 | 31.560 | 24.292 | 1.299x |
| Llama-3.1-8B-Instruct | 16384 | 8 | 31.670 | 24.211 | 1.308x |
| Llama-3.1-8B-Instruct | 16384 | 16 | 31.643 | 24.517 | 1.291x |
| Llama-3.1-8B-Instruct | 65536 | 1 | 30.709 | 23.955 | 1.282x |
| Llama-3.1-8B-Instruct | 65536 | 4 | 32.352 | 26.759 | 1.209x |
| Llama-3.1-8B-Instruct | 65536 | 8 | 31.540 | 23.846 | 1.323x |
| Llama-3.1-8B-Instruct | 130048 | 1 | 30.999 | 23.869 | 1.299x |
| Llama-3.1-8B-Instruct | 130048 | 4 | 31.223 | 23.977 | 1.302x |
| Llama-3.1-8B-Instruct | 130048 | 8 | 33.641 | 27.850 | 1.208x |
| Qwen3-32B | 65536 | 1 | 69.787 | 57.503 | 1.214x |
| Qwen3-32B | 65536 | 2 | 73.451 | 57.006 | 1.288x |
| Qwen3-32B | 65536 | 4 | 71.900 | 56.562 | 1.271x |
| Qwen3-32B | 65536 | 8 | 72.084 | 58.843 | 1.225x |
| Qwen3-32B | 130048 | 1 | 70.235 | 55.736 | 1.260x |
| Qwen3-32B | 130048 | 2 | 71.000 | 58.468 | 1.214x |
| Qwen3-32B | 130048 | 4 | 71.391 | 56.710 | 1.259x |

## Short-context large-batch interpretation

At 4K, Basis retains 2048 support tokens, so the reduction in attention work is limited.
Its full-scan routing, page selection, K-slot maintenance, and missing-K retrieval add work absent from Dense-local.
The current Joint output path also AllGathers latent features and applies a global decoder on each rank.
For Llama TP8/V96 this decoder is 3072-to-4096 per rank, versus Dense's local 512-to-4096 output projection plus AllReduce.
The roughly 6x arithmetic ratio applies only to that matrix multiplication, not total model computation or latency.
The historical 4K/B128 Basis run has no component profile; these structural differences do not establish which component dominates the slowdown.
