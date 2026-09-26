# Qwen3-8B-Base TP8 NUQ4: formal

Environment: `basis`; 8 x L40S; 4096 input + 128 output tokens (127 decode forwards).
Scheduler budget 8192; GPU memory budget 80%; one warmup per batch; CUDA Graph decode.

Three measured repeats per batch; table reports medians.

| Arm | Batch | Request s | Output tok/s | TTFT ms | TPOT ms | Speedup | Preemptions max |
|---|---:|---:|---:|---:|---:|---:|---:|
| dense | 1 | 1.106 | 115.74 | 272.37 | 6.561 | 1.000x | not recorded |
| r64 | 1 | 1.406 | 91.07 | 252.57 | 9.073 | 0.787x | 0 |
| r96 | 1 | 1.436 | 89.13 | 275.55 | 9.129 | 0.770x | 0 |
| dense | 2 | 1.432 | 178.83 | 553.56 | 6.906 | 1.000x | not recorded |
| r64 | 2 | 1.679 | 152.49 | 497.18 | 9.298 | 0.853x | 0 |
| r96 | 2 | 1.741 | 147.07 | 539.94 | 9.448 | 0.822x | 0 |
| dense | 4 | 2.070 | 247.39 | 831.32 | 9.670 | 1.000x | not recorded |
| r64 | 4 | 2.269 | 225.65 | 747.99 | 11.850 | 0.912x | 0 |
| r96 | 4 | 2.387 | 214.45 | 811.93 | 12.292 | 0.867x | 0 |
| dense | 8 | 3.260 | 314.12 | 1663.64 | 12.458 | 1.000x | not recorded |
| r64 | 8 | 3.484 | 293.96 | 1498.51 | 15.479 | 0.936x | 0 |
| r96 | 8 | 3.671 | 278.92 | 1626.50 | 15.947 | 0.888x | 0 |
| dense | 16 | 5.763 | 355.34 | 2776.87 | 23.281 | 1.000x | not recorded |
| r64 | 16 | 5.961 | 343.57 | 2499.55 | 26.931 | 0.967x | 0 |
| r96 | 16 | 6.270 | 326.62 | 2713.27 | 27.678 | 0.919x | 0 |
| dense | 32 | 10.796 | 379.41 | 5007.67 | 45.039 | 1.000x | not recorded |
| r64 | 32 | 10.915 | 375.28 | 4514.85 | 49.590 | 0.989x | 0 |
| r96 | 32 | 11.475 | 356.95 | 4893.53 | 51.013 | 0.941x | 0 |
| dense | 64 | 20.880 | 392.34 | 9489.74 | 88.284 | 1.000x | not recorded |
| r64 | 64 | 20.876 | 392.42 | 8565.09 | 94.726 | 1.000x | 0 |
| r96 | 64 | 22.024 | 371.95 | 9296.64 | 98.074 | 0.948x | 0 |
| dense | 128 | 41.128 | 398.36 | 18553.17 | 173.583 | 1.000x | not recorded |
| r64 | 128 | 40.586 | 403.68 | 16821.88 | 180.221 | 1.013x | 0 |
| r96 | 128 | 42.978 | 381.22 | 18280.98 | 187.922 | 0.957x | 0 |
| dense | 256 | 83.668 | 391.64 | 37835.36 | 348.145 | 1.000x | not recorded |
| r64 | 256 | 79.629 | 411.51 | 33944.44 | 335.891 | 1.051x | 0 |
| r96 | 256 | 84.922 | 385.86 | 36915.04 | 355.896 | 0.985x | 0 |

## Interpretation

Speedup is Dense request wall time divided by quantized request wall time at the same batch.
Throughput includes prefill and all output tokens; it is not isolated steady decode throughput.
TTFT includes paused admission time. TPOT spans first to last token and can include scheduling interleaving.
Per-request medians are computed within each repeat, then across repeats.
CUDA peak allocation in CSV is cumulative since engine initialization, not per-trial resident cache memory.
Dense is reused from 2026-09-20T06:28:10.276885+00:00, not rerun alongside NUQ4.
The model snapshot, software versions and recorded scheduling settings match; Dense logs identify FlashAttention 2.
Dense used its default cache block size (not recorded explicitly); NUQ4 explicitly uses 16-token pages.
The prior Dense file has no preemption counter; missing values are not zeros.
No outlier pool overflow was reported in accepted NUQ4 trials. No output-equivalence or PPL claim is made here.
Packed cache includes BF16 outliers and metadata; the end-to-end storage reduction is not 4x.
The quantized path includes global BF16 V-statistics AllGather and uint8 latent AllGather plus W8A8 GEMM.
Full-context attention; no sparse routing or MLP changes; no SHA256.

## Commands and Raw Data

All benchmark commands use the `basis` environment and `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`.
Additional environment: `CUDA_HOME=/usr/local/cuda MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=8.9 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`.

- [dense raw JSON](../../vllm_tp8/dense.json):
  Recorded command: `/workspace/miniforge3/envs/basis/bin/python evaluation/benchmark_vllm_qwen3_8b_c1.py --arm dense --model /workspace/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --factor-dir /workspace/.cache/huggingface/hub/models--alexz949--BasisServe-CALS/snapshots/0872566b1da66eb4c813d7a1cb3313325f22b287/ICLR-results/qwen3-8b/c1/factor-banks/R64-S6 --output-dir results/vllm_tp8 --batch-sizes 1 2 4 8 16 32 64 128 256 --prefill-tokens 4096 --decode-tokens 128 --max-num-batched-tokens 8192 --repeats 3 --warmups 1 --profile-batches 1 32 256`
- [r64 raw JSON](r64/graph_splitk_4096.json):
  Recorded command: `evaluation/benchmark_vllm_qwen3_nuq4.py --phase formal --prefill-tokens 4096 --output results/q3-nuq4-vllm --rank 64`
- [r96 raw JSON](r96/graph_splitk_4096.json):
  Recorded command: `evaluation/benchmark_vllm_qwen3_nuq4.py --phase formal --prefill-tokens 4096 --output results/q3-nuq4-vllm --rank 96`

Summary command:

```bash
/workspace/miniforge3/bin/conda run --no-capture-output -n basis python evaluation/summarize_qwen3_nuq4.py --root results/q3-nuq4-vllm/formal --dense results/vllm_tp8/dense.json
```
