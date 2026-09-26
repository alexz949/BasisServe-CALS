# Qwen3-8B-Base TP8 NUQ4: preflight

Environment: `basis`; 8 x L40S; 4096 input + 128 output tokens (127 decode forwards).
Scheduler budget 8192; GPU memory budget 80%; one warmup per batch; CUDA Graph decode.

Single-repeat quantized endpoint checks against prior three-repeat Dense; not formal speed conclusions.

| Arm | Batch | Request s | Output tok/s | TTFT ms | TPOT ms | Speedup | Preemptions max |
|---|---:|---:|---:|---:|---:|---:|---:|
| dense | 1 | 1.106 | 115.74 | 272.37 | 6.561 | 1.000x | not recorded |
| r64 | 1 | 1.413 | 90.56 | 252.41 | 9.135 | 0.782x | 0 |
| r96 | 1 | 1.444 | 88.64 | 275.44 | 9.194 | 0.766x | 0 |
| dense | 256 | 83.668 | 391.64 | 37835.36 | 348.145 | 1.000x | not recorded |
| r64 | 256 | 79.654 | 411.38 | 33962.11 | 336.080 | 1.050x | 0 |
| r96 | 256 | 84.809 | 386.37 | 36874.83 | 355.303 | 0.987x | 0 |

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
  Recorded command: `evaluation/benchmark_vllm_qwen3_nuq4.py --phase preflight --rank 64 --prefill-tokens 4096 --output results/q3-nuq4-vllm`
- [r96 raw JSON](r96/graph_splitk_4096.json):
  Recorded command: `evaluation/benchmark_vllm_qwen3_nuq4.py --phase preflight --rank 96 --prefill-tokens 4096 --output results/q3-nuq4-vllm`

Summary command:

```bash
/workspace/miniforge3/bin/conda run --no-capture-output -n basis python evaluation/summarize_qwen3_nuq4.py --root results/q3-nuq4-vllm/preflight --dense results/vllm_tp8/dense.json
```
