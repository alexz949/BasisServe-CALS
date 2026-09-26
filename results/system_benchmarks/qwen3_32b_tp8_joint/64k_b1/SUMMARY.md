# Qwen3-32B TP8 Joint V96: 64K/B1 Pilot

Baseline caveat: Dense decode uses the custom `gpu_paged_attention` CUDA kernel,
not the Flash SDPA backend used by the earlier TP1 Dense benchmark. Speedup is
relative to that custom implementation, not a FlashAttention/vLLM comparison.

All three arms completed. No OOM. Single trial per arm, not a three-cohort
formal result. This is fixed-batch full-model steady decode, not vLLM or E2E
request latency.

## Settings

- Environment: `basis`; eight L40S GPUs; TP8/BF16.
- Prompt: 65536 tokens, batch 1; 16 conditioning and 128 measured forwards.
- Input: row 0, first 65536 tokens from
  `/workspace/runs/qwen3-32b-densev-ruler30/calibration/windows.safetensors`.
  Identical input across arms; a calibration-bank input, not a held-out quality test.
- Matched 128K-calibrated V96 and B16R16 banks, static YaRN factor 4.
- Full-scan routing, Page32, 62 selected pages including sink, recent64,
  2048 support slots per KV head. No two-stage selection.
- Dense/ALS-full: GPU K/V. Basis: pinned-host historical exact K, GPU K slots
  and V96/residual cache. No new MLP optimization.

## Results

| Arm | Decode ms/step | Decode tokens/s | Prefill wall s | Prefill peak allocated GiB/rank | Decode resident allocated GiB/rank |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dense | 145.714 | 6.863 | 15.929 | 12.904 | 10.915 |
| ALS-full V96 | 146.367 | 6.832 | 15.616 | 17.205 | 15.215 |
| Basis Joint V96 | 57.503 | 17.436 | 15.810 | 16.384 | 14.394 |

Basis speedup: 2.534x versus Dense, 2.545x versus ALS-full.
The mean decode step uses per-step maximum CUDA-event time across ranks.
Throughput uses maximum-rank decode wall time. Prefill wall and GPU memory
columns use maximum across ranks. GPU memory is PyTorch allocated, not total
device usage. Basis additionally allocates 8.018 GiB historical host K across
the eight ranks; it is not included in the GPU memory columns.

All 24 rank JSON files have complete status and all ranks agree on generated
token IDs within each arm. Finite logits passed the benchmark checks. No
cross-arm quality equivalence is implied. Host NUMA memory binding remains
unverified due to container `set_mempolicy` restrictions. No SHA256 checks.

## Command

Working directory: `/workspace/BasisServe-CALS-opt`.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 CUDA_HOME=/usr/local/cuda MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=8.9 /workspace/miniforge3/bin/conda run --no-capture-output -n basis python benchmarks/system/run_qwen3_32b_joint_smoke.py --length 65536 --batch 1 --conditioning-steps 16 --measure-steps 128 --tokens /workspace/runs/qwen3-32b-densev-ruler30/calibration/windows.safetensors --output results/system_benchmarks/qwen3_32b_tp8_joint/64k_b1
```

Exact child commands: `trials.json`. Logs: `dense.log`, `als_full.log`,
`basis_joint.log`; raw results under `smoke_<arm>_p65536_b1_r0/rank*.json`.
See [initial smoke summary](../SMOKE_SUMMARY.md) for factor provenance and
54 passing kernel tests.

Only B1 has been measured at 64K here. Larger batches and repeated formal
measurements remain pending; this pilot does not establish a capacity boundary.
