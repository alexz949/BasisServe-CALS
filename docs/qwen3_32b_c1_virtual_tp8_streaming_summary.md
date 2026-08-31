# Qwen3-32B C1 Virtual-TP8 Streaming Simulation

## Executive summary

This implementation provides a single-GPU harness for validating the C1
post-attention TP8 data path without requiring eight physical GPUs. Eight TP
ranks are executed sequentially on one NVIDIA A100 80GB PCIe GPU, while the
64 Qwen3-32B layers are loaded and released one at a time.

The run validates:

- C1 checkpoint identity, tensor metadata, rank schedule, and TP8 ownership;
- per-rank headwise encoding without a zero-filled block-diagonal GEMM;
- compact ragged rank ordering and decoder packing;
- a feature-major `[K, rows]` receive arena decoded by one GEMM;
- numerical equivalence across all 64 layers;
- sequential single-GPU compute latency, a communication-free parallel compute
  floor, and logical collective byte volume;
- actual per-layer CUDA allocation and cleanup during streaming.

The Slurm job `8274620` completed successfully in 48 seconds.

## Important scope: there is no sequence length in this run

The benchmark values `256` and `512` are active decode rows, approximately the
batch size of one decode step. They are not sequence lengths.

The simulated local input begins at the dense attention output boundary:

```text
[rows, 8 query heads, 128 head dimensions]
```

The harness does not construct Q/K tensors, apply RoPE, access a KV cache, or
execute the attention score/value-reduction kernel. Consequently, it has no
`seqlen=1024`, `2048`, or `4096` setting. Its measurements cannot be used as
end-to-end latency results for any context length.

The 48-second Slurm elapsed time is the duration of the complete benchmark job,
including factor loading, warmups, 100 timing iterations per case, correctness
checks, and result serialization. It is not model inference latency.

## Model and TP geometry

The validated Qwen3-32B configuration is:

| Property | Value |
|:---|---:|
| Hidden size | 5120 |
| Query heads | 64 |
| KV heads | 8 |
| Head dimension | 128 |
| Transformer layers | 64 |
| Virtual TP size | 8 |
| Query heads owned by each TP rank | 8 |

Process rank `s` owns physical KV head `s` and its contiguous group of eight
query heads.

## Implementation

### 1. C1 factor validation and packing

The factor loader in
[`basisserve/core/c1_tp_decode.py`](../basisserve/core/c1_tp_decode.py):

1. verifies the expected `result.json` SHA-256;
2. verifies the model-config hash and complete 64-layer coverage;
3. checks the selected per-layer rank schedule and its recorded budget;
4. verifies every layer artifact hash, tensor hash, shape, and dtype;
5. enforces one physical KV head per TP8 process rank;
6. extracts the local encoder `A_s` with shape `[128, r_s]`;
7. packs the eight decoder blocks owned by source `s` into a local decoder with
   shape `[8 r_s, 5120]`;
8. concatenates all source blocks in process-rank order into the compact global
   decoder `D` with shape `[K, 5120]`, where
   `K = sum_s 8 r_s`;
9. constructs a `StaticRaggedPlan` containing each source width `8 r_s` and its
   cumulative offset.

The same source order is used by the local encoders, ragged communication
layout, and compact global decoder.

### 2. Headwise encoder

For each virtual rank, the local dense attention output is encoded as:

```text
[rows, 8, 128]
  -> reshape [rows * 8, 128]
  -> GEMM with A_s [128, r_s]
  -> reshape [rows, 8 * r_s]
```

This is one dense GEMM per source. It does not expand `A_s` into a large
block-diagonal matrix containing zeros.

This encoder is an oracle for the current post-attention harness. In the final
serving path, variable-width compressed-V attention should emit the local C1
coordinates directly.

### 3. Feature-major ragged arena

The new simulation path preallocates a reusable feature-major arena:

```text
arena: [K, rows]
decoder: [K, 5120]
```

For virtual source `s`, its `[rows, 8 r_s]` coordinates are transposed and
written into:

```text
arena[offset_s : offset_s + 8 r_s, :]
```

The completed arena is exactly the transpose of the compact token-major latent:

```text
arena.T == cat([z_0, z_1, ..., z_7], dim=-1)
```

The output decode is therefore one GEMM:

```text
Y = arena.T @ D
```

The layout is implemented by
[`basisserve/kernels/feature_ragged_allgather.py`](../basisserve/kernels/feature_ragged_allgather.py)
and exercised in
[`evaluation/simulate_qwen3_32b_c1_tp_decode.py`](../evaluation/simulate_qwen3_32b_c1_tp_decode.py).
It matches the layout expected by both the two-sided `feature_direct` transport
and the NCCL 2.29+ `feature_rma` transport. Neither distributed transport was
executed in this single-GPU run.

### 4. Per-layer streaming

For each layer, the harness:

1. loads and validates one C1 artifact on CPU;
2. packs all eight virtual ranks onto the GPU;
3. runs correctness and timing measurements for rows 256 and 512;
4. records peak CUDA allocation;
5. clears tensors retained by benchmark closures;
6. releases the packed factors and calls `torch.cuda.empty_cache()`;
7. records the remaining allocated CUDA memory before loading the next layer.

Only the model configuration is read. The 32B model weights are not loaded.
The memory result therefore demonstrates that the C1 harness and streamed
factors fit on one GPU; it does not demonstrate that the complete Qwen3-32B
model fits on that GPU.

## Compared paths

| Path | Meaning |
|:---|:---|
| `dense_projected_allreduce` | Per-rank dense output projection followed by a logical dense-output AllReduce |
| `local_c1_allreduce` | Per-rank headwise C1 encode and local decode followed by the same logical dense-output AllReduce |
| `compact_ragged_allgather` | Compact token-major C1 coordinates followed by one compact global decoder GEMM |
| `feature_major_ragged_allgather` | Rank-offset feature-major arena followed by one global decoder GEMM |
| `padded_allgather` | Every source padded to the largest source width before gather and decode |

All eight virtual ranks run sequentially on one GPU. The reported parallel
compute floor uses the slowest virtual-rank component for each layer and
excludes all communication.

## Experiment configuration

| Setting | Value |
|:---|:---|
| GPU | NVIDIA A100 80GB PCIe |
| Slurm job | `8274620` |
| Conda environment | `lowrank` |
| Runtime dtype | BF16 |
| Layers | All 64 |
| Active rows | 256 and 512 |
| Correctness rows | 512 |
| Warmup iterations | 10 |
| Measured iterations | 100 |
| Relative L2 tolerance | 0.02 |
| Checkpoint manifest SHA-256 | `fa34eac71b0f7cca112fd9e071a43c5cb8459d9ec6fd2ddbaff22a19703b4661` |

## Performance results

The latency columns sum the independently measured per-layer medians or p95
values across all 64 layers.

| Rows | Path | Sequential p50 (ms/64 layers) | Sequential p95 (ms/64 layers) | Parallel compute floor (ms/64 layers) | Logical wire MiB/64 layers |
|---:|:---|---:|---:|---:|---:|
| 256 | `compact_ragged_allgather` | 7.5264 | 7.7404 | 4.9454 | 896 |
| 256 | `dense_projected_allreduce` | 13.2198 | 13.3315 | 1.4418 | 2240 |
| 256 | `feature_major_ragged_allgather` | 10.5882 | 10.9076 | 5.0616 | 896 |
| 256 | `local_c1_allreduce` | 13.3586 | 13.5137 | 1.7956 | 2240 |
| 256 | `padded_allgather` | 13.4098 | 13.7615 | 6.7297 | 1340.5 |
| 512 | `compact_ragged_allgather` | 11.0295 | 11.1370 | 7.9969 | 1792 |
| 512 | `dense_projected_allreduce` | 21.3996 | 21.5644 | 2.2282 | 4480 |
| 512 | `feature_major_ragged_allgather` | 14.1251 | 14.2561 | 8.1418 | 1792 |
| 512 | `local_c1_allreduce` | 18.4310 | 18.7013 | 2.2707 | 4480 |
| 512 | `padded_allgather` | 17.4648 | 17.5616 | 11.6695 | 2681 |

### Derived observations

- Compact ragged sequential compute is `1.76x` faster than the dense path at
  256 rows and `1.94x` faster at 512 rows.
- Feature-major sequential compute is `1.25x` faster than the dense path at
  256 rows and `1.52x` faster at 512 rows.
- Feature-major is currently `40.7%` slower than the idealized compact
  token-major path at 256 rows and `28.1%` slower at 512 rows. Its sequential
  measurement includes eight explicit arena writes. The purpose of this layout
  is to let real rank-local attention/communication write the final arena
  directly and avoid a later global repacking kernel; the single-GPU result
  does not yet prove that distributed feature-major transport is faster.
- Compact and feature-major C1 communication both account for 60% fewer
  logical wire bytes than dense-output ring AllReduce for this checkpoint.
- Padded AllGather transfers about 49.6% more bytes than compact ragged
  AllGather because layers containing larger source ranks determine each
  source's padded width.

## Communication accounting

Let:

- `P = 8` be TP size;
- `B` be active rows;
- `H = 5120` be hidden size;
- `e = 2` bytes for BF16;
- `w_s = 8 r_s` be source `s`'s local coordinate width;
- `K = sum_s w_s`.

The logical total wire volume across all ranks is:

```text
compact ragged AllGather = (P - 1) * B * K * e
dense ring AllReduce     = 2 * (P - 1) * B * H * e
padded AllGather         = (P - 1) * P * B * max_s(w_s) * e
```

Therefore:

```text
ragged / dense = K / (2H)
```

For the selected Qwen3-32B schedule, the aggregate ratio is `0.40`, giving a
`60%` reduction rather than `75%`. Both dense and C1 output communication scale
linearly with active rows. Sequence length is absent because KV attention is
outside this benchmark boundary.

## Correctness

All 64 layers passed the `0.02` relative L2 tolerance, and no path produced
NaN or Inf:

| Path | Maximum relative L2 error | Maximum absolute error | Failures |
|:---|---:|---:|---:|
| `dense_projected_allreduce` | 0.005512 | 1.5 | 0 |
| `local_c1_allreduce` | 0.004975 | 1.5 | 0 |
| `feature_major_ragged_allgather` | 0.002181 | 0.125 | 0 |
| `padded_allgather` | 0 | 0 | 0 |

The small differences are consistent with BF16 GEMM/reduction ordering. The
feature-major arena ordering itself is also covered by a unit test asserting
`arena.T == compact_latent`.

Static validation completed before the Slurm run:

- Ruff passed;
- 14 relevant tests passed;
- Python compilation passed;
- `git diff --check` passed.

## Memory results

| Metric | Value |
|:---|---:|
| Maximum per-layer CUDA allocation | 351,649,792 bytes (335.4 MiB) |
| CUDA allocation after every layer cleanup | 8,519,680 bytes (8.1 MiB) |

The constant post-cleanup allocation across all 64 layers confirms that the
simulation does not accumulate prior-layer C1 tensors on the GPU.

## What this result establishes

This run establishes:

- checkpoint-to-runtime C1 layout consistency;
- correct TP8 ownership and rank ordering;
- correctness of compact and feature-major decode semantics;
- feasibility of streaming all 64 C1 layers on one A100;
- isolated single-GPU compute costs;
- exact logical output-collective byte accounting.

It does not establish:

- actual eight-GPU NCCL latency or throughput;
- communication/computation overlap;
- PCIe or NVLink contention;
- variable-width V-cache attention latency;
- behavior at any sequence length;
- full-model decode latency or tokens per second.

## Required next integration step

An end-to-end decode simulation must add an explicit sequence length and model
the components currently outside this boundary:

- Q/K projection and RoPE;
- dense or compressed KV-cache reads;
- variable-width V-cache attention;
- actual TP communication timing or a calibrated topology model;
- RMSNorm, residual connections, MLP, LM head, and scheduler overhead.

That experiment can then compare `seqlen=1024`, `2048`, and `4096` at active
batch sizes 256 and 512 without conflating post-attention C1 decode cost with
context-dependent attention cost.

## Reproduction

```bash
/home/zhangal/.conda/envs/lowrank/bin/python \
  evaluation/simulate_qwen3_32b_c1_tp_decode.py \
  --factor-dir results/checkpoints/qwen3_32b_c1_aasvd_gkl_v64_ragged_als_10s_256f64h_full2048_d1e5 \
  --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137 \
  --expected-result-sha256 fa34eac71b0f7cca112fd9e071a43c5cb8459d9ec6fd2ddbaff22a19703b4661 \
  --layers all \
  --batches 256,512 \
  --correctness-batch 512 \
  --dtype bfloat16 \
  --warmup 10 \
  --iterations 100 \
  --relative-tolerance 0.02 \
  --device cuda:0 \
  --output-json results/evaluation/qwen3_32b_c1_feature_major_virtual_tp8_b256_b512.json \
  --output-markdown results/evaluation/qwen3_32b_c1_feature_major_virtual_tp8_b256_b512.md
```

## Artifacts

- [Full JSON result](../results/evaluation/qwen3_32b_c1_feature_major_virtual_tp8_b256_b512.json)
- [Generated result table](../results/evaluation/qwen3_32b_c1_feature_major_virtual_tp8_b256_b512.md)
- [Slurm log](../results/logs/c1_fm_vtp8_8274620.log)
- [Simulation implementation](../evaluation/simulate_qwen3_32b_c1_tp_decode.py)
- [Factor-loader and packer implementation](../basisserve/core/c1_tp_decode.py)
- [Feature-major transport implementation](../basisserve/kernels/feature_ragged_allgather.py)
- [Relevant tests](../tests/test_c1_tp_decode.py)
