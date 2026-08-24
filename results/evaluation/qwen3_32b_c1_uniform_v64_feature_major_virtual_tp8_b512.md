# Qwen3-32B C1 virtual-TP8 decode simulation

GPU: `NVIDIA A100 80GB PCIe`; layers: `64`; dtype: `torch.bfloat16`.

| Rows | Path | Sequential p50 (ms/model) | Parallel compute floor (ms/model) | Logical wire MiB/model |
|---:|:---|---:|---:|---:|
| 512 | compact_ragged_allgather | 10.5692 | 7.47469 | 1792 |
| 512 | dense_projected_allreduce | 20.353 | 2.22822 | 4480 |
| 512 | feature_major_ragged_allgather | 13.5373 | 7.52128 | 1792 |
| 512 | local_c1_allreduce | 17.4515 | 1.83552 | 4480 |
| 512 | padded_allgather | 12.1718 | 7.58221 | 1792 |

The parallel-compute floor excludes all communication. Logical wire bytes are accounting values, not NCCL latency measurements. Neither column is measured TP8 throughput or full-model tokens/s. The feature-major path writes a reusable arena and decodes its transposed view with one GEMM.

Layers were streamed one at a time; maximum measured per-layer CUDA allocation was `261357568` bytes.
