# Qwen3-32B C1 virtual-TP8 decode simulation

GPU: `NVIDIA A100 80GB PCIe`; layers: `64`; dtype: `torch.bfloat16`.

| Rows | Path | Sequential p50 (ms/model) | Parallel compute floor (ms/model) | Logical wire MiB/model |
|---:|:---|---:|---:|---:|
| 256 | compact_ragged_allgather | 7.5264 | 4.94541 | 896 |
| 256 | dense_projected_allreduce | 13.2198 | 1.44179 | 2240 |
| 256 | feature_major_ragged_allgather | 10.5882 | 5.06163 | 896 |
| 256 | local_c1_allreduce | 13.3586 | 1.79558 | 2240 |
| 256 | padded_allgather | 13.4098 | 6.72973 | 1340.5 |
| 512 | compact_ragged_allgather | 11.0295 | 7.99693 | 1792 |
| 512 | dense_projected_allreduce | 21.3996 | 2.22822 | 4480 |
| 512 | feature_major_ragged_allgather | 14.1251 | 8.14182 | 1792 |
| 512 | local_c1_allreduce | 18.431 | 2.27072 | 4480 |
| 512 | padded_allgather | 17.4648 | 11.6695 | 2681 |

The parallel-compute floor excludes all communication. Logical wire bytes are accounting values, not NCCL latency measurements. Neither column is measured TP8 throughput or full-model tokens/s. The feature-major path writes a reusable arena and decodes its transposed view with one GEMM.

Layers were streamed one at a time; maximum measured per-layer CUDA allocation was `351649792` bytes.
