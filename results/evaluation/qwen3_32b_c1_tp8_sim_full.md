# Qwen3-32B C1 virtual-TP8 decode simulation

GPU: `NVIDIA A100-PCIE-40GB`; layers: `64`; dtype: `torch.bfloat16`.

| Rows | Path | Sequential p50 (ms/model) | Parallel compute floor (ms/model) | Logical wire MiB/model |
|---:|:---|---:|---:|---:|
| 1 | compact_ragged_allgather | 11.8518 | 4.73702 | 3.5 |
| 1 | dense_projected_allreduce | 9.86829 | 1.7367 | 8.75 |
| 1 | local_c1_allreduce | 18.9947 | 3.08173 | 8.75 |
| 1 | padded_allgather | 25.2846 | 5.83578 | 5.23633 |
| 4 | compact_ragged_allgather | 11.6536 | 4.69146 | 14 |
| 4 | dense_projected_allreduce | 10.0936 | 1.75923 | 35 |
| 4 | local_c1_allreduce | 19.2138 | 3.10323 | 35 |
| 4 | padded_allgather | 24.2918 | 5.77178 | 20.9453 |
| 8 | compact_ragged_allgather | 11.7361 | 4.72883 | 28 |
| 8 | dense_projected_allreduce | 10.5784 | 1.86163 | 70 |
| 8 | local_c1_allreduce | 19.1094 | 3.13242 | 70 |
| 8 | padded_allgather | 24.342 | 5.80403 | 41.8906 |
| 16 | compact_ragged_allgather | 11.863 | 4.79386 | 56 |
| 16 | dense_projected_allreduce | 10.6286 | 1.88006 | 140 |
| 16 | local_c1_allreduce | 19.3679 | 3.13856 | 140 |
| 16 | padded_allgather | 24.5816 | 5.85984 | 83.7812 |
| 32 | compact_ragged_allgather | 12.0151 | 4.88704 | 112 |
| 32 | dense_projected_allreduce | 10.5303 | 1.86419 | 280 |
| 32 | local_c1_allreduce | 19.2543 | 3.13754 | 280 |
| 32 | padded_allgather | 24.9498 | 5.99808 | 167.562 |
| 64 | compact_ragged_allgather | 12.1119 | 4.79539 | 224 |
| 64 | dense_projected_allreduce | 10.4745 | 1.87392 | 560 |
| 64 | local_c1_allreduce | 19.7463 | 3.16621 | 560 |
| 64 | padded_allgather | 25.0491 | 5.82963 | 335.125 |
| 128 | compact_ragged_allgather | 12.225 | 5.10976 | 448 |
| 128 | dense_projected_allreduce | 10.7556 | 1.86982 | 1120 |
| 128 | local_c1_allreduce | 19.2937 | 3.11347 | 1120 |
| 128 | padded_allgather | 25.0644 | 6.24486 | 670.25 |

The parallel-compute floor excludes all communication. Logical wire bytes are accounting values, not NCCL latency measurements. Neither column is measured TP8 throughput or full-model tokens/s.
