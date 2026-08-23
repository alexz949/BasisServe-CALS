# Qwen3-32B C1 virtual-TP8 decode simulation

GPU: `NVIDIA A100-PCIE-40GB`; layers: `1`; dtype: `torch.bfloat16`.

| Rows | Path | Sequential p50 (ms/model) | Parallel compute floor (ms/model) | Logical wire MiB/model |
|---:|:---|---:|---:|---:|
| 1 | compact_ragged_allgather | 0.195584 | 0.111616 | 0.0769043 |
| 1 | dense_projected_allreduce | 0.17408 | 0.031744 | 0.136719 |
| 1 | local_c1_allreduce | 0.310272 | 0.058368 | 0.136719 |
| 1 | padded_allgather | 0.434176 | 0.128 | 0.109375 |
| 4 | compact_ragged_allgather | 0.196608 | 0.104448 | 0.307617 |
| 4 | dense_projected_allreduce | 0.17408 | 0.032768 | 0.546875 |
| 4 | local_c1_allreduce | 0.289792 | 0.049152 | 0.546875 |
| 4 | padded_allgather | 0.400384 | 0.123904 | 0.4375 |

The parallel-compute floor excludes all communication. Logical wire bytes are accounting values, not NCCL latency measurements. Neither column is measured TP8 throughput or full-model tokens/s.
