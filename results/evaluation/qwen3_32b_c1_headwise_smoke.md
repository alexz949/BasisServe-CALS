# Qwen3-32B C1 virtual-TP8 decode simulation

GPU: `NVIDIA A100-PCIE-40GB`; layers: `4`; dtype: `torch.bfloat16`.

| Rows | Path | Sequential p50 (ms/model) | Parallel compute floor (ms/model) | Logical wire MiB/model |
|---:|:---|---:|---:|---:|
| 1 | compact_ragged_allgather | 0.781824 | 0.343552 | 0.2854 |
| 1 | dense_projected_allreduce | 0.649216 | 0.114688 | 0.546875 |
| 1 | local_c1_allreduce | 1.26618 | 0.19456 | 0.546875 |
| 1 | padded_allgather | 1.64147 | 0.397312 | 0.382812 |
| 4 | compact_ragged_allgather | 0.761856 | 0.346112 | 1.1416 |
| 4 | dense_projected_allreduce | 0.644096 | 0.113152 | 2.1875 |
| 4 | local_c1_allreduce | 1.21498 | 0.196608 | 2.1875 |
| 4 | padded_allgather | 1.54317 | 0.397312 | 1.53125 |
| 16 | compact_ragged_allgather | 0.754688 | 0.345088 | 4.56641 |
| 16 | dense_projected_allreduce | 0.626176 | 0.114688 | 8.75 |
| 16 | local_c1_allreduce | 1.20115 | 0.195072 | 8.75 |
| 16 | padded_allgather | 1.52883 | 0.39424 | 6.125 |
| 64 | compact_ragged_allgather | 0.743424 | 0.340992 | 18.2656 |
| 64 | dense_projected_allreduce | 0.633856 | 0.114688 | 35 |
| 64 | local_c1_allreduce | 1.23341 | 0.195584 | 35 |
| 64 | padded_allgather | 1.53395 | 0.398336 | 24.5 |

The parallel-compute floor excludes all communication. Logical wire bytes are accounting values, not NCCL latency measurements. Neither column is measured TP8 throughput or full-model tokens/s.
