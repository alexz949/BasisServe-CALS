# Qwen3-8B real-TP4 decode time breakdown

This run uses four L40S processes, the Transformers TP4 dense path, and the existing attention-C1 direct-slot CUDA path. Context is the attention length including the current decode token.

## Interconnect

At the largest measured payload, NCCL AllReduce moved an effective `22.339 GB/s` of ring-equivalent traffic per rank and AllGather moved `21.840 GB/s`. The `64 MiB` one-way send/receive matrix ranged from `25.066` to `25.269 GB/s`.

These are NCCL end-to-end GPU payload rates on the allocated topology, not the PCIe link's marketing line rate. Small decode payloads should be read from the latency curve in `interconnect.json`.

| Local payload | AllReduce p50 | AllGather p50 | AllReduce effective traffic | AllGather effective traffic |
|---:|---:|---:|---:|---:|
| 8 KiB | 32.768 us | 28.672 us | 0.375 GB/s | 0.857 GB/s |
| 64 KiB | 31.744 us | 39.936 us | 3.097 GB/s | 4.923 GB/s |
| 512 KiB | 70.656 us | 98.304 us | 11.130 GB/s | 16.000 GB/s |

## End-to-end decode

| Batch | Context | Dense ms | C1 ms | Speedup | Dense tok/s | C1 tok/s |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 128 | 31.0156 | 28.6388 | 1.0830x | 32.24 | 34.92 |
| 1 | 2048 | 31.6358 | 28.2759 | 1.1188x | 31.61 | 35.37 |
| 1 | 4096 | 30.8188 | 28.2016 | 1.0928x | 32.45 | 35.46 |
| 8 | 128 | 31.3440 | 29.5427 | 1.0610x | 255.23 | 270.79 |
| 8 | 2048 | 31.6578 | 29.4336 | 1.0756x | 252.70 | 271.80 |
| 8 | 4096 | 31.6130 | 29.4975 | 1.0717x | 253.06 | 271.21 |
| 64 | 128 | 31.7533 | 29.6759 | 1.0700x | 2015.54 | 2156.63 |
| 64 | 2048 | 32.1676 | 30.0548 | 1.0703x | 1989.58 | 2129.45 |
| 64 | 4096 | 32.6068 | 30.0583 | 1.0848x | 1962.78 | 2129.20 |

## Major stage totals

Stage totals are CUDA-event measurements from short instrumented passes; the table above remains the trusted uninstrumented latency.

| Batch | Context | Dense attention ms | C1 attention ms | Dense MLP ms | C1 MLP ms | Dense collective ablation ms | C1 collective ablation ms |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 128 | 22.9487 | 20.5015 | 11.4138 | 9.6012 | 7.6904 | 4.8790 |
| 1 | 2048 | 22.0868 | 19.8137 | 11.2541 | 9.4813 | 8.0649 | 4.6326 |
| 1 | 4096 | 21.1598 | 20.0926 | 10.7637 | 9.5999 | 7.2722 | 4.4217 |
| 8 | 128 | 22.1249 | 20.9058 | 10.6288 | 9.6271 | 6.9795 | 4.4901 |
| 8 | 2048 | 21.9796 | 21.3101 | 10.5127 | 9.4317 | 6.8824 | 4.3868 |
| 8 | 4096 | 21.6393 | 21.6292 | 10.7198 | 8.9152 | 6.8937 | 4.3318 |
| 64 | 128 | 21.5720 | 19.8852 | 10.9104 | 10.7905 | 6.9754 | 4.4343 |
| 64 | 2048 | 26.4809 | 23.7172 | 7.0322 | 7.3722 | 7.1682 | 4.8365 |
| 64 | 4096 | 26.7286 | 23.8012 | 7.0101 | 7.2584 | 7.3503 | 4.8859 |

## Collective bottleneck shift

| Batch | Context | Dense attention collective ms | Dense MLP collective ms | C1 attention collective ms | C1 MLP collective ms |
|---:|---:|---:|---:|---:|---:|
| 1 | 128 | 3.5884 | 3.7062 | 0.3309 | 3.8596 |
| 1 | 2048 | 3.9456 | 4.2489 | 0.6084 | 4.0056 |
| 1 | 4096 | 3.6605 | 3.4982 | 0.2854 | 3.7619 |
| 8 | 128 | 3.0331 | 3.2592 | 0.6378 | 3.8536 |
| 8 | 2048 | 3.1238 | 3.4215 | 0.4618 | 3.8464 |
| 8 | 4096 | 3.1428 | 3.0682 | 0.4678 | 3.8814 |
| 64 | 128 | 3.0739 | 3.1836 | 0.4212 | 3.8576 |
| 64 | 2048 | 3.2417 | 3.2515 | 0.7655 | 4.3375 |
| 64 | 4096 | 3.2047 | 3.4190 | 0.5794 | 4.0892 |

## Main findings

- Mean E2E speedup across the 9 points is `1.0809x`. Dense main-collective stall averages `7.253 ms`; C1 lowers it to `4.589 ms`.
- After attention-C1, attention collective stall averages only `0.506 ms`, while unchanged MLP AllReduce averages `3.944 ms`. MLP is therefore the next communication bottleneck.
- Batch 1/8 payloads sit in the latency regime: 8 KiB and 64 KiB AllReduce both take about 32 us in isolation. In-model rank skew and 72 synchronization points make their E2E cost substantially larger than bytes/bandwidth alone predicts.
- Context has modest latency impact. At batch 64, dense rises from 31.75 ms at 128 tokens to 32.61 ms at 4096; C1 rises from 29.68 to 30.06 ms. Projection, MLP, launch, and synchronization work dominate over the tested attention-length range.

## Interpretation guardrails

- Dense executes 36 attention and 36 MLP row-wise AllReduces per step. Attention-C1 replaces only the first 36; the MLP AllReduces remain.
- Collective cost uses uninstrumented counterfactuals that retain every local kernel and tensor shape while bypassing selected communication. This measures the E2E stall removed, including rank skew.
- Instrumented stage totals and fine output splits insert CUDA events. Their recorded overhead is included in the JSON; per-collective event intervals are diagnostic and are not used for communication conclusions.
- The static cache contains a zero-valued prefix to reach the exact length cheaply; all current-token projections, attention kernels, collectives, MLP, LM head, and distributed greedy selection are real.

Raw artifacts: `interconnect.json`, `dense.json`, and `c1_mean_dp.json`.
