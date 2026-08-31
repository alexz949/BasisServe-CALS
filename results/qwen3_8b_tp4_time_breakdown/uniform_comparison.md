# Qwen3-8B TP4 uniform-R64 vs mean-DP avg-R64

Both C1 arms retain exactly 50% of the Value width on average and use the same direct-slot CUDA/NCCL runtime. The only change is the per-layer rank schedule.

| Batch | Context | Dense ms | Mean-DP ms | Uniform ms | Uniform vs mean-DP | Uniform speedup vs dense |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 128 | 31.0156 | 28.6388 | 28.5331 | -0.369% | 1.0870x |
| 1 | 2048 | 31.6358 | 28.2759 | 28.0628 | -0.754% | 1.1273x |
| 1 | 4096 | 30.8188 | 28.2016 | 27.6936 | -1.801% | 1.1128x |
| 8 | 128 | 31.3440 | 29.5427 | 29.0211 | -1.765% | 1.0800x |
| 8 | 2048 | 31.6578 | 29.4336 | 29.0940 | -1.154% | 1.0881x |
| 8 | 4096 | 31.6130 | 29.4975 | 29.1249 | -1.263% | 1.0854x |
| 64 | 128 | 31.7533 | 29.6759 | 29.2309 | -1.500% | 1.0863x |
| 64 | 2048 | 32.1676 | 30.0548 | 29.5203 | -1.778% | 1.0897x |
| 64 | 4096 | 32.6068 | 30.0583 | 29.6452 | -1.374% | 1.0999x |

## Aggregate

- Uniform mean speedup vs dense: `1.0952x`.
- Mean-DP mean speedup vs dense: `1.0809x`.
- Uniform mean latency delta vs mean-DP: `-0.3837 ms` (`-1.307%`).
- Uniform is faster at `9` of `9` points.

## Quality

- Dense WikiText-2 PPL: `7.002509`.
- Mean-DP avg-R64 PPL: `8.594518`.
- Uniform-R64 PPL: `8.416738`.

Latency is measured on real 4xL40S TP4. Instrumented substage data remains diagnostic; E2E and collective-removal ablations are the primary comparison.
