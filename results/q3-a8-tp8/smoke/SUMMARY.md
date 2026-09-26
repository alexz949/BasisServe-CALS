# Real TP8 Latent A8 Transport and Decoder

Eight L40S GPUs, basis environment. Actual NCCL transport; replicated global decoder.
Encoder/attention/KV cache execution is excluded. Fixtures come from BF16 encoder + fixed NUQ4 on WT2 train.
This is not full-model E2E, packed KV4 serving, or a throughput measurement.
Same adaptive checkpoint, scale, exact-width transport, samples and timing protocol across arms.

Median across trial medians; each sample uses the maximum CUDA-event latency over all eight ranks.
Full pipeline includes pack/quantize, gather, receive conversion and decoder GEMM.

| Rank | Layer | Rows | Mode | BF16 ms | Fused A8 ms | Fused W8A8 ms | BF16 / W8A8 | Unfused / fused W8A8 |
|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 64 | 18 | 1 | eager | 0.050912 | 0.061440 | 0.091168 | 0.558 | 1.114 |
| 64 | 18 | 1 | graph | 0.072896 | 0.079584 | 0.076640 | 0.951 | 1.040 |
| 64 | 18 | 16 | eager | 0.043072 | 0.065440 | 0.099968 | 0.431 | 0.905 |
| 64 | 18 | 16 | graph | 0.104448 | 0.059520 | 0.056768 | 1.840 | 0.745 |
| 96 | 18 | 1 | eager | 0.075392 | 0.068512 | 0.087104 | 0.866 | 1.132 |
| 96 | 18 | 1 | graph | 0.089728 | 0.080608 | 0.090944 | 0.987 | 1.093 |
| 96 | 18 | 16 | eager | 0.059744 | 0.060896 | 0.083936 | 0.712 | 1.123 |
| 96 | 18 | 16 | graph | 0.115680 | 0.106336 | 0.083744 | 1.381 | 0.928 |

A8 kernel correctness is bit-exact against the frozen quantization expression on tested inputs.
Communication payload halving is not a promise of latency halving. No additional decoder output all-reduce is included.
Source widths are adaptive per KV group; ragged cases use exact-width NCCL ring, uniform cases prepared NCCL AllGather.
Separate component timings do not sum exactly to the pipeline due to launch gaps, overlap and measurement boundaries.
All paths use source-local packing, although a BF16 serving attention kernel may write directly into its send slot.
Graph results replay a fixed input/shape, as appropriate for a boundary microbenchmark; not continuous batching.
