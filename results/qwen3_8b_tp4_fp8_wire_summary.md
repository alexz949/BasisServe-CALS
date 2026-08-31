# Qwen3-8B TP4 latent-FP8 wire results

Date: 2026-08-26

## Configuration

- Model: Qwen3-8B-Base
- Runtime: 4-way tensor parallelism on 4 NVIDIA L40S GPUs
- Software: PyTorch 2.6.0+cu124, CUDA 12.4, `basis` environment
- C1 factors: per-layer Global-KL mean-DP schedule, average rank 64, five ALS sweeps
- Attention: compressed-V Triton prefill and architecture-specific CUDA decode
- Factor compute: BF16 encoder and BF16 decoder
- Wire format: static E4M3 latent coordinates carried as raw one-byte NCCL payloads
- Scale granularity: one scale for each layer and TP source, calibrated on 256 C4 documents of 2048 tokens

The source scale is absorbed into the corresponding BF16 decoder row block. The
hot path is therefore

```text
BF16 compact-V encoder -> E4M3 quantization -> uint8 NCCL AllGather
-> E4M3-to-BF16 arena cast -> BF16 decoder GEMM
```

The encoder, decoder, K cache, and compressed V cache remain BF16. Only the
communicated latent coordinates use FP8.

## Communication

The mean local wire width is 512 elements per layer and TP rank. Across all 36
layers, the direct send volume per token and rank is

| Wire format | Bytes/token/rank/all layers | Relative volume |
|---|---:|---:|
| BF16 | 110,592 | 100% |
| E4M3 | 55,296 | 50% |

## Decode subset

The table compares otherwise identical BF16-wire and latent-FP8-wire C1 runs.

| Batch | Decode length | BF16 tok/s | FP8 tok/s | FP8 change |
|---:|---:|---:|---:|---:|
| 1 | 1024 | 36.04 | 33.63 | -6.69% |
| 1 | 4096 | 36.03 | 33.62 | -6.70% |
| 8 | 1024 | 276.68 | 259.64 | -6.16% |
| 8 | 4096 | 276.17 | 259.47 | -6.05% |
| 32 | 1024 | 1,098.57 | 1,023.61 | -6.82% |
| 128 | 1024 | 4,324.99 | 4,047.80 | -6.41% |

The arithmetic mean throughput change is -6.47%. Halving the collective payload
does not yet cover the per-layer quantization and full-arena E4M3-to-BF16 cast.

## Prefill subset

Each configuration uses three warmup runs and ten measured runs, followed by 32
output tokens.

| Batch | Prompt | BF16 prefill tok/s | FP8 prefill tok/s | FP8 change |
|---:|---:|---:|---:|---:|
| 1 | 128 | 3,927.64 | 3,710.39 | -5.53% |
| 1 | 1024 | 17,723.81 | 18,256.27 | +3.00% |
| 8 | 128 | 18,013.57 | 18,516.84 | +2.79% |
| 8 | 1024 | 16,088.53 | 16,736.05 | +4.02% |
| 32 | 128 | 18,635.30 | 19,460.51 | +4.43% |
| 32 | 1024 | 15,410.12 | 15,903.78 | +3.20% |
| 128 | 128 | 15,800.35 | 16,355.58 | +3.51% |

The arithmetic mean prefill throughput change is +2.21%. Except for the smallest
`B1,S128` configuration, the larger token workloads cover the conversion cost.
The 32-token end-to-end protocol remains decode-sensitive: only `B32,S1024`
shows a small end-to-end gain (+0.34% processed-token throughput).

## Prompt-heavy crossover

The prompt-heavy comparison uses batch 64, prompt length 4096, 128 output tokens,
three warmup runs, and ten measured runs.

| Metric | BF16 wire | FP8 wire | FP8 change |
|---|---:|---:|---:|
| Prefill throughput | 12,877.25 tok/s | 13,121.74 tok/s | +1.90% |
| Time to first token | 20,357.15 ms | 19,977.84 ms | -1.86% |
| Decode throughput | 2,110.47 tok/s | 2,028.79 tok/s | -3.87% |
| Decode latency | 3,851.28 ms | 4,006.33 ms | +4.03% |
| End-to-end latency | 24,208.43 ms | 23,984.17 ms | -0.93% |
| Processed-token throughput | 11,167.02 tok/s | 11,271.44 tok/s | +0.94% |
| Peak allocated memory/GPU | 28.02 GiB | 27.15 GiB | -3.11% |

FP8 saves 379.31 ms in prefill and adds 155.05 ms in decode, for a net 224.26 ms
latency reduction. This is the observed workload crossover where the prompt is
large enough for the 50% wire-volume reduction to produce an end-to-end gain.

## Capacity boundary

`B64,S8192` with 128 output tokens does not fit on a 48 GB L40S for either wire
format. Both arms fail during the first warmup in the stock Qwen3 RMSNorm. The
implementation converts the full `[64,8192,4096]` activation to FP32 and then
materializes another 8 GiB tensor for `hidden_states.pow(2)`. A fused RMSNorm that
performs FP32 reduction without materializing full FP32 intermediates is required
before this point can be benchmarked on L40S.

## Conclusion

Latent-only FP8 halves C1 AllGather traffic and consistently accelerates
nontrivial prefill workloads, but the current standalone conversion kernels make
decode 6-7% slower. The next runtime optimization should let the decoder consume
the E4M3 arena directly, applying scales in its input path rather than launching a
separate full-arena cast for every layer.
