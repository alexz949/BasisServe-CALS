# Qwen3-32B C1 variable-width V-cache decode microbenchmark

This is a single-GPU rank-local PyTorch reference benchmark. It does not measure NCCL, the global decoder, full transformer layers, or serving tokens/s.

- Device: `NVIDIA A100 80GB PCIe`
- Dtype: `torch.bfloat16`
- Layers: `[0, 20, 48, 63]`
- Virtual TP ranks: `[0, 1, 2, 3, 4, 5, 6, 7]`
- Warmup / iterations: `5 / 30`
- Maximum projection relative L2 error: `0.00364147`
- Maximum attention relative L2 error: `0.00318416`

## Sampled-layer critical path

For each sampled layer, the table takes the slowest selected virtual rank, then sums those layer latencies. Communication is excluded.

| Batch | Context | Dense p50 ms | Compact p50 ms | Speedup | Median rank speedup | Mean V-cache reduction |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 1024 | 76.0192 | 70.8762 | 1.073x | 1.2x | 50.000% |

## Width scaling at the largest workload

Batch `512`, context `1024`; medians are across every occurrence of that source rank in the sampled layers.

| Source rank | V-cache reduction | Attention speedup | End-to-end speedup |
|---:|---:|---:|---:|
| 64 | 50.000% | 1.204x | 1.2x |

## TP maximum-rank straggler at the largest workload

The per-layer critical path is the slowest selected virtual rank. A low mean rank does not improve synchronized TP latency when the layer still contains an uncompressed rank-128 source.

| Layer | Mean rank | Max rank | Dense p50 ms | Compact p50 ms | Speedup |
|---:|---:|---:|---:|---:|---:|
| 0 | 64.000 | 64 | 18.945 | 17.7075 | 1.07x |
| 20 | 64.000 | 64 | 18.9978 | 17.7219 | 1.072x |
| 48 | 64.000 | 64 | 19.0382 | 17.7418 | 1.073x |
| 63 | 64.000 | 64 | 19.0382 | 17.705 | 1.075x |

## Exact command

```bash
python evaluation/benchmark_qwen3_32b_c1_variable_v_decode.py --factor-dir results/checkpoints/qwen3_32b_c1_uniform_v64_tp8_256f64h_full2048_d1e5 --model Qwen/Qwen3-32B --expected-result-sha256 988e939a9f2aea9de2e10dce03896c33367f59115e8832487685bd5dc9c24c56 --layers 0,20,48,63 --ranks all --batches 512 --contexts 1024 --dtype bfloat16 --warmup 5 --iterations 30 --relative-tolerance 0.03 --output-json results/evaluation/qwen3_32b_c1_uniform_v64_variable_v_decode_b512_s1024.json --output-markdown results/evaluation/qwen3_32b_c1_uniform_v64_variable_v_decode_b512_s1024.md --device cuda:0
```
