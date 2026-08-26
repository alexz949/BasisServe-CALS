# Qwen3-32B C1 variable-width V-cache decode microbenchmark

This is a single-GPU rank-local PyTorch reference benchmark. It does not measure NCCL, the global decoder, full transformer layers, or serving tokens/s.

- Device: `NVIDIA A100 80GB PCIe`
- Dtype: `torch.bfloat16`
- Layers: `[0, 20, 48, 63]`
- Virtual TP ranks: `[0, 1, 2, 3, 4, 5, 6, 7]`
- Warmup / iterations: `5 / 30`
- Maximum projection relative L2 error: `0.00403026`
- Maximum attention relative L2 error: `0.00319308`

## Sampled-layer critical path

For each sampled layer, the table takes the slowest selected virtual rank, then sums those layer latencies. Communication is excluded.

| Batch | Context | Dense p50 ms | Compact p50 ms | Speedup | Median rank speedup | Mean V-cache reduction |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 1024 | 75.9475 | 74.5887 | 1.018x | 1.068x | 34.766% |

## Width scaling at the largest workload

Batch `512`, context `1024`; medians are across every occurrence of that source rank in the sampled layers.

| Source rank | V-cache reduction | Attention speedup | End-to-end speedup |
|---:|---:|---:|---:|
| 32 | 75.000% | 1.242x | 1.294x |
| 48 | 62.500% | 1.223x | 1.214x |
| 64 | 50.000% | 1.069x | 1.07x |
| 80 | 37.500% | 1.101x | 1.102x |
| 96 | 25.000% | 1.12x | 1.087x |
| 112 | 12.500% | 1.031x | 1.022x |
| 128 | 0.000% | 0.9952x | 1.002x |

## TP maximum-rank straggler at the largest workload

The per-layer critical path is the slowest selected virtual rank. A low mean rank does not improve synchronized TP latency when the layer still contains an uncompressed rank-128 source.

| Layer | Mean rank | Max rank | Dense p50 ms | Compact p50 ms | Speedup |
|---:|---:|---:|---:|---:|---:|
| 0 | 90.000 | 128 | 18.9891 | 18.9071 | 1.004x |
| 20 | 42.000 | 64 | 18.9763 | 17.7423 | 1.07x |
| 48 | 120.000 | 128 | 18.985 | 19.0075 | 0.9988x |
| 63 | 82.000 | 128 | 18.9972 | 18.9317 | 1.003x |

## Exact command

```bash
python evaluation/benchmark_qwen3_32b_c1_variable_v_decode.py --factor-dir results/checkpoints/qwen3_32b_c1_aasvd_gkl_v64_ragged_als_10s_256f64h_full2048_d1e5 --model Qwen/Qwen3-32B --expected-result-sha256 fa34eac71b0f7cca112fd9e071a43c5cb8459d9ec6fd2ddbaff22a19703b4661 --layers 0,20,48,63 --ranks all --batches 512 --contexts 1024 --dtype bfloat16 --warmup 5 --iterations 30 --relative-tolerance 0.03 --output-json results/evaluation/qwen3_32b_c1_variable_v_decode_b512_s1024.json --output-markdown results/evaluation/qwen3_32b_c1_variable_v_decode_b512_s1024.md --device cuda:0
```
