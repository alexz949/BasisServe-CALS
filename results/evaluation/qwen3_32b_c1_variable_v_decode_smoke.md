# Qwen3-32B C1 variable-width V-cache decode microbenchmark

This is a single-GPU rank-local PyTorch reference benchmark. It does not measure NCCL, the global decoder, full transformer layers, or serving tokens/s.

- Device: `NVIDIA A100-PCIE-40GB`
- Dtype: `torch.bfloat16`
- Layers: `[0, 20, 48, 63]`
- Virtual TP ranks: `[0, 1, 2, 3, 4, 5, 6, 7]`
- Warmup / iterations: `5 / 30`
- Maximum projection relative L2 error: `0.0045368`
- Maximum attention relative L2 error: `0.00365445`

## Sampled-layer critical path

For each sampled layer, the table takes the slowest selected virtual rank, then sums those layer latencies. Communication is excluded.

| Batch | Context | Dense p50 ms | Compact p50 ms | Speedup | Median rank speedup | Mean V-cache reduction |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 128 | 1.05574 | 0.986624 | 1.07x | 1.04x | 34.766% |
| 1 | 2048 | 1.05626 | 0.999936 | 1.056x | 1.04x | 34.766% |
| 8 | 128 | 1.09978 | 1.06803 | 1.03x | 1.03x | 34.766% |
| 8 | 2048 | 1.61178 | 1.47046 | 1.096x | 1.128x | 34.766% |
| 32 | 128 | 1.08595 | 1.04653 | 1.038x | 1.033x | 34.766% |
| 32 | 2048 | 4.85171 | 4.66739 | 1.039x | 1.086x | 34.766% |

## Width scaling at the largest workload

Batch `32`, context `2048`; medians are across every occurrence of that source rank in the sampled layers.

| Source rank | V-cache reduction | Attention speedup | End-to-end speedup |
|---:|---:|---:|---:|
| 32 | 75.000% | 1.236x | 1.253x |
| 48 | 62.500% | 1.166x | 1.17x |
| 64 | 50.000% | 1.134x | 1.147x |
| 80 | 37.500% | 1.079x | 1.086x |
| 96 | 25.000% | 1.051x | 1.064x |
| 112 | 12.500% | 1.021x | 1.023x |
| 128 | 0.000% | 0.9872x | 0.9983x |

## TP maximum-rank straggler at the largest workload

The per-layer critical path is the slowest selected virtual rank. A low mean rank does not improve synchronized TP latency when the layer still contains an uncompressed rank-128 source.

| Layer | Mean rank | Max rank | Dense p50 ms | Compact p50 ms | Speedup |
|---:|---:|---:|---:|---:|---:|
| 0 | 90.000 | 128 | 1.21139 | 1.20422 | 1.006x |
| 20 | 42.000 | 64 | 1.21651 | 1.03936 | 1.17x |
| 48 | 120.000 | 128 | 1.21139 | 1.21446 | 0.9975x |
| 63 | 82.000 | 128 | 1.21242 | 1.20934 | 1.003x |

## Exact command

```bash
/home/zhangal/.conda/envs/lowrank/bin/python evaluation/benchmark_qwen3_32b_c1_variable_v_decode.py --factor-dir /deac/csc/yangGrp/zhangal/BasisServe-CALS/results/checkpoints/qwen3_32b_c1_aasvd_gkl_v64_ragged_als_10s_256f64h_full2048_d1e5 --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137 --expected-result-sha256 fa34eac71b0f7cca112fd9e071a43c5cb8459d9ec6fd2ddbaff22a19703b4661 --layers 0,20,48,63 --ranks all --batches 1,8,32 --contexts 128,2048 --dtype bfloat16 --warmup 5 --iterations 30 --relative-tolerance 0.03 --output-json /deac/csc/yangGrp/zhangal/BasisServe-CALS/results/evaluation/qwen3_32b_c1_variable_v_decode_smoke.json --output-markdown /deac/csc/yangGrp/zhangal/BasisServe-CALS/results/evaluation/qwen3_32b_c1_variable_v_decode_smoke.md --device cuda:0
```
