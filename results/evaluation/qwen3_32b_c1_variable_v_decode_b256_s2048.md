# Qwen3-32B C1 variable-width V-cache decode microbenchmark

This is a single-GPU rank-local PyTorch reference benchmark. It does not measure NCCL, the global decoder, full transformer layers, or serving tokens/s.

- Device: `NVIDIA A100-PCIE-40GB`
- Dtype: `torch.bfloat16`
- Layers: `[0, 20, 48, 63]`
- Virtual TP ranks: `[0, 1, 2, 3, 4, 5, 6, 7]`
- Warmup / iterations: `5 / 30`
- Maximum projection relative L2 error: `0.00365318`
- Maximum attention relative L2 error: `0.00319666`

## Sampled-layer critical path

For each sampled layer, the table takes the slowest selected virtual rank, then sums those layer latencies. Communication is excluded.

| Batch | Context | Dense p50 ms | Compact p50 ms | Speedup | Median rank speedup | Mean V-cache reduction |
|---:|---:|---:|---:|---:|---:|---:|
| 256 | 2048 | 37.7329 | 36.1236 | 1.045x | 1.108x | 34.766% |

## Width scaling at the largest workload

Batch `256`, context `2048`; medians are across every occurrence of that source rank in the sampled layers.

| Source rank | V-cache reduction | Attention speedup | End-to-end speedup |
|---:|---:|---:|---:|
| 32 | 75.000% | 1.271x | 1.269x |
| 48 | 62.500% | 1.214x | 1.217x |
| 64 | 50.000% | 1.174x | 1.173x |
| 80 | 37.500% | 1.107x | 1.108x |
| 96 | 25.000% | 1.071x | 1.073x |
| 112 | 12.500% | 1.033x | 1.036x |
| 128 | 0.000% | 1.002x | 1.003x |

## TP maximum-rank straggler at the largest workload

The per-layer critical path is the slowest selected virtual rank. A low mean rank does not improve synchronized TP latency when the layer still contains an uncompressed rank-128 source.

| Layer | Mean rank | Max rank | Dense p50 ms | Compact p50 ms | Speedup |
|---:|---:|---:|---:|---:|---:|
| 0 | 90.000 | 128 | 9.41158 | 9.3143 | 1.01x |
| 20 | 42.000 | 64 | 9.40544 | 7.93702 | 1.185x |
| 48 | 120.000 | 128 | 9.42438 | 9.38957 | 1.004x |
| 63 | 82.000 | 128 | 9.49146 | 9.48275 | 1.001x |

## Exact command

```bash
/home/zhangal/.conda/envs/lowrank/bin/python evaluation/benchmark_qwen3_32b_c1_variable_v_decode.py --factor-dir results/checkpoints/qwen3_32b_c1_aasvd_gkl_v64_ragged_als_10s_256f64h_full2048_d1e5 --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137 --expected-result-sha256 fa34eac71b0f7cca112fd9e071a43c5cb8459d9ec6fd2ddbaff22a19703b4661 --layers 0,20,48,63 --ranks all --batches 256 --contexts 2048 --dtype bfloat16 --warmup 5 --iterations 30 --relative-tolerance 0.03 --output-json results/evaluation/qwen3_32b_c1_variable_v_decode_b256_s2048.json --output-markdown results/evaluation/qwen3_32b_c1_variable_v_decode_b256_s2048.md --device cuda:0
```
