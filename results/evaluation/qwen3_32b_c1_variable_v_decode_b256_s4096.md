# Qwen3-32B C1 variable-width V-cache decode microbenchmark

This is a single-GPU rank-local PyTorch reference benchmark. It does not measure NCCL, the global decoder, full transformer layers, or serving tokens/s.

- Device: `NVIDIA A100-PCIE-40GB`
- Dtype: `torch.bfloat16`
- Layers: `[0, 20, 48, 63]`
- Virtual TP ranks: `[0, 1, 2, 3, 4, 5, 6, 7]`
- Warmup / iterations: `5 / 30`
- Maximum projection relative L2 error: `0.00368321`
- Maximum attention relative L2 error: `0.00317818`

## Sampled-layer critical path

For each sampled layer, the table takes the slowest selected virtual rank, then sums those layer latencies. Communication is excluded.

| Batch | Context | Dense p50 ms | Compact p50 ms | Speedup | Median rank speedup | Mean V-cache reduction |
|---:|---:|---:|---:|---:|---:|---:|
| 256 | 4096 | 75.9916 | 72.958 | 1.042x | 1.111x | 34.766% |

## Width scaling at the largest workload

Batch `256`, context `4096`; medians are across every occurrence of that source rank in the sampled layers.

| Source rank | V-cache reduction | Attention speedup | End-to-end speedup |
|---:|---:|---:|---:|
| 32 | 75.000% | 1.281x | 1.277x |
| 48 | 62.500% | 1.219x | 1.214x |
| 64 | 50.000% | 1.168x | 1.164x |
| 80 | 37.500% | 1.11x | 1.111x |
| 96 | 25.000% | 1.073x | 1.072x |
| 112 | 12.500% | 1.034x | 1.036x |
| 128 | 0.000% | 1.002x | 1.001x |

## TP maximum-rank straggler at the largest workload

The per-layer critical path is the slowest selected virtual rank. A low mean rank does not improve synchronized TP latency when the layer still contains an uncompressed rank-128 source.

| Layer | Mean rank | Max rank | Dense p50 ms | Compact p50 ms | Speedup |
|---:|---:|---:|---:|---:|---:|
| 0 | 90.000 | 128 | 18.7653 | 18.6665 | 1.005x |
| 20 | 42.000 | 64 | 18.9768 | 16.2335 | 1.169x |
| 48 | 120.000 | 128 | 19.0848 | 19.0013 | 1.004x |
| 63 | 82.000 | 128 | 19.1647 | 19.0566 | 1.006x |

## Exact command

```bash
/home/zhangal/.conda/envs/lowrank/bin/python evaluation/benchmark_qwen3_32b_c1_variable_v_decode.py --factor-dir results/checkpoints/qwen3_32b_c1_aasvd_gkl_v64_ragged_als_10s_256f64h_full2048_d1e5 --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137 --expected-result-sha256 fa34eac71b0f7cca112fd9e071a43c5cb8459d9ec6fd2ddbaff22a19703b4661 --layers 0,20,48,63 --ranks all --batches 256 --contexts 4096 --dtype bfloat16 --warmup 5 --iterations 30 --relative-tolerance 0.03 --output-json results/evaluation/qwen3_32b_c1_variable_v_decode_b256_s4096.json --output-markdown results/evaluation/qwen3_32b_c1_variable_v_decode_b256_s4096.md --device cuda:0
```
