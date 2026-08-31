# Qwen3-8B Wo-only TP4 CUDA Graph: wo_lr_ar_capacity

One fixed-context decode step; lower latency is better.

| Batch | Context | Eager mean ms | Graph mean ms | Graph median ms | Graph p95 ms | Graph std ms | Graph tokens/s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 512 | 30.890867 | 9.909179 | 9.906176 | 9.943040 | 0.013046 | 100.917 |
| 8 | 512 | 32.623308 | 11.200907 | 11.196416 | 11.213824 | 0.028160 | 714.228 |
| 32 | 512 | 32.615359 | 15.024673 | 15.020544 | 15.067040 | 0.028436 | 2129.830 |
| 64 | 512 | 32.731863 | 16.951352 | 16.946177 | 16.982016 | 0.027728 | 3775.510 |

## Protocol

- Projection correctness: `passed`
- Warmup runs per mode: `10`
- Timed runs per mode: `50`
- Prefix cache: deterministic zeros, initialized outside timing
- Capture/build time: reported separately and excluded

## Command

```bash
/home/zhangal/.conda/envs/basis/bin/python evaluation/benchmark_qwen3_8b_wo_cuda_graph.py --arm wo_lr_ar_capacity --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --phase1-dir /deac/csc/yangGrp/zhangal/BasisServe-CALS/results/checkpoints/qwen3_8b_wo_c1_lr_ar_phase1_tp4_exact_fp64_s100 --quality-results /deac/csc/yangGrp/zhangal/BasisServe-CALS/results/evaluation/qwen3_8b_wo_c1_lr_ar_phase2_quality/results.json --configurations 1x512,8x512,32x512,64x512 --warmup 10 --repeats 50 --prompt-token-id 1 --torch-num-threads 2 --output-dir /deac/csc/yangGrp/zhangal/BasisServe-CALS/results/evaluation/qwen3_8b_wo_cuda_graph_l40s/wo_lr_ar_capacity
```
