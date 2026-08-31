# Qwen3-8B Wo-only TP4 CUDA Graph: wo_lr_ar_wire

One fixed-context decode step; lower latency is better.

| Batch | Context | Eager mean ms | Graph mean ms | Graph median ms | Graph p95 ms | Graph std ms | Graph tokens/s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 512 | 30.028447 | 9.371053 | 9.360896 | 9.424896 | 0.022968 | 106.712 |
| 8 | 512 | 31.474440 | 10.633007 | 10.629120 | 10.645504 | 0.019701 | 752.374 |
| 32 | 512 | 31.520821 | 13.542952 | 13.540352 | 13.578240 | 0.016054 | 2362.853 |
| 64 | 512 | 31.944770 | 16.084827 | 16.084991 | 16.115711 | 0.020326 | 3978.905 |

## Protocol

- Projection correctness: `passed`
- Warmup runs per mode: `10`
- Timed runs per mode: `50`
- Prefix cache: deterministic zeros, initialized outside timing
- Capture/build time: reported separately and excluded

## Command

```bash
/home/zhangal/.conda/envs/basis/bin/python evaluation/benchmark_qwen3_8b_wo_cuda_graph.py --arm wo_lr_ar_wire --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --phase1-dir /deac/csc/yangGrp/zhangal/BasisServe-CALS/results/checkpoints/qwen3_8b_wo_c1_lr_ar_phase1_tp4_exact_fp64_s100 --quality-results /deac/csc/yangGrp/zhangal/BasisServe-CALS/results/evaluation/qwen3_8b_wo_c1_lr_ar_phase2_quality/results.json --configurations 1x512,8x512,32x512,64x512 --warmup 10 --repeats 50 --prompt-token-id 1 --torch-num-threads 2 --output-dir /deac/csc/yangGrp/zhangal/BasisServe-CALS/results/evaluation/qwen3_8b_wo_cuda_graph_l40s/wo_lr_ar_wire
```
