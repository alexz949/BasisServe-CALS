# Qwen3-8B Wo-only TP4 CUDA Graph: dense

One fixed-context decode step; lower latency is better.

| Batch | Context | Eager mean ms | Graph mean ms | Graph median ms | Graph p95 ms | Graph std ms | Graph tokens/s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 512 | 30.332303 | 9.217063 | 9.210368 | 9.261056 | 0.018931 | 108.494 |
| 8 | 512 | 31.676943 | 10.535599 | 10.529792 | 10.565632 | 0.023400 | 759.330 |
| 32 | 512 | 31.937589 | 14.443126 | 14.449664 | 14.504960 | 0.044158 | 2215.587 |
| 64 | 512 | 32.059693 | 16.521152 | 16.516096 | 16.588800 | 0.028904 | 3873.822 |

## Protocol

- Projection correctness: `not_applicable_dense_reference`
- Warmup runs per mode: `10`
- Timed runs per mode: `50`
- Prefix cache: deterministic zeros, initialized outside timing
- Capture/build time: reported separately and excluded

## Command

```bash
/home/zhangal/.conda/envs/basis/bin/python evaluation/benchmark_qwen3_8b_wo_cuda_graph.py --arm dense --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --phase1-dir /deac/csc/yangGrp/zhangal/BasisServe-CALS/results/checkpoints/qwen3_8b_wo_c1_lr_ar_phase1_tp4_exact_fp64_s100 --quality-results /deac/csc/yangGrp/zhangal/BasisServe-CALS/results/evaluation/qwen3_8b_wo_c1_lr_ar_phase2_quality/results.json --configurations 1x512,8x512,32x512,64x512 --warmup 10 --repeats 50 --prompt-token-id 1 --torch-num-threads 2 --output-dir /deac/csc/yangGrp/zhangal/BasisServe-CALS/results/evaluation/qwen3_8b_wo_cuda_graph_l40s/dense_v2
```
