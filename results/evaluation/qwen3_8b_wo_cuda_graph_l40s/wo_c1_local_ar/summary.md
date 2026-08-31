# Qwen3-8B Wo-only TP4 CUDA Graph: wo_c1_local_ar

One fixed-context decode step; lower latency is better.

| Batch | Context | Eager mean ms | Graph mean ms | Graph median ms | Graph p95 ms | Graph std ms | Graph tokens/s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 512 | 30.364037 | 9.265754 | 9.261056 | 9.277440 | 0.021659 | 107.924 |
| 8 | 512 | 31.580773 | 10.584117 | 10.578944 | 10.621952 | 0.018690 | 755.850 |
| 32 | 512 | 31.366916 | 14.443844 | 14.436336 | 14.496768 | 0.026660 | 2215.477 |
| 64 | 512 | 32.214654 | 16.551686 | 16.531456 | 16.620544 | 0.127303 | 3866.676 |

## Protocol

- Projection correctness: `passed`
- Warmup runs per mode: `10`
- Timed runs per mode: `50`
- Prefix cache: deterministic zeros, initialized outside timing
- Capture/build time: reported separately and excluded

## Command

```bash
/home/zhangal/.conda/envs/basis/bin/python evaluation/benchmark_qwen3_8b_wo_cuda_graph.py --arm wo_c1_local_ar --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --phase1-dir /deac/csc/yangGrp/zhangal/BasisServe-CALS/results/checkpoints/qwen3_8b_wo_c1_lr_ar_phase1_tp4_exact_fp64_s100 --quality-results /deac/csc/yangGrp/zhangal/BasisServe-CALS/results/evaluation/qwen3_8b_wo_c1_lr_ar_phase2_quality/results.json --configurations 1x512,8x512,32x512,64x512 --warmup 10 --repeats 50 --prompt-token-id 1 --torch-num-threads 2 --output-dir /deac/csc/yangGrp/zhangal/BasisServe-CALS/results/evaluation/qwen3_8b_wo_cuda_graph_l40s/wo_c1_local_ar
```
