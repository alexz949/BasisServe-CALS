# Qwen3-8B Wo-only TP4 CUDA Graph: wo_c1_ag

One fixed-context decode step; lower latency is better.

| Batch | Context | Eager mean ms | Graph mean ms | Graph median ms | Graph p95 ms | Graph std ms | Graph tokens/s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 512 | 28.439817 | 9.612262 | 9.605632 | 9.649152 | 0.018901 | 104.034 |
| 8 | 512 | 29.808307 | 11.003964 | 11.003872 | 11.014144 | 0.005683 | 727.011 |
| 32 | 512 | 30.103059 | 14.672071 | 14.664192 | 14.707712 | 0.036288 | 2181.015 |
| 64 | 512 | 30.218445 | 16.536365 | 16.529408 | 16.567200 | 0.035830 | 3870.258 |

## Protocol

- Projection correctness: `passed`
- Warmup runs per mode: `10`
- Timed runs per mode: `50`
- Prefix cache: deterministic zeros, initialized outside timing
- Capture/build time: reported separately and excluded

## Command

```bash
/home/zhangal/.conda/envs/basis/bin/python evaluation/benchmark_qwen3_8b_wo_cuda_graph.py --arm wo_c1_ag --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --phase1-dir /deac/csc/yangGrp/zhangal/BasisServe-CALS/results/checkpoints/qwen3_8b_wo_c1_lr_ar_phase1_tp4_exact_fp64_s100 --quality-results /deac/csc/yangGrp/zhangal/BasisServe-CALS/results/evaluation/qwen3_8b_wo_c1_lr_ar_phase2_quality/results.json --configurations 1x512,8x512,32x512,64x512 --warmup 10 --repeats 50 --prompt-token-id 1 --torch-num-threads 2 --output-dir /deac/csc/yangGrp/zhangal/BasisServe-CALS/results/evaluation/qwen3_8b_wo_cuda_graph_l40s/wo_c1_ag
```
