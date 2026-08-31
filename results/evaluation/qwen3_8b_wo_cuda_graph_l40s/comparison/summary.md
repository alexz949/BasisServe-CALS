# Qwen3-8B Wo-only TP4 five-arm CUDA Graph comparison

All rows use the same model, factors, shapes, dtype, and replay protocol.

| Batch | Context | Arm | Graph mean ms | Graph p95 ms | Tokens/s | Speedup vs dense | Peak GiB/rank |
|---:|---:|:---|---:|---:|---:|---:|---:|
| 1 | 512 | dense | 9.217063 | 9.261056 | 108.494 | 1.0000x | 4.710 |
| 1 | 512 | wo_lr_ar_wire | 9.371053 | 9.424896 | 106.712 | 0.9836x | 5.069 |
| 1 | 512 | wo_lr_ar_capacity | 9.909179 | 9.943040 | 100.917 | 0.9302x | 5.421 |
| 1 | 512 | wo_c1_ag | 9.612262 | 9.649152 | 104.034 | 0.9589x | 5.315 |
| 1 | 512 | wo_c1_local_ar | 9.265754 | 9.277440 | 107.924 | 0.9947x | 4.893 |
| 8 | 512 | dense | 10.535599 | 10.565632 | 759.330 | 1.0000x | 4.843 |
| 8 | 512 | wo_lr_ar_wire | 10.633007 | 10.645504 | 752.374 | 0.9908x | 4.921 |
| 8 | 512 | wo_lr_ar_capacity | 11.200907 | 11.213824 | 714.228 | 0.9406x | 5.273 |
| 8 | 512 | wo_c1_ag | 11.003964 | 11.014144 | 727.011 | 0.9574x | 5.168 |
| 8 | 512 | wo_c1_local_ar | 10.584117 | 10.621952 | 755.850 | 0.9954x | 4.746 |
| 32 | 512 | dense | 14.443126 | 14.504960 | 2215.587 | 1.0000x | 5.278 |
| 32 | 512 | wo_lr_ar_wire | 13.542952 | 13.578240 | 2362.853 | 1.0665x | 5.357 |
| 32 | 512 | wo_lr_ar_capacity | 15.024673 | 15.067040 | 2129.830 | 0.9613x | 5.708 |
| 32 | 512 | wo_c1_ag | 14.672071 | 14.707712 | 2181.015 | 0.9844x | 5.603 |
| 32 | 512 | wo_c1_local_ar | 14.443844 | 14.496768 | 2215.477 | 1.0000x | 5.181 |
| 64 | 512 | dense | 16.521152 | 16.588800 | 3873.822 | 1.0000x | 5.857 |
| 64 | 512 | wo_lr_ar_wire | 16.084827 | 16.115711 | 3978.905 | 1.0271x | 5.935 |
| 64 | 512 | wo_lr_ar_capacity | 16.951352 | 16.982016 | 3775.510 | 0.9746x | 6.287 |
| 64 | 512 | wo_c1_ag | 16.536365 | 16.567200 | 3870.258 | 0.9991x | 6.182 |
| 64 | 512 | wo_c1_local_ar | 16.551686 | 16.620544 | 3866.676 | 0.9982x | 5.759 |

`wo_c1_ag` and `wo_c1_local_ar` use exactly the same C1 factors and approximate function; their difference isolates the collective boundary.
All five arms retain dense V and a dense KV cache.
