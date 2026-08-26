# Qwen3-8B-Base C1 per-layer Global-KL allocation

## Outcome

| Schedule | Confirmation KL | Changed layers | Ragged padding |
|:---|---:|---:|---:|
| uniform_anchor | 0.106879798 | 0 | 0.00% |
| mean_dp | 0.0957220894 | 21 | 0.00% |
| ucb_dp | 0.0993615624 | 11 | 0.00% |

Selected: **mean_dp**.

## Selected layer ranks

| Layer | Rank for all eight physical KV sources | Collective width |
|---:|---:|---:|
| 0 | 80 | 640 |
| 1 | 48 | 384 |
| 2 | 32 | 256 |
| 3 | 32 | 256 |
| 4 | 32 | 256 |
| 5 | 48 | 384 |
| 6 | 80 | 640 |
| 7 | 64 | 512 |
| 8 | 80 | 640 |
| 9 | 64 | 512 |
| 10 | 80 | 640 |
| 11 | 80 | 640 |
| 12 | 64 | 512 |
| 13 | 48 | 384 |
| 14 | 64 | 512 |
| 15 | 64 | 512 |
| 16 | 64 | 512 |
| 17 | 64 | 512 |
| 18 | 80 | 640 |
| 19 | 80 | 640 |
| 20 | 64 | 512 |
| 21 | 64 | 512 |
| 22 | 96 | 768 |
| 23 | 112 | 896 |
| 24 | 80 | 640 |
| 25 | 64 | 512 |
| 26 | 64 | 512 |
| 27 | 64 | 512 |
| 28 | 48 | 384 |
| 29 | 64 | 512 |
| 30 | 64 | 512 |
| 31 | 48 | 384 |
| 32 | 48 | 384 |
| 33 | 80 | 640 |
| 34 | 64 | 512 |
| 35 | 32 | 256 |

Every layer uses an equal message width on all eight TP ranks; no ragged collective or padding is required.

## Command

`evaluation/run_qwen3_8b_c1_layer_global_kl_sharded.py finalize --model Qwen/Qwen3-8B-Base --windows results/calibration/qwen3_8b_c4_320prefix16gkl_s2048/windows.safetensors --snapshot-dir results/calibration/qwen3_8b_c1_256f64h_full2048_cov --profile-dir results/evaluation/qwen3_8b_c1_layer_gkl_avg64_profile --factor-dir 32=results/checkpoints/qwen3_8b_c1_v32_als5 --factor-dir 48=results/checkpoints/qwen3_8b_c1_v48_als5 --factor-dir 64=results/checkpoints/qwen3_8b_c1_v64_als5 --factor-dir 80=results/checkpoints/qwen3_8b_c1_v80_als5 --factor-dir 96=results/checkpoints/qwen3_8b_c1_v96_als5 --factor-dir 112=results/checkpoints/qwen3_8b_c1_v112_als5 --anchor-rank 64 --candidate-ranks 32,48,64,80,96,112,128 --window-start 320 --profile-windows 8 --confirmation-windows 8 --sequence-length 2048 --batch-size 8 --covariance-damping 1e-5 --decoder-relative-jitter 0 --vocab-chunk-size 8192 --torch-num-threads 4 --device-map balanced --max-memory-per-gpu-gib 44 --model-dtype bfloat16 --attn-implementation sdpa --local-files-only --profile-shard-count 2 --output-dir results/checkpoints/qwen3_8b_c1_layer_gkl_avg64_als5`
