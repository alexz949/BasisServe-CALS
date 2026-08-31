# Qwen3-32B C1 per-layer Global-KL allocation

## Outcome

| Schedule | Confirmation KL | Changed layers | Ragged padding |
|:---|---:|---:|---:|
| uniform_anchor | 0.109319622 | 0 | 0.00% |
| mean_dp | 0.0953354952 | 55 | 0.00% |
| ucb_dp | 0.0896405836 | 39 | 0.00% |

Selected: **ucb_dp**.

## Selected layer ranks

| Layer | Rank for all eight physical KV sources | Collective width |
|---:|---:|---:|
| 0 | 64 | 512 |
| 1 | 48 | 384 |
| 2 | 64 | 512 |
| 3 | 48 | 384 |
| 4 | 64 | 512 |
| 5 | 32 | 256 |
| 6 | 48 | 384 |
| 7 | 32 | 256 |
| 8 | 48 | 384 |
| 9 | 32 | 256 |
| 10 | 32 | 256 |
| 11 | 48 | 384 |
| 12 | 32 | 256 |
| 13 | 64 | 512 |
| 14 | 32 | 256 |
| 15 | 32 | 256 |
| 16 | 32 | 256 |
| 17 | 32 | 256 |
| 18 | 32 | 256 |
| 19 | 64 | 512 |
| 20 | 32 | 256 |
| 21 | 64 | 512 |
| 22 | 32 | 256 |
| 23 | 32 | 256 |
| 24 | 64 | 512 |
| 25 | 64 | 512 |
| 26 | 64 | 512 |
| 27 | 64 | 512 |
| 28 | 64 | 512 |
| 29 | 48 | 384 |
| 30 | 32 | 256 |
| 31 | 48 | 384 |
| 32 | 64 | 512 |
| 33 | 32 | 256 |
| 34 | 48 | 384 |
| 35 | 64 | 512 |
| 36 | 48 | 384 |
| 37 | 64 | 512 |
| 38 | 64 | 512 |
| 39 | 64 | 512 |
| 40 | 96 | 768 |
| 41 | 64 | 512 |
| 42 | 80 | 640 |
| 43 | 96 | 768 |
| 44 | 96 | 768 |
| 45 | 128 | 1024 |
| 46 | 96 | 768 |
| 47 | 128 | 1024 |
| 48 | 128 | 1024 |
| 49 | 96 | 768 |
| 50 | 112 | 896 |
| 51 | 80 | 640 |
| 52 | 128 | 1024 |
| 53 | 112 | 896 |
| 54 | 112 | 896 |
| 55 | 96 | 768 |
| 56 | 64 | 512 |
| 57 | 64 | 512 |
| 58 | 64 | 512 |
| 59 | 64 | 512 |
| 60 | 64 | 512 |
| 61 | 64 | 512 |
| 62 | 64 | 512 |
| 63 | 64 | 512 |

Every layer uses an equal message width on all eight TP ranks; no ragged collective or padding is required.

## Command

`evaluation/run_qwen3_32b_c1_layer_global_kl_sharded.py finalize --model Qwen/Qwen3-32B --windows results/calibration/qwen3_32b_c4_320prefix16gkl_s2048/windows.safetensors --snapshot-dir results/calibration/qwen3_32b_c1_256f64h_full2048_cov --profile-dir results/evaluation/q32_c1_lgkl_postals_v64_profile --factor-dir 32=results/checkpoints/qwen3_32b_c1_v32_als_256f64h_full2048_d1e5 --factor-dir 48=results/checkpoints/qwen3_32b_c1_v48_als_256f64h_full2048_d1e5 --factor-dir 64=results/checkpoints/qwen3_32b_c1_v64_als_256f64h_full2048_d1e5 --factor-dir 80=results/checkpoints/qwen3_32b_c1_v80_als_256f64h_full2048_d1e5 --factor-dir 96=results/checkpoints/qwen3_32b_c1_v96_als_256f64h_full2048_d1e5 --factor-dir 112=results/checkpoints/qwen3_32b_c1_v112_als_256f64h_full2048_d1e5 --anchor-rank 64 --candidate-ranks 32,48,64,80,96,112,128 --window-start 320 --profile-windows 8 --confirmation-windows 8 --sequence-length 512 --batch-size 8 --covariance-damping 1e-5 --decoder-relative-jitter 0 --vocab-chunk-size 8192 --torch-num-threads 4 --device-map balanced --max-memory-per-gpu-gib 76 --model-dtype bfloat16 --attn-implementation sdpa --local-files-only --profile-shard-count 2 --output-dir results/evaluation/q32_c1_lgkl_postals_v64`
