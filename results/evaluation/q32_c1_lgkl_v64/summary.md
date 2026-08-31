# Qwen3-32B C1 per-layer Global-KL allocation

## Outcome

| Schedule | Confirmation KL | Changed layers | Ragged padding |
|:---|---:|---:|---:|
| uniform_anchor | 0.119749534 | 0 | 0.00% |
| mean_dp | 0.10007378 | 58 | 0.00% |
| ucb_dp | 0.102840345 | 42 | 0.00% |

Selected: **mean_dp**.

## Selected layer ranks

| Layer | Rank for all eight physical KV sources | Collective width |
|---:|---:|---:|
| 0 | 96 | 768 |
| 1 | 64 | 512 |
| 2 | 48 | 384 |
| 3 | 32 | 256 |
| 4 | 48 | 384 |
| 5 | 32 | 256 |
| 6 | 48 | 384 |
| 7 | 32 | 256 |
| 8 | 32 | 256 |
| 9 | 32 | 256 |
| 10 | 32 | 256 |
| 11 | 48 | 384 |
| 12 | 32 | 256 |
| 13 | 32 | 256 |
| 14 | 32 | 256 |
| 15 | 32 | 256 |
| 16 | 32 | 256 |
| 17 | 32 | 256 |
| 18 | 32 | 256 |
| 19 | 32 | 256 |
| 20 | 32 | 256 |
| 21 | 48 | 384 |
| 22 | 48 | 384 |
| 23 | 32 | 256 |
| 24 | 64 | 512 |
| 25 | 32 | 256 |
| 26 | 64 | 512 |
| 27 | 48 | 384 |
| 28 | 32 | 256 |
| 29 | 32 | 256 |
| 30 | 32 | 256 |
| 31 | 48 | 384 |
| 32 | 48 | 384 |
| 33 | 32 | 256 |
| 34 | 32 | 256 |
| 35 | 64 | 512 |
| 36 | 32 | 256 |
| 37 | 96 | 768 |
| 38 | 48 | 384 |
| 39 | 64 | 512 |
| 40 | 80 | 640 |
| 41 | 80 | 640 |
| 42 | 112 | 896 |
| 43 | 112 | 896 |
| 44 | 112 | 896 |
| 45 | 128 | 1024 |
| 46 | 96 | 768 |
| 47 | 128 | 1024 |
| 48 | 128 | 1024 |
| 49 | 96 | 768 |
| 50 | 112 | 896 |
| 51 | 80 | 640 |
| 52 | 112 | 896 |
| 53 | 112 | 896 |
| 54 | 128 | 1024 |
| 55 | 96 | 768 |
| 56 | 96 | 768 |
| 57 | 64 | 512 |
| 58 | 48 | 384 |
| 59 | 80 | 640 |
| 60 | 112 | 896 |
| 61 | 80 | 640 |
| 62 | 96 | 768 |
| 63 | 80 | 640 |

Every layer uses an equal message width on all eight TP ranks; no ragged collective or padding is required.

## Command

`evaluation/run_qwen3_32b_c1_layer_global_kl_sharded.py finalize --model Qwen/Qwen3-32B --windows results/calibration/qwen3_32b_c4_320prefix16gkl_s2048/windows.safetensors --snapshot-dir results/calibration/qwen3_32b_c1_256f64h_full2048_cov --profile-dir results/evaluation/q32_c1_lgkl_v64_profile --factor-dir 32=results/checkpoints/qwen3_32b_c1_aasvd_r32_256f64h_full2048_d1e5 --factor-dir 48=results/checkpoints/qwen3_32b_c1_aasvd_r48_256f64h_full2048_d1e5 --factor-dir 64=results/checkpoints/qwen3_32b_c1_aasvd_r64_256f64h_full2048_d1e5 --factor-dir 80=results/checkpoints/qwen3_32b_c1_aasvd_r80_256f64h_full2048_d1e5 --factor-dir 96=results/checkpoints/qwen3_32b_c1_aasvd_r96_256f64h_full2048_d1e5 --factor-dir 112=results/checkpoints/qwen3_32b_c1_aasvd_r112_256f64h_full2048_d1e5 --anchor-rank 64 --candidate-ranks 32,48,64,80,96,112,128 --window-start 320 --profile-windows 8 --confirmation-windows 8 --sequence-length 512 --batch-size 8 --covariance-damping 1e-5 --vocab-chunk-size 8192 --device-map balanced --max-memory-per-gpu-gib 40 --model-dtype bfloat16 --attn-implementation sdpa --local-files-only --torch-num-threads 4 --profile-shard-count 2 --output-dir results/evaluation/q32_c1_lgkl_v64`
