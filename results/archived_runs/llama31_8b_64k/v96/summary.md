# Llama-3.1-8B C1 per-layer Global-KL allocation

## Outcome

| Schedule | Confirmation KL | Changed layers | Ragged padding |
|:---|---:|---:|---:|
| uniform_anchor | 0.497666554 | 0 | 0.00% |
| two_sided_factorized_kl | 0.0557250578 | 28 | 0.00% |

Selected: **two_sided_factorized_kl**.

## Selected layer ranks

| Layer | Rank for all eight physical KV sources | Collective width |
|---:|---:|---:|
| 0 | 128 | 1024 |
| 1 | 128 | 1024 |
| 2 | 64 | 512 |
| 3 | 96 | 768 |
| 4 | 128 | 1024 |
| 5 | 128 | 1024 |
| 6 | 128 | 1024 |
| 7 | 128 | 1024 |
| 8 | 128 | 1024 |
| 9 | 96 | 768 |
| 10 | 128 | 1024 |
| 11 | 64 | 512 |
| 12 | 96 | 768 |
| 13 | 128 | 1024 |
| 14 | 128 | 1024 |
| 15 | 128 | 1024 |
| 16 | 128 | 1024 |
| 17 | 128 | 1024 |
| 18 | 64 | 512 |
| 19 | 64 | 512 |
| 20 | 64 | 512 |
| 21 | 64 | 512 |
| 22 | 80 | 640 |
| 23 | 64 | 512 |
| 24 | 96 | 768 |
| 25 | 64 | 512 |
| 26 | 64 | 512 |
| 27 | 64 | 512 |
| 28 | 112 | 896 |
| 29 | 64 | 512 |
| 30 | 64 | 512 |
| 31 | 64 | 512 |

Every layer uses an equal message width on all eight TP ranks; no ragged collective or padding is required.

## Command

`/deac/csc/yangGrp/zhangal/BasisServe-CALS/evaluation/run_llama31_8b_c1_two_sided_factorized_kl_sharded.py finalize --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b --windows /home/zhangal/BasisServe-CALS-runs/llama31_8b_64k/calibration/windows.safetensors --snapshot-dir /home/zhangal/BasisServe-CALS-runs/llama31_8b_64k/covariance --profile-dir /home/zhangal/BasisServe-CALS-runs/llama31_8b_64k/kl_profile --factor-dir 64=/home/zhangal/BasisServe-CALS-runs/llama31_8b_64k/v_bank_ieee/R64 --factor-dir 80=/home/zhangal/BasisServe-CALS-runs/llama31_8b_64k/v_bank_ieee/R80 --factor-dir 96=/home/zhangal/BasisServe-CALS-runs/llama31_8b_64k/v_bank_ieee/R96 --factor-dir 112=/home/zhangal/BasisServe-CALS-runs/llama31_8b_64k/v_bank_ieee/R112 --anchor-rank 96 --candidate-ranks 64,80,96,112,128 --probe-source fit --window-start 0 --profile-windows 8 --confirmation-windows 8 --sequence-length 65536 --batch-size 1 --covariance-damping 1e-7 --factorized-probe-rank 112 --factorized-compression-probe-rank 80 --local-error-split fit --profile-backend sampled_suffix --terminal-position-counts 64,128,256,512,1024 --profile-shard-count 4 --torch-num-threads 2 --max-memory-per-gpu-gib 44 --local-files-only --output-dir /home/zhangal/BasisServe-CALS-runs/llama31_8b_64k/v96 --target-average-rank 96 --force-selected-candidate two_sided_factorized_kl --mlp-chunk-size 1024`
