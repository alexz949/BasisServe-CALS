# DeepSeek-V2-Lite C1 per-layer Global-KL allocation

- MLA latent KV and KV cache are unchanged.
- Exact average source rank: `128`.
- Attention-output reduction versus dense AllGather: `50%`.

| Schedule | Confirmation KL | WikiText-2 PPL | Changed layers |
|:---|---:|---:|---:|
| uniform_anchor | 0.0404588516 | 6.65634968 | 0 |
| mean_dp | 0.0323032056 | 6.69518284 | 17 |
| ucb_dp | 0.0331719558 |  | 15 |

Selected: **mean_dp**.

| Layer | Source rank | Gathered width |
|---:|---:|---:|
| 0 | 64 | 512 |
| 1 | 96 | 768 |
| 2 | 128 | 1024 |
| 3 | 128 | 1024 |
| 4 | 128 | 1024 |
| 5 | 128 | 1024 |
| 6 | 160 | 1280 |
| 7 | 128 | 1024 |
| 8 | 192 | 1536 |
| 9 | 160 | 1280 |
| 10 | 192 | 1536 |
| 11 | 192 | 1536 |
| 12 | 160 | 1280 |
| 13 | 160 | 1280 |
| 14 | 128 | 1024 |
| 15 | 128 | 1024 |
| 16 | 128 | 1024 |
| 17 | 128 | 1024 |
| 18 | 64 | 512 |
| 19 | 192 | 1536 |
| 20 | 96 | 768 |
| 21 | 64 | 512 |
| 22 | 96 | 768 |
| 23 | 128 | 1024 |
| 24 | 160 | 1280 |
| 25 | 64 | 512 |
| 26 | 64 | 512 |

## Command

`evaluation/run_deepseek_v2_lite_c1_layer_global_kl.py finalize --model deepseek-ai/DeepSeek-V2-Lite --windows results/deepseek_v2_lite_c1/c4_256fit_64val_8profile_8confirm_seq2048/windows.safetensors --snapshot-dir results/deepseek_v2_lite_c1/cov_256fit_64val_seq2048_bf16 --profile-dir results/deepseek_v2_lite_c1/global_kl_avg128_profile --factor-dir 64=results/deepseek_v2_lite_c1/c1_tp8_r64_als5 --factor-dir 96=results/deepseek_v2_lite_c1/c1_tp8_r96_als5 --factor-dir 128=results/deepseek_v2_lite_c1/c1_tp8_r128_als5 --factor-dir 160=results/deepseek_v2_lite_c1/c1_tp8_r160_als5 --factor-dir 192=results/deepseek_v2_lite_c1/c1_tp8_r192_als5 --anchor-rank 128 --candidate-ranks 64,96,128,160,192 --window-start 320 --profile-windows 8 --confirmation-windows 8 --sequence-length 2048 --batch-size 8 --vocab-chunk-size 8192 --device-map balanced --max-memory-per-gpu-gib 44 --model-dtype bfloat16 --attn-implementation sdpa --torch-num-threads 4 --profile-shard-count 2 --output-dir results/deepseek_v2_lite_c1/c1_tp8_avg128_global_kl_als5 --eval-seqlen 2048 --eval-batch-size 4`
