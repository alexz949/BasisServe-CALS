# Qwen3.5-9B C1 layer Global-KL allocation

- Average local rank: `192.0`
- Communication reduction: `81.25%`
- Changed layers: `17`
- Predicted additive profile delta-KL: `-0.016705093`

## Independent confirmation

- Windows: `16` starting at offset `336`; not used for schedule selection.
- Uniform-r192 KL: `0.12387259`
- Ragged KL: `0.11224957`
- Paired delta-KL: `-0.011623021 +/- 0.0032720322`; improved on `14/16` windows.
- Uniform-r192 PPL: `12.01602`
- Ragged PPL: `11.852626`
- Paired delta-NLL: `-0.013691351 +/- 0.0039532102`; improved on `14/16` windows.

## Selected artifacts

- GDN: `results/qwen35_9b_c1/global_kl_r192_grid64/mean_dp/gdn_selected_factors.pt`
- Full attention: `results/qwen35_9b_c1/global_kl_r192_grid64/mean_dp/full_selected_factors.pt`

## Layer schedule

| Layer | Type | Rank |
|---:|:---|---:|
| 0 | GDN | 192 |
| 1 | GDN | 256 |
| 2 | GDN | 256 |
| 3 | Full | 192 |
| 4 | GDN | 192 |
| 5 | GDN | 256 |
| 6 | GDN | 320 |
| 7 | Full | 192 |
| 8 | GDN | 256 |
| 9 | GDN | 192 |
| 10 | GDN | 192 |
| 11 | Full | 128 |
| 12 | GDN | 128 |
| 13 | GDN | 192 |
| 14 | GDN | 128 |
| 15 | Full | 128 |
| 16 | GDN | 192 |
| 17 | GDN | 192 |
| 18 | GDN | 192 |
| 19 | Full | 256 |
| 20 | GDN | 192 |
| 21 | GDN | 192 |
| 22 | GDN | 192 |
| 23 | Full | 192 |
| 24 | GDN | 256 |
| 25 | GDN | 192 |
| 26 | GDN | 128 |
| 27 | Full | 256 |
| 28 | GDN | 128 |
| 29 | GDN | 128 |
| 30 | GDN | 128 |
| 31 | Full | 128 |

## Command

`evaluation/run_qwen35_9b_c1_layer_global_kl.py finalize --model-path Qwen/Qwen3.5-9B --windows results/calibration/qwen35_9b_c4_256f64h16p16c_s512 --factor-root results/qwen35_9b_c1/rank_grid --anchor-gdn-factors results/qwen35_9b_c1/gdn_private_ag_r192_als10_all.pt --anchor-full-factors results/qwen35_9b_c1/full_private_ag_r192_als10_all.pt --profile-dir results/qwen35_9b_c1/global_kl_r192_grid64/profile --output-dir results/qwen35_9b_c1/global_kl_r192_grid64/mean_dp --candidate-ranks 128,192,256,320,384,448,512 --anchor-rank 192 --profile-offset 320 --profile-windows 16 --confirmation-offset 336 --confirmation-windows 16 --batch-size 1 --vocab-chunk-size 8192 --dtype float16 --device cuda:0 --torch-num-threads 2 --local-files-only --profile-shard-count 4`
