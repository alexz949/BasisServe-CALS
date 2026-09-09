# All-layer exact/proxy page overlap

36 layers, 8 GQA groups, 16 C4 diagnostic windows, 32 common terminal8k Q; Page32/B2048, page0 pinned

Overlap is the intersection size divided by 64, not attention-mass coverage and not IoU. The non-pinned metric subtracts the shared page0 and divides by 63. All means equally weight queries, windows, GQA groups and layers.

The exact and proxy selectors use the same GQA-max policy and budget. The exact reference computes FP32 QK from captured BF16 Q/K; proxy arithmetic is native BF16. This is an offline diagnostic using dense-teacher captures, not an end-to-end accuracy test. Layers 0/1 are also subjected to the sparse rule for this comparison.

147,456 shared conditions per arm; 442,368 arm comparisons. All 288 output hashes, budgets and per-query overlap metrics were checked against saved selected-page masks.

## Overall

| Router | Mean intersection / 64 | Overlap | Excluding pinned page0 |
| --- | ---: | ---: | ---: |
| Q32 | 52.3170 | 81.7453% | 81.4556% |
| UNIFORM8 | 49.9598 | 78.0621% | 77.7139% |
| QGRAM8 | 49.1038 | 76.7247% | 76.3552% |

## Per-layer overlap (including pinned page0)

| Layer | q32 | uniform8 | qgram8 |
| --- | ---: | ---: | ---: |
| 0 | 86.4819% | 81.8020% | 80.1693% |
| 1 | 85.4237% | 81.8523% | 80.3516% |
| 2 | 81.2420% | 78.6537% | 76.8597% |
| 3 | 77.1835% | 72.1703% | 69.5801% |
| 4 | 84.4742% | 81.2389% | 80.8712% |
| 5 | 84.6043% | 79.3804% | 78.2120% |
| 6 | 81.2626% | 78.1200% | 76.7941% |
| 7 | 72.1046% | 68.3655% | 66.7068% |
| 8 | 84.3639% | 81.1954% | 80.3104% |
| 9 | 75.2068% | 71.1422% | 70.8588% |
| 10 | 88.4651% | 85.7693% | 84.8877% |
| 11 | 83.5949% | 80.1453% | 78.6064% |
| 12 | 82.9590% | 79.2248% | 78.4245% |
| 13 | 70.6638% | 66.2228% | 65.3389% |
| 14 | 85.9169% | 82.4257% | 81.9912% |
| 15 | 79.1462% | 74.2626% | 73.3776% |
| 16 | 79.9736% | 75.7877% | 75.4917% |
| 17 | 83.2737% | 80.1495% | 79.6112% |
| 18 | 82.9498% | 79.6822% | 78.5709% |
| 19 | 82.9884% | 79.4250% | 77.4708% |
| 20 | 81.5186% | 77.8278% | 77.7500% |
| 21 | 82.3448% | 78.9867% | 78.2375% |
| 22 | 82.8568% | 78.3001% | 76.2558% |
| 23 | 84.7992% | 80.7556% | 78.0918% |
| 24 | 80.4523% | 76.5579% | 75.3418% |
| 25 | 84.7523% | 81.9111% | 80.4047% |
| 26 | 82.1526% | 78.9722% | 78.1784% |
| 27 | 82.4348% | 79.5326% | 79.0466% |
| 28 | 84.8701% | 81.5594% | 80.6728% |
| 29 | 77.6123% | 74.5846% | 71.5279% |
| 30 | 81.2359% | 77.1080% | 75.3132% |
| 31 | 80.6976% | 76.7357% | 71.2498% |
| 32 | 80.3524% | 76.2024% | 75.1598% |
| 33 | 77.7615% | 73.9662% | 73.3707% |
| 34 | 81.9633% | 78.3504% | 76.6850% |
| 35 | 84.7481% | 81.8707% | 80.3173% |

## Detailed rankings

All valid pages, not just misses, are saved in evaluate/l{layer}_g{group}/pages.safetensors. Rows are keyed by document and query_position; pages >= page_count are padding.
Each arm stores group_score, rank_min/rank_max (inclusive tie intervals), selected IDs/masks, cutoff, and owning GQA head. Page0 is pinned and has rank0; routed cutoff rank is63, not64. Actual selected masks resolve boundary ties.
Exact teacher mass is saved both per head and head-averaged. Scores use non-sink normalized per-head mass then GQA max, so they are not head-mean teacher mass.
missed_page_rankings.csv contains the20 highest head-mean teacher-mass misses per layer for qgram8; this is a diagnostic subset, not the full distribution. It includes all arms, rank intervals, selection categories, head owners and score-minus-cutoff margins.
evaluation/export_page_rankings.py can export every valid page for any saved layer/group/document/query to CSV without model inference.
layer_overlap.csv contains all36 layer averages. Historical exact and q32 tables were verified bitwise against --reference-root.

## Environment and commands

Conda environment: `basis`.

Evaluation command for layer0/group0; layer and group vary over all 36 × 8 combinations:

```bash
evaluation/compare_residual_selected_pages.py --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --c1-checkpoint results/checkpoints/qwen3_8b_c1_v80_32f4h_s32768_als6 --query-capture results/calibration/q32_terminal8k --comparison-bank q32=results/checkpoints/mse_base_q32_r8 --comparison-bank uniform8=results/checkpoints/terminal8_uniform_r8 --comparison-bank qgram8=results/checkpoints/terminal8_qgram_r8 --output-dir results/evaluation/terminal8_pages --stage evaluate --layer 0 --group 0
```

Aggregation command:

```bash
evaluation/summarize_page_overlap.py --root results/evaluation/terminal8_pages --reference-root results/evaluation/page_overlap_all --ranking-arm qgram8
```
