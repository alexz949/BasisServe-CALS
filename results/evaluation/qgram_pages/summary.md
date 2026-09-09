# All-layer exact/proxy page overlap

36 layers, 8 GQA groups, 16 C4 diagnostic windows, 32 common terminal8k Q; Page32/B2048, page0 pinned

Overlap is the intersection size divided by 64, not attention-mass coverage and not IoU. The non-pinned metric subtracts the shared page0 and divides by 63. All means equally weight queries, windows, GQA groups and layers.

The exact and proxy selectors use the same GQA-max policy and budget. The exact reference computes FP32 QK from captured BF16 Q/K; proxy arithmetic is native BF16. This is an offline diagnostic using dense-teacher captures, not an end-to-end accuracy test. Layers 0/1 are also subjected to the sparse rule for this comparison.

147,456 shared conditions per arm; 294,912 arm comparisons. All 288 output hashes, budgets and per-query overlap metrics were checked against saved selected-page masks.

## Overall

| Router | Mean intersection / 64 | Overlap | Excluding pinned page0 |
| --- | ---: | ---: | ---: |
| Q32 | 52.3170 | 81.7453% | 81.4556% |
| QGRAM32 | 51.2775 | 80.1211% | 79.8056% |

## Per-layer overlap (including pinned page0)

| Layer | q32 | qgram32 |
| --- | ---: | ---: |
| 0 | 86.4819% | 84.7504% |
| 1 | 85.4237% | 82.7591% |
| 2 | 81.2420% | 79.6558% |
| 3 | 77.1835% | 73.7259% |
| 4 | 84.4742% | 82.4848% |
| 5 | 84.6043% | 81.0974% |
| 6 | 81.2626% | 79.3518% |
| 7 | 72.1046% | 69.6537% |
| 8 | 84.3639% | 83.0391% |
| 9 | 75.2068% | 73.5325% |
| 10 | 88.4651% | 87.5710% |
| 11 | 83.5949% | 82.5153% |
| 12 | 82.9590% | 81.7768% |
| 13 | 70.6638% | 68.9777% |
| 14 | 85.9169% | 84.5249% |
| 15 | 79.1462% | 77.5928% |
| 16 | 79.9736% | 78.3325% |
| 17 | 83.2737% | 82.0686% |
| 18 | 82.9498% | 81.6933% |
| 19 | 82.9884% | 81.4285% |
| 20 | 81.5186% | 80.3825% |
| 21 | 82.3448% | 81.4716% |
| 22 | 82.8568% | 81.0966% |
| 23 | 84.7992% | 83.4091% |
| 24 | 80.4523% | 79.3343% |
| 25 | 84.7523% | 83.8234% |
| 26 | 82.1526% | 81.1939% |
| 27 | 82.4348% | 81.3660% |
| 28 | 84.8701% | 83.6960% |
| 29 | 77.6123% | 75.8572% |
| 30 | 81.2359% | 79.5341% |
| 31 | 80.6976% | 78.6030% |
| 32 | 80.3524% | 78.3188% |
| 33 | 77.7615% | 76.2817% |
| 34 | 81.9633% | 80.4157% |
| 35 | 84.7481% | 83.0452% |

## Detailed rankings

All valid pages, not just misses, are saved in evaluate/l{layer}_g{group}/pages.safetensors. Rows are keyed by document and query_position; pages >= page_count are padding.
Each arm stores group_score, rank_min/rank_max (inclusive tie intervals), selected IDs/masks, cutoff, and owning GQA head. Page0 is pinned and has rank0; routed cutoff rank is63, not64. Actual selected masks resolve boundary ties.
Exact teacher mass is saved both per head and head-averaged. Scores use non-sink normalized per-head mass then GQA max, so they are not head-mean teacher mass.
missed_page_rankings.csv contains the20 highest head-mean teacher-mass misses per layer for qgram32; this is a diagnostic subset, not the full distribution. It includes all arms, rank intervals, selection categories, head owners and score-minus-cutoff margins.
evaluation/export_page_rankings.py can export every valid page for any saved layer/group/document/query to CSV without model inference.
layer_overlap.csv contains all36 layer averages. Historical exact and q32 tables were verified bitwise against --reference-root.

## Environment and commands

Conda environment: `basis`.

Evaluation command for layer0/group0; layer and group vary over all 36 × 8 combinations:

```bash
evaluation/compare_residual_selected_pages.py --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --c1-checkpoint results/checkpoints/qwen3_8b_c1_v80_32f4h_s32768_als6 --query-capture results/calibration/q32_terminal8k --comparison-bank q32=results/checkpoints/mse_base_q32_r8 --comparison-bank qgram32=results/checkpoints/mse_base_qgram32_r8 --output-dir results/evaluation/qgram_pages --stage evaluate --layer 0 --group 0
```

Aggregation command:

```bash
evaluation/summarize_page_overlap.py --root results/evaluation/qgram_pages --reference-root results/evaluation/page_overlap_all --ranking-arm qgram32
```
