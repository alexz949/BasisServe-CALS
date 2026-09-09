# All-layer exact/proxy page overlap

36 layers, 8 GQA groups, 16 C4 diagnostic windows, 32 common terminal8k Q; Page32/B2048, page0 pinned

Overlap is the intersection size divided by 64, not attention-mass coverage and not IoU. The non-pinned metric subtracts the shared page0 and divides by 63. All means equally weight queries, windows, GQA groups and layers.

The exact and proxy selectors use the same GQA-max policy and budget. The exact reference computes FP32 QK from captured BF16 Q/K; proxy arithmetic is native BF16. This is an offline diagnostic using dense-teacher captures, not an end-to-end accuracy test. Layers 0/1 are also subjected to the sparse rule for this comparison.

147,456 shared conditions per arm; 442,368 arm comparisons. All 288 output hashes, budgets and per-query overlap metrics were checked against saved selected-page masks.

## Overall

| Router | Mean intersection / 64 | Overlap | Excluding pinned page0 |
| --- | ---: | ---: | ---: |
| Q32 | 52.3170 | 81.7453% | 81.4556% |
| Q64 | 52.5616 | 82.1275% | 81.8438% |
| Q128 | 52.7143 | 82.3661% | 82.0862% |

## Per-layer overlap (including pinned page0)

| Layer | Q32 | Q64 | Q128 |
| --- | ---: | ---: | ---: |
| 0 | 86.4819% | 86.9194% | 87.6438% |
| 1 | 85.4237% | 85.7979% | 86.1519% |
| 2 | 81.2420% | 81.4644% | 81.6475% |
| 3 | 77.1835% | 77.7790% | 78.0087% |
| 4 | 84.4742% | 84.4372% | 84.4810% |
| 5 | 84.6043% | 84.3090% | 84.2468% |
| 6 | 81.2626% | 81.4667% | 81.4903% |
| 7 | 72.1046% | 72.4140% | 72.6898% |
| 8 | 84.3639% | 84.7775% | 85.0147% |
| 9 | 75.2068% | 75.6813% | 76.0136% |
| 10 | 88.4651% | 88.8210% | 89.0240% |
| 11 | 83.5949% | 84.0569% | 84.3704% |
| 12 | 82.9590% | 83.4583% | 83.7658% |
| 13 | 70.6638% | 71.1792% | 71.5065% |
| 14 | 85.9169% | 86.2804% | 86.5036% |
| 15 | 79.1462% | 79.6417% | 79.9599% |
| 16 | 79.9736% | 80.4569% | 80.7251% |
| 17 | 83.2737% | 83.6891% | 83.9443% |
| 18 | 82.9498% | 83.2714% | 83.4236% |
| 19 | 82.9884% | 83.3664% | 83.5594% |
| 20 | 81.5186% | 82.0393% | 82.4047% |
| 21 | 82.3448% | 82.8735% | 83.1059% |
| 22 | 82.8568% | 83.4515% | 83.7124% |
| 23 | 84.7992% | 85.1017% | 85.1543% |
| 24 | 80.4523% | 81.2038% | 81.4571% |
| 25 | 84.7523% | 85.0132% | 85.1013% |
| 26 | 82.1526% | 82.5550% | 82.8396% |
| 27 | 82.4348% | 82.6786% | 83.2153% |
| 28 | 84.8701% | 85.3409% | 85.4816% |
| 29 | 77.6123% | 78.2024% | 78.3726% |
| 30 | 81.2359% | 81.4438% | 81.6269% |
| 31 | 80.6976% | 81.0745% | 81.2485% |
| 32 | 80.3524% | 80.7564% | 80.8270% |
| 33 | 77.7615% | 78.1754% | 78.5206% |
| 34 | 81.9633% | 82.4116% | 82.7957% |
| 35 | 84.7481% | 84.9991% | 85.1463% |

## Environment and commands

Conda environment: `basis`.

Evaluation command for layer0/group0; layer and group vary over all 36 × 8 combinations:

```bash
evaluation/compare_residual_selected_pages.py --stage evaluate --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --c1-checkpoint results/checkpoints/qwen3_8b_c1_v80_32f4h_s32768_als6 --query-capture results/calibration/q32_terminal8k --layer 0 --group 0 --output-dir results/evaluation/page_overlap_all
```

Aggregation command:

```bash
evaluation/summarize_page_overlap.py --root results/evaluation/page_overlap_all
```
