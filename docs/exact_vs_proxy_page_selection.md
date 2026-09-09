# Exact-QK versus proxy selected-page audit

## Protocol

Completed read-only diagnostic on existing C4 captures: layers 15 and 33, eight GQA groups/layer, windows 64–79, all 32 common terminal query positions 24831:256:32767. Each query uses its own full causal prefix. This is 8192 shared query/group conditions and 32768 comparisons across four arms: Q32 Fisher R8, Q64 Fisher R8, Q128 Fisher R8, and the two-window Q32 page-repaired factors.

All arms use the same closed-form Base16 and frozen C1-V80 encoder. Proxy scoring reproduces native BF16 concatenated Base128+R8 sidecar arithmetic. The reference uses FP32 exact QK of the same captured BF16 queries and Keys. Both use the same physical selector: non-sink per-head page-LSE normalization, GQA max, Page32/B2048, page0 pinned, 63 remaining routed pages. No factors were fitted and no model generation, PPL or RULER was run. Offline C1 codes are computed from the dense-teacher Value captures, not a fresh end-to-end C1 rollout.

Exact-QK selection is a same-rule sparse reference, not full attention and not necessarily the set maximizing head-mean teacher mass. Full attention mass is used to measure coverage of either set. Reference/proxy differences include their specified FP32/BF16 arithmetic, not only low-rank approximation.

Smoke 8300779 passed on one L40S (12 seconds). Formal array 8300780, 0–15 with concurrency four, completed on four L40S; each task exited 0:0 in 14–16 seconds. Environment basis, two CPUs and 16 GiB host RAM per worker. Logs: `logs/page-compare-smoke-8300779.{out,err}` and `logs/page-compare-8300780_{0..15}.{out,err}`. Temporary submission scripts were removed. Only the known rotary-embedding deprecation warning appeared.

## Aggregate results

Each row averages 16 windows × 32 queries × 8 groups = 4096 conditions for that layer. Page overlap includes the common pinned page. A non-sink overlap metric is separately saved. Exact mass and proxy mass are head-mean full-teacher attention mass on the respective selected sets, not task accuracy.

| Layer | Router | Page overlap | Mean missed pages / 64 | Proxy teacher mass | Exact-selector teacher mass | Mean mass gap, pp | P95 mass gap, pp |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 15 | q32 | 79.1462% | 13.346 | 94.2170% | 94.7462% | 0.5292 | 2.2108 |
| 15 | q64 | 79.6417% | 13.029 | 94.2542% | 94.7462% | 0.4920 | 1.9186 |
| 15 | q128 | 79.9599% | 12.826 | 94.2723% | 94.7462% | 0.4739 | 1.9505 |
| 15 | q32_repaired | 77.9819% | 14.092 | 94.1282% | 94.7462% | 0.6180 | 2.5634 |
| 33 | q32 | 77.7615% | 14.233 | 92.9133% | 94.1255% | 1.2122 | 4.7506 |
| 33 | q64 | 78.1754% | 13.968 | 92.9548% | 94.1255% | 1.1707 | 4.8512 |
| 33 | q128 | 78.5206% | 13.747 | 92.9995% | 94.1255% | 1.1260 | 4.6881 |
| 33 | q32_repaired | 76.3020% | 15.167 | 92.7034% | 94.1255% | 1.4221 | 5.5332 |

Mass gap is exact-set coverage minus proxy-set coverage, equivalently missed-page mass minus extra-page mass. P95 is the nearest sorted order statistic. Because GQA-max is not a head-mean-mass optimizer, individual mass gaps can be negative. Quantiles over correlated queries are descriptive, not confidence intervals based on independent documents.

Q32 → Q64 → Q128 slightly improves average page overlap and teacher-mass coverage on these same C4 diagnostic queries in both layers. This differs from the previous RULER accuracy ordering; worse RULER scores do not demonstrate that denser Q fitting worsened C4 routing. This audit does not determine the cause of the cross-metric difference.

The repaired arm has lower average diagnostic overlap and coverage than original Q32 in both layers. Its factors were repaired using only two fit windows/four queries per window, unlike the original Fisher bank. These common-Q diagnostic results do not support claiming broad repair improvement.

## Important versus low-mass page differences

For Q32, mean missed-page count is 13.35 in layer 15 and 14.23 in layer 33, while mean full-teacher mass gaps are 0.5292 and 1.2122 percentage points. Page mismatch alone can therefore overstate the typical mass loss. However, the tail includes important omitted pages: 13.77% of layer-15 conditions and 15.87% of layer-33 conditions omit at least one page with exact routed rank-min ≤ 16.

A severe concrete Q32 case is layer 15/group 7, diagnostic window 72, query position 31231 (saved row 281):

| Selector | Captured full-teacher mass |
| --- | ---: |
| Exact QK | 93.6185% |
| Q32 | 69.0520% |
| Q64 | 70.8390% |
| Q128 | 91.0549% |
| Q32 repaired | 68.2902% |

Q32 misses page 908 (token positions 29056–29087), which carries 18.8109% head-mean full-teacher mass. It is ranked 1 by the exact selector but 87 by Q32; only 63 non-pinned pages are selected. Thus this example is not merely a harmless exchange between ranks 63 and 64. The max-score owner also changes from local head 0 to head 1, but that alone does not prove that the GQA-max operator caused the error.

Other omitted pages in the same condition:

| Page ID | Teacher mass | Exact routed rank-min | Q32 routed rank-min |
| --- | ---: | ---: | ---: |
| 908 | 18.8109% | 1 | 87 |
| 823 | 0.8046% | 14 | 105 |
| 949 | 0.6962% | 17 | 92 |
| 948 | 0.6698% | 16 | 70 |
| 879 | 0.4351% | 29 | 76 |

This identifies actual omitted pages and ranking errors. It does not yet separate within-page LSE linearization error, per-head normalization error, GQA ownership changes, or factor-solver effects as their cause.

## Saved page-level records

Each `results/evaluation/page_compare/evaluate/l{layer}_g{group}/pages.safetensors` stores 512 rows keyed by `document` and `query_position`. The valid page range is `0 <= page_id < page_count`; later padded entries must be ignored. Page IDs are zero-based; token start equals 32 × page ID.

- `exact.teacher_mass` and `exact.teacher_non_sink_mass`: head-mean teacher page masses, full and conditional non-sink respectively.
- `{arm}.group_score`: actual normalized GQA-max page score; pinned page0 is excluded from routed ranking.
- `{arm}.owner`: local query-head index 0–3 attaining the maximum.
- `{arm}.rank_min` / `.rank_max`: inclusive rank interval among routed pages, preserving ties; page0 has rank zero.
- `{arm}.cutoff`: score at the 63rd routed page.
- `{arm}.selected` and `.selected_ids`: chosen page mask and actual IDs for exact and each proxy arm.
- `{proxy}.intersection_ids`, `.missed_ids`, `.extra_ids`: exact intersection, exact-minus-proxy, and proxy-minus-exact sets. ID-list padding is -1.

Per-query and aggregate metrics, input hashes, array schema and executed commands are in each neighboring `result.json`. Full token-score arrays are not saved; this is a page-level audit.

All sixteen artifact hashes, selected-set budgets, intersection/missed/extra IDs and masks were independently verified. Teacher-mass coverage was recomputed from saved tensors and compared with recorded metrics for all 32768 arm/query/group comparisons. Three unit tests passed: identical selectors, a controlled page swap with mass accounting, and rank intervals for ties. Production selection is called directly, not replaced by a reimplemented Top-k policy.

## Files and command

[Machine-readable aggregate](../results/evaluation/page_compare/summary.json). [Comparison core](../basisserve/core/page_selection_comparison.py), [driver](../evaluation/compare_residual_selected_pages.py), [tests](../tests/test_page_selection_comparison.py).

Working directory `/deac/csc/yangGrp/zhangal/BasisServe-CALS`; `comparison_layer` takes 15/33 and `comparison_group` takes 0–7. The smoke used `--stage smoke --layer 33 --group 0`; the full run used:

```bash
/home/zhangal/.conda/envs/basis/bin/python -u evaluation/compare_residual_selected_pages.py --stage evaluate --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --c1-checkpoint results/checkpoints/qwen3_8b_c1_v80_32f4h_s32768_als6 --query-capture results/calibration/q32_terminal8k --layer "$comparison_layer" --group "$comparison_group" --output-dir results/evaluation/page_compare
```

No checkpoint was modified. No GitHub commit or push was performed.

