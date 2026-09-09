# Stratified Query-Gram pivot sampler: implementation and smoke

## Scope

Completed on one NVIDIA L40S using the basis environment: job 8300824, 70 seconds, exit 0:0. This is a two-fit-window smoke (C4 windows 0 and 1), layer 15. Query geometry uses all 32 query heads; residual-statistics integration uses GQA group 0 (four query heads). It is NOT the intended 64-window sampler, a fitted residual checkpoint, or a RULER evaluation.

Frozen C1-V80 and closed-form MSE-RRR Base16 were reused. True post-RoPE residuals, Page32, excluded pinned page0, and the existing Page-Fisher statistics function are unchanged. No Adam, autograd, BCD update, new Base fit, downstream answers, or task metadata were used in this smoke. R8/B2048 remain the target configuration; no new R8 factors were produced.

## Implemented method

Each layer independently estimates uncentered per-head second moments from FIT candidate queries, computes an eigendecomposition-based pseudoinverse square root, and constructs four candidate-position Grams. Accumulation, whitening computation, Grams and pivots use FP64; saved whitening is FP32. Relative eigenvalue threshold: 1e-6. Each head retained all 128 directions in this smoke.

Candidates are stride-block ends 63, 127, ..., 32767. Each of four 8K bins has 128 candidates and selects exactly eight pivots. Sorted selected positions are shared by every window within the layer. Exact ties choose the lowest absolute position; numerically exhausted pivots use deterministic zero-energy completion. No position is filtered based on the B2048 budget.

The implementation is greedy pivoted Cholesky, not strong RRQR or classical leverage-score sampling. The implicit concatenated multi-window/head vectors are not materialized. General context lengths, bin counts, per-bin counts and candidate strides are supported by the sampler.

## Selected positions

| Bin | Sorted selected positions |
| --- | --- |
| 0 | 1535, 3007, 3903, 5055, 5311, 7615, 7935, 7999 |
| 1 | 8255, 8959, 9919, 10687, 11071, 11455, 11519, 15103 |
| 2 | 17407, 19263, 19583, 19775, 19967, 20543, 21055, 23871 |
| 3 | 26303, 28799, 29247, 30719, 31743, 31935, 32063, 32319 |

One selected query is at position1535 (prefix length1536), where B2048 would cover the full prefix. This is reported, not silently excluded.

## Geometry diagnostics

| Bin | Mean absolute off-diagonal correlation | Residual trace after 8 pivots | Overlap with top8 original diagonal |
| --- | ---: | ---: | ---: |
| 0 | 0.043332 | 90.8752% | 7/8 |
| 1 | 0.040229 | 91.0753% | 8/8 |
| 2 | 0.041479 | 91.1648% | 7/8 |
| 3 | 0.042975 | 90.8244% | 6/8 |

Residual trace refers only to the candidate Query Gram, not attention-mass loss, routing recall, or task accuracy. The weak cross-position correlations and 28/32 overlap with largest-diagonal selection show limited diversification beyond whitened energy ranking in THIS TWO-WINDOW sample. No conclusion about the 64-window method follows from this smoke alone.

## Pivot diagnostics

Energy is the squared residual feature norm immediately BEFORE adding that pivot. Pivot order is local to its bin, zero-based.

| Bin | Pivot order | Position | Residual energy | Original energy | Relative residual energy |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 3007 | 154.018590 | 154.018590 | 1.000000 |
| 0 | 1 | 7615 | 146.966383 | 147.057846 | 0.999378 |
| 0 | 2 | 5055 | 146.770825 | 146.850101 | 0.999460 |
| 0 | 3 | 5311 | 144.463518 | 145.766468 | 0.991061 |
| 0 | 4 | 1535 | 141.731013 | 141.916345 | 0.998694 |
| 0 | 5 | 7999 | 141.004141 | 142.958318 | 0.986330 |
| 0 | 6 | 3903 | 138.951233 | 141.002048 | 0.985455 |
| 0 | 7 | 7935 | 137.629983 | 139.480272 | 0.986734 |
| 1 | 0 | 8959 | 157.531280 | 157.531280 | 1.000000 |
| 1 | 1 | 8255 | 155.614486 | 155.634319 | 0.999873 |
| 1 | 2 | 11071 | 152.884508 | 153.016127 | 0.999140 |
| 1 | 3 | 11519 | 151.989370 | 152.293437 | 0.998003 |
| 1 | 4 | 10687 | 148.927940 | 155.251980 | 0.959266 |
| 1 | 5 | 9919 | 148.723342 | 150.420638 | 0.988716 |
| 1 | 6 | 15103 | 145.991473 | 147.483794 | 0.989881 |
| 1 | 7 | 11455 | 144.481052 | 148.214019 | 0.974814 |
| 2 | 0 | 17407 | 153.064226 | 153.064226 | 1.000000 |
| 2 | 1 | 21055 | 151.196341 | 151.311375 | 0.999240 |
| 2 | 2 | 23871 | 146.056067 | 146.493859 | 0.997012 |
| 2 | 3 | 20543 | 145.365955 | 146.313655 | 0.993523 |
| 2 | 4 | 19583 | 144.193565 | 144.330793 | 0.999049 |
| 2 | 5 | 19967 | 139.272290 | 140.811687 | 0.989068 |
| 2 | 6 | 19263 | 138.419359 | 143.186236 | 0.966709 |
| 2 | 7 | 19775 | 136.492451 | 138.713580 | 0.983988 |
| 3 | 0 | 31743 | 168.869075 | 168.869075 | 1.000000 |
| 3 | 1 | 29247 | 157.563760 | 157.986966 | 0.997321 |
| 3 | 2 | 28799 | 155.373887 | 156.193670 | 0.994751 |
| 3 | 3 | 31935 | 149.071737 | 149.755840 | 0.995432 |
| 3 | 4 | 30719 | 146.947098 | 147.398471 | 0.996938 |
| 3 | 5 | 32063 | 145.725961 | 147.648844 | 0.986977 |
| 3 | 6 | 32319 | 143.838097 | 146.706182 | 0.980450 |
| 3 | 7 | 26303 | 142.280302 | 143.563881 | 0.991059 |

## Integration and validation

All three same-Q-count policies completed: terminal_uniform_q32, full_uniform_q32, stratified_query_gram_pivot. Each built [4, 64, 128, 128] finite Page-Fisher Grams (4 heads × 32 queries × 2 windows). Query-major/document-minor ordering was checked exactly; the first query/window Gram was independently reconstructed and matched. The Base checkpoint was not modified.

Teacher Fisher energies were 51.5303098 (terminal), 49.2132179 (full uniform), and 53.0570226 (pivot). These use DIFFERENT Q and therefore are not comparable fitting losses or evidence that one policy is more accurate.

Unit tests cover explicit greedy residual equivalence, zero/rank-deficient/tied Grams, PSD/symmetry, singular whitening, candidate invariants, deterministic selection hashes, fit-only loading with a poisoned diagnostic file, and per-layer selected-capture loading for fit/diagnostic splits. Candidate Q overlapping existing terminal-Q128 fit captures agreed bitwise.

Candidate mode stores Q only. Manifest-selected capture mode reuses selected candidate rows for fit windows and supports fresh diagnostic capture after positions are frozen. The fitter accepts --query-position-manifest; formal fitting requires all36 layers and 64fit+16diagnostic observations. That formal capture/fitting path has not been run in this smoke. No full RULER run was submitted.

Candidate/selected capture now records per-document token hashes; formal fitting checks them against its actual K/V source windows. The two-window fit-token hashes were independently verified after adding these provenance checks.

## Files

- [Sampling core](../basisserve/core/query_position_sampling.py)
- [Existing capture CLI](../scripts/capture_qwen3_8b_q16.py) and [Q-only helper](../scripts/query_position_capture.py)
- [Manifest selection/loading](../evaluation/select_query_positions.py)
- [Residual fitter integration](../evaluation/fit_qwen3_8b_q8_fisher_residual.py)
- [Tests](../tests/test_query_position_sampling.py)
- [Selected positions manifest](../results/evaluation/query_gram_smoke/stratified_query_gram_pivot/positions.json)

Logs: logs/query-gram-smoke-8300824.out and .err. Only the known rotary-embedding device deprecation warning appeared; no job failures. Temporary Slurm submission script removed. No GitHub commit or push.

## Executed commands

Working directory: /deac/csc/yangGrp/zhangal/BasisServe-CALS.

```bash
/home/zhangal/.conda/envs/basis/bin/python -u scripts/capture_qwen3_8b_q16.py --stage candidates --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --candidate-window-count 2 --candidate-layers 15 --candidate-stride 64 --num-shards 1 --output-dir results/calibration/query_gram_smoke
```

```bash
/home/zhangal/.conda/envs/basis/bin/python -u evaluation/select_query_positions.py --candidate-capture results/calibration/query_gram_smoke --output-dir results/evaluation/query_gram_smoke/stratified_query_gram_pivot --policy stratified_query_gram_pivot --device cpu
```

```bash
/home/zhangal/.conda/envs/basis/bin/python -u evaluation/fit_qwen3_8b_q8_fisher_residual.py --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --c1-checkpoint results/checkpoints/qwen3_8b_c1_v80_32f4h_s32768_als6 --initial-bank results/checkpoints/q8_residual_kl_bank --base-kind closed_form_rrr --query-capture results/calibration/query_gram_smoke --query-position-manifest results/evaluation/query_gram_smoke/stratified_query_gram_pivot/positions.json --sampler-smoke --smoke-layer 15 --smoke-group 0 --num-shards 1 --output-dir results/evaluation/query_gram_smoke/stratified_query_gram_pivot/statistics_smoke
```

```bash
/home/zhangal/.conda/envs/basis/bin/python -u evaluation/select_query_positions.py --candidate-capture results/calibration/query_gram_smoke --output-dir results/evaluation/query_gram_smoke/full_uniform_q32 --policy full_uniform_q32 --device cpu
```

```bash
/home/zhangal/.conda/envs/basis/bin/python -u evaluation/fit_qwen3_8b_q8_fisher_residual.py --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --c1-checkpoint results/checkpoints/qwen3_8b_c1_v80_32f4h_s32768_als6 --initial-bank results/checkpoints/q8_residual_kl_bank --base-kind closed_form_rrr --query-capture results/calibration/query_gram_smoke --query-position-manifest results/evaluation/query_gram_smoke/full_uniform_q32/positions.json --sampler-smoke --smoke-layer 15 --smoke-group 0 --num-shards 1 --output-dir results/evaluation/query_gram_smoke/full_uniform_q32/statistics_smoke
```

```bash
/home/zhangal/.conda/envs/basis/bin/python -u evaluation/select_query_positions.py --candidate-capture results/calibration/query_gram_smoke --output-dir results/evaluation/query_gram_smoke/terminal_uniform_q32 --policy terminal_uniform_q32 --device cpu
```

```bash
/home/zhangal/.conda/envs/basis/bin/python -u evaluation/fit_qwen3_8b_q8_fisher_residual.py --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --c1-checkpoint results/checkpoints/qwen3_8b_c1_v80_32f4h_s32768_als6 --initial-bank results/checkpoints/q8_residual_kl_bank --base-kind closed_form_rrr --query-capture results/calibration/query_gram_smoke --query-position-manifest results/evaluation/query_gram_smoke/terminal_uniform_q32/positions.json --sampler-smoke --smoke-layer 15 --smoke-group 0 --num-shards 1 --output-dir results/evaluation/query_gram_smoke/terminal_uniform_q32/statistics_smoke
```

