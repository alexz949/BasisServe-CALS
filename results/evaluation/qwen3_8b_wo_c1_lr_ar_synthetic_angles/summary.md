# C1 controlled source-subspace angles

This deterministic CPU/FP64 experiment is a scaled Qwen3-8B TP4 representation sanity check. It isolates source output-subspace angle from calibration noise and model effects.

## Geometry and budget

- TP sources: `4`
- Per-source C1 rank: `4`
- Total C1 AllGather rank: `16`
- Equal-wire LR-AllReduce rank: `8`
- Fit and heldout rows: `128` each
- Work/factor dtype: `float64`
- Ideal ring traffic per rank per row: C1 `24` bytes, LR `24` bytes

## Results

| Target angle | Measured angle | Projection overlap | Normalized chordal d^2 | Union r95 | C1 heldout MSE | LR heldout MSE | Analytic LR MSE |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0° | 0.000000° | 1.000000 | 0.000000 | 4 | 1.20788e-30 | 3.51844e-31 | 0 |
| 15° | 15.000000° | 0.933013 | 0.066987 | 4 | 4.39998e-31 | 0.0170371 | 0.0170371 |
| 30° | 30.000000° | 0.750000 | 0.250000 | 11 | 5.47127e-31 | 0.0669873 | 0.0669873 |
| 45° | 45.000000° | 0.500000 | 0.500000 | 14 | 4.64763e-31 | 0.146447 | 0.146447 |
| 60° | 60.000000° | 0.250000 | 0.750000 | 15 | 4.0282e-31 | 0.25 | 0.25 |
| 75° | 75.000000° | 0.066987 | 0.933013 | 15 | 4.26026e-31 | 0.37059 | 0.37059 |
| 90° | 90.000000° | 0.000000 | 1.000000 | 16 | 5.33563e-31 | 0.5 | 0.5 |

## Validation

- Pearson diversity/C1-advantage correlation: `0.967344`
- Spearman diversity/C1-advantage correlation: `1.000000`
- `all_trials_passed`: `pass`
- `lr_error_monotonic_non_decreasing`: `pass`
- `shared_endpoint_exact`: `pass`
- `orthogonal_endpoint_is_half_error`: `pass`
- `diversity_advantage_spearman_positive`: `pass`

The synthetic result validates the conceptual prediction only. It does not establish that pretrained-model TP sources have diverse subspaces; that is the next real-model analysis.

## Command

```bash
/home/zhangal/.conda/envs/basis/bin/python evaluation/run_c1_synthetic_controlled_angles.py --tp-size 4 --source-width 16 --source-rank 4 --output-width 32 --rows 128 --angles 0,15,30,45,60,75,90 --seeds 20260828,20260829,20260830 --wire-dtype-bytes 2 --torch-num-threads 1 --output-dir results/evaluation/qwen3_8b_wo_c1_lr_ar_synthetic_angles
```
