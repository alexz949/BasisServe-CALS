# Qwen3-8B Wo-only uniform-rank Pareto sweep

All headline comparisons use uniform source ranks, identical C4 calibration covariances, dense V/KV cache, and equal ideal ring bytes between private C1-AllGather and strong LR-AllReduce.

| C1 rank | Retained | Wire LR rank | Bytes/row/rank | Joint C1 PPL | Local C1 PPL | Wire LR PPL | Capacity LR PPL |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 50.0% | 1024 | 3072 | 7.28194919 | 7.25140044 | 7.54659120 | 7.03970536 |
| 640 | 62.5% | 1280 | 3840 | 7.12059246 | 7.11920026 | 7.22780202 | 7.01337058 |
| 768 | 75.0% | 1536 | 4608 | 7.04818263 | 7.04413056 | 7.11532689 | 7.00165584 |
| 896 | 87.5% | 1792 | 5376 | 7.00946826 | 7.01022766 | 7.07494976 | 7.00284462 |
| 1024 | 100.0% | 2048 | 6144 | 7.00202554 | 7.00261369 | 7.03970536 | 7.00235694 |

| C1 rank | Joint C1 MSE | Local C1 MSE | Wire LR MSE | Capacity LR MSE | C1 PPL advantage vs wire LR |
|---:|---:|---:|---:|---:|---:|
| 512 | 0.045990029 | 0.048528799 | 0.086512558 | 0.016642956 | +3.634% |
| 640 | 0.025244215 | 0.026363483 | 0.058503875 | 0.0060047525 | +1.506% |
| 768 | 0.012129333 | 0.012540113 | 0.03927722 | 0.0015595101 | +0.953% |
| 896 | 0.0041487073 | 0.0042424393 | 0.025915276 | 0.00017761794 | +0.934% |
| 1024 | 0 | 7.2012364e-06 | 0.016642956 | 5.2045704e-06 | +0.538% |

Dense PPL is rerun inside every rank job. Its observed range is `7.00202554`–`7.00202554`.

## Command

```bash
/home/zhangal/.conda/envs/basis/bin/python evaluation/summarize_qwen3_8b_wo_c1_uniform_rank_sweep.py --quality-results results/evaluation/qwen3_8b_wo_c1_uniform_r512_quality/results.json results/evaluation/qwen3_8b_wo_c1_uniform_r640_quality/results.json results/evaluation/qwen3_8b_wo_c1_uniform_r768_quality/results.json results/evaluation/qwen3_8b_wo_c1_uniform_r896_quality/results.json results/evaluation/qwen3_8b_wo_c1_uniform_r1024_quality/results.json --expected-ranks 512,640,768,896,1024 --output-dir results/evaluation/qwen3_8b_wo_c1_uniform_rank_sweep
```
