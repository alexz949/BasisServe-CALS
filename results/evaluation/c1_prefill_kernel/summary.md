# C1 prefill kernel numerical diagnosis

Eight fixed LongBench prompts,36 layers each; uniform C1-V80. Read-only diagnosis: production kernel and C1 factors unchanged. No benchmark accuracy run.

| Sample | Task | Tokens | Triton top1 | Reference-C1 top1 | Dense top1 | KL(dense,Triton) | KL(dense,reference) |
|---|---|---:|---:|---:|---:|---:|---:|
| 0 | qasper | 5984 | 650 | 650 | 650 | 0.248822 | 0.26557 |
| 32 | multifieldqa_en | 4948 | 220 | 220 | 6527 | 0.270679 | 0.319005 |
| 64 | hotpotqa | 9856 | 576 | 576 | 37007 | 1.35218 | 1.28714 |
| 96 | 2wikimqa | 7647 | 4657 | 4657 | 18880 | 0.931075 | 0.925254 |
| 128 | gov_report | 13572 | 576 | 576 | 576 | 0.229313 | 0.236787 |
| 160 | qmsum | 12118 | 576 | 576 | 576 | 0.244808 | 0.25643 |
| 119 | 2wikimqa | 1192 | 2308 | 2308 | 2308 | 0.594357 | 0.519652 |
| 175 | qmsum | 30431 | 576 | 576 | 576 | 0.145175 | 0.162261 |

Maximum relative L2 errors over288 layer/input pairs:

```json
{
  "triton_vs_flash_all": 0.0018410613993182778,
  "triton_vs_fp32_sampled": 0.001954043284058571,
  "flash_vs_fp32_sampled": 0.0018514328403398395,
  "decoded_triton_vs_fp32": 0.002532375045120716,
  "decoded_flash_vs_fp32": 0.002506941556930542
}
```

Triton/reference-C1 terminal top1 disagreements: 0/8.
Both full C1 trajectories use identical factors and differ only in the prefill attention kernel. The local three-way comparisons use exactly the same Q/K/V tensors on the original Triton trajectory.
Flash reference pads only the Value feature dimension with zeros; Q/K scale and causal mask remain unchanged. Independent sampled-query reference uses explicit FP32 operations with TF32 disabled. Decoded-output errors use the same C1 output projection in FP32.
Errors and logits diagnose numerical behavior, not downstream task accuracy or all possible inputs. No generation, refitting, kernel modification, or RULER rerun is included.

## Additional checks and interpretation

All 288 layer/input comparisons were finite. Across 6,624 sampled layer/query positions, the largest single-row Triton/FP32 relative L2 was0.003004 (0.3004%), at sample96, layer34, query255. Each row metric aggregates all query heads. No large error was observed at the sampled causal or tail-block boundaries.

Mean final-position KL(dense,Triton-C1):0.502051. Mean KL(dense,Flash-reference-C1):0.496512. Mean KL(Triton-C1,Flash-reference-C1):0.003239. Replacing the prefill kernel did not remove the large C1-versus-dense distribution difference in these cases. All eight C1 top1 tokens were unchanged; the three C1/dense top1 disagreements persisted.

These results do not support an obvious Triton-specific numerical implementation error as the main explanation of the observed LongBench prefill-path gap. They do not establish that every input is safe or that downstream generated answers are identical. No reference-kernel generation scores were measured.

Probe workers completed in24,23,20 and34 seconds; CPU summary took13 seconds. All jobs exited0 without retries. Environment:basis; four L40S; production kernel and C1 checkpoint unchanged. Commands and setup are in `docs/c1_prefill_kernel_diagnosis.md`. Final result and runtime-source hashes were independently checked; see `audit.json`.
