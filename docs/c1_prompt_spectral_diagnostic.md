# Closed-form prompt-specific residual diagnostic

## Scope

Qwen3-8B-Base, fixed C1-V96 and Base16; prompt-specific R8. Reuse the three hard prompts (160, 128, 161), layers 15/19/24/33, and the 256 fit / 32 diagnostic query positions of the previous prompt-subspace experiment. Page32/B2048, identical group-max selection and pinned prefix. This is a selected hard-case diagnostic, not generation accuracy or an unbiased benchmark.

## Fit

For each prompt, layer and KV group, let R be the post-RoPE exact-Key minus Base-Key residual. Compute uncentered moments C = R^T R / T and H = Q_fit^T Q_fit / N, pooling the four GQA query heads. Regularize the query moment by 1e-5 times its mean diagonal. If V contains the top-eight eigenvectors of H^(1/2) C H^(1/2), use E = H^(1/2) V and U = H^(-1/2) V. Store token codes R E; residual scores are (q U)(R E)^T. E and U are shared across the four query heads.

This minimizes the shared-query-metric residual projection objective. It does not minimize causal per-position Page-Fisher or the page-selection loss. The diagnostic computes teacher Page-Fisher statistics for reporting only; they are not spectral fitting inputs. No PCG, BCD, Adam, or autograd optimizer is used for fitting.

Prefill uses FP16 C1 full-Key memory-efficient attention on V100. Routing, moments and eigensolves use FP32. Fit queries come only from the current completed prompt, with diagnostic positions excluded from the query moment. All prompt residual-Key rows are used. This is not a streaming-causal calibration protocol.

## Jobs and outputs

Environment: basis. Entry point: evaluation/diagnose_c1_prompt_spectral.py, stages smoke/evaluate/summarize. Jobs: 8301242 smoke; 8301243 array 0–2; summary is dependency-gated. Outputs: results/evaluation/c1_prompt_spectral. Logs: spec_smoke_*, spec_fit_* and spec_sum_* at the repository root. Existing PCG results are retained under results/evaluation/c1_prompt_subspace.

## Deployment accounting

These jobs measure selection quality, not serving throughput. Covariance and eigensolve timings exclude feature construction, cache re-encoding and model prefill. A deployed version must encode the prompt cache once after determining E, then freeze E/U during decode. No repeated online solve is required during decode.

Each request needs its own E/U, so batching requires per-request batched/grouped matrix products. If all 36 layers use shared rank-eight factors, FP32 E/U occupy 2.25 MiB per request, or 144 MiB for batch 64; the existing eight-dimensional token sidecar does not grow. Two FP32 128-by-128 moments per KV group require 1 MiB per request per layer and can be constructed layer by layer. Keeping all layers' moments simultaneously would use 36 MiB per request. Exact residual data need not be retained across all layers, but computing the final token codes requires another pass after the basis is known.

Large batches may amortize launches, but do not eliminate per-request covariance construction, cache encoding, full-length sidecar scans, selection, or shared PCIe bandwidth limits. No large-batch speedup is established by this diagnostic.

## Completed results

All jobs completed successfully: smoke 8301242 (28 seconds), three workers 8301243_0/1/2 (39/40/39 seconds), summary 8301246 (15 seconds). Two CPU mathematical tests passed, including equality of the weighted projection error and discarded eigenvalue sum. Prefill smoke relative L2 error against FP32 reference was 0.000235466.

The previous PCG and new spectral runs have identical sample records, fit/diagnostic positions, offline aggregate metrics and exact aggregate metrics. Means below cover 96 prompt/layer/group combinations.

| Method | Teacher mass (%) | Non-sink mass (%) | Page overlap (%) | Diagnostic Fisher loss |
|---|---:|---:|---:|---:|
| Offline R8 | 89.9693 | 87.3533 | 75.9374 | 0.721584 |
| PCG prompt-U | 89.8077 | 87.1619 | 73.9182 | 0.812851 |
| PCG prompt-E/U | 91.2694 | 88.9641 | 79.9606 | 0.506577 |
| Closed-form prompt-E/U | 91.4398 | 89.1799 | 82.6604 | 0.762213 |
| Same-budget exact-QK | 92.7527 | 90.8107 | 100.0000 | 0 |

Closed-form versus offline: +1.4704 percentage points teacher mass, +6.7230 points overlap. Versus PCG prompt-E/U: +0.1704 points teacher mass, +2.6998 points overlap. Fisher loss increases versus offline despite improved mean selection metrics. These are diagnostic observations, not evidence of increased generation accuracy.

| Prompt | Offline mass (%) | Closed-form mass (%) | Exact mass (%) | Moment construction (s) | Spectral fit (s) |
|---|---:|---:|---:|---:|---:|
| 160 | 88.1222 | 90.1173 | 91.6784 | 0.00525 | 0.17509 |
| 128 | 90.9530 | 91.9154 | 93.0976 | 0.00747 | 0.17996 |
| 161 | 90.8328 | 92.2865 | 93.4821 | 0.00518 | 0.18507 |

Timings sum across four layers and eight groups per prompt, with GPU synchronization. They exclude feature construction and sidecar re-encoding, and are not complete prefill overhead. Diagnostic workers process one prompt each; no large-batch deployment benchmark was run.

Command in the basis environment (stage smoke, then evaluate with shard-index 0/1/2, then summarize):

```bash
/home/zhangal/.conda/envs/basis/bin/python evaluation/diagnose_c1_prompt_spectral.py --model /deac/csc/yangGrp/zhangal/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4 --stage evaluate --shard-index 0
```
