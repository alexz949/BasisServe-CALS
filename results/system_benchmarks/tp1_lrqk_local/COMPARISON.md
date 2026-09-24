# LRQK GPU-local Comparison

15 formal trials: 3 complete, 12 GPU OOM. Record/source checks passed.
Three-cohort medians; 128 output tokens. Decode is the request tail wall mean, including greedy selection.

| Context | CPU-offload request s | GPU-local request s | CPU-offload decode ms | GPU-local decode ms | GPU-local build s |
|---:|---:|---:|---:|---:|---:|
| 32768 | 55.603 | 28.659 | 379.791 | 187.501 | 0.910 |
| 65536 | 70.996 | 0/3 complete | 401.399 | 0/3 complete | - |
| 130048 | 2/3 complete | 0/3 complete | 2/3 complete | 0/3 complete | - |

GPU-local changes only storage/placement and equivalent gather: exact K/V and routing state remain on GPU; fitting, rank, support and hit/miss policy remain unchanged.
70 synthetic decode steps matched CPU/GPU selected indices, selected K/V, routing factors and full cache exactly, including lite-buffer turnover. Full-model 8K smoke matched generated tokens.
Paired full-request generation checks: 3/3 matched all generated token IDs. Details are in verification.json.
GPU-local retains the upstream 1.5x allocation policy and long-context fit intermediates; OOM is not an algorithmic capacity claim.
In this run, 64K failures occur at prefill MLP temporary allocation; approximately 128K failures occur inside online fitting. Both happen during warmup, before primary request timing. No MLP or allocation-policy optimization was introduced.
This storage control does not remove LRQK's prompt-specific fit or its decode-time factor updates.
No quality equivalence is claimed from short correctness checks. Compiler warnings remain in logs; absence during primary timing is not proof that every call was compilation-free.
See [SUMMARY.md](SUMMARY.md) for all configurations, reproduction command and failure phases.

Raw artifacts: [HF GPU-local supplement](https://huggingface.co/alexz949/BasisServe-CALS/tree/main/system_benchmarks/tp1_lrqk_local), immutable raw-data revision `86c03092c663dc6654584143130b5e5abf2fedaa`. Retrieval and shared-dependency instructions are linked from SUMMARY.md.
