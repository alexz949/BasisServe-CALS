# Results and kernel settings — 2026-09-15

This summary describes existing runs. This upload contains Markdown reports and code only; raw logs, predictions, tensors, plots and profiler traces remain on the cluster. No experiment was rerun for this upload.

## Current native comparison and kernel settings

The native comparison uses **Llama-3.1-8B base**, one **L40S**, batch 1, Dense V128 and original Wo. It is separate from the earlier TP4/V96 grid and from Instruct quality evaluations.

| Setting | Basis K offload | Native ShadowKV |
|---|---|---|
| Nominal budget | Hard 2048, including sink32 and recent64 | 2048 plus native outlier/local/generated support |
| Selection | Page32; GQA max aggregation | Chunk8 landmarks |
| Routing | B16R16, or first 8 original Base/Residual dimensions without refitting | Rank160 |
| CPU data | Exact K | Full V |
| GPU data | Full V, routing codes, selected K when reuse enabled | Low-rank K state, reconstructed/cached selected KV |

B8R8 retains the original residual definition: the full fitted Base16 prediction is used when constructing residual codes, then the original codes are truncated. It is not a newly fitted rank-8 model.

The fused Base router reconstructs a token's approximate 128D K in shared memory, applies rounding/bias/RoPE and query dot products, adds the residual score, and reduces Page32 scores with log-sum-exp. It does not materialize full token scores or reconstructed K in HBM. Base ranks below 16 still use padded WMMA work; scan cost cannot be inferred solely from the nominal rank.

The latest persistent-slot implementation stores one 2048-token GPU K array per KV head/layer. Hits retain their physical slots; only misses read CPU K. Triton attention uses independent K-slot and original V-token indices, with 16 splits, block32, four warps and FP32 partial accumulation. There is no compact V copy. The reload control uses the same planner and attention with hits disabled.

Main sources:

- `basisserve/kernels/csrc/conditional_router_page32.cu`
- `basisserve/kernels/csrc/mapped_host_paged_attention.cu`
- `basisserve/kernels/csrc/selected_key_reuse.cu`
- `basisserve/kernels/csrc/persistent_key_slots.cu`
- `basisserve/kernels/slot_indexed_attention.py`
- `benchmarks/system/native_basis_cache.py`
- `benchmarks/system/bench_persistent_slots.py`

Native ShadowKV source revision: `e51904cdeab7d4d34013370f09f2cf5fcd655e15`. Import adaptations and dependency versions are preserved with its records. Native cache/CUDA behavior was retained; this is distinct from the older adapted ShadowKV quality-evaluation path. Neither method should be described as keeping all KV exclusively on CPU.

## Timing interpretation

`native_wall_ms_step` covers the native 100-step decode loop, including sampling and host dependencies; prefill and initial cache placement are separate. CUDA timing columns cover inference intervals. The median after the first 10 steps is a steady-step statistic, not end-to-end request latency. First-step allocation/JIT and occasional long spikes materially affect averages. Smoke and instrumented profile timings are not production throughput measurements.

Representative steady CUDA medians (ms/step; 90 steps after discarding 10):

| Context | Router | Original CPU K reads | Compact reuse | Persistent slots reuse |
|---|---|---:|---:|---:|
| 64K | B16R16 | 39.92 | 38.79 | 37.30 |
| 64K | B8R8 | 39.27 | 37.97 | 36.60 |
| 128K | B16R16 | 50.58 | 49.23 | 47.93 |
| 128K | B8R8 | 49.07 | 48.20 | 46.77 |

Persistent slots save approximately 245.59 MiB of extra cache/workspace versus the compact double-buffer implementation. This is not a percentage of total model memory. For 64K B16R16, persistent-slot reuse had a 40.48 ms mean after 10 steps despite the 37.30 ms median, due to spikes. Whole-loop costs remain in the local per-case result JSON files.

The stage profile found approximately 10.54 ms of Base16 routing per 32-layer decode step and 5.44 ms of original K fetch plus attention. Reducing K transfer does not remove routing, model projections/MLP or orchestration costs. Internal `clock64` fractions include stalls/barriers and are not an additive GPU wall-time breakdown. ShadowKV transfer and reconstruction can overlap.

Some matched runs shared GPU memory with an existing approximately 6 GiB process and a validation context. Their OOM points are observed conditions, not an exclusive-card capacity limit. The earlier TP4 128K extension is incomplete: dense and C1 completed, sparse-local hit the then-existing page-selector limit, and offload only had a smoke result. The later selector fix does not retroactively complete that grid.

Validation records include 14 synthetic persistent-slot cases and four 8K full-model smoke configurations. Selected K matched bitwise; attention comparison used 0.003 tolerance and finite-output checks. No new performance or quality evaluation was run for this upload.

## Historical Dense-V RULER overview

These rows use 11 tasks × 8 examples = 88 examples; they are separate from other result sets already on the remote branch. Scores are percentages.

| Model | Context | Nominal budget | Dense | B16R16 | LRQK | ShadowKV |
|---|---|---:|---:|---:|---:|---:|
| Llama-3.1-8B base | 64K | 2048 | 85.91 | 83.16 | 84.92 | 81.25 |
| Llama-3.1-8B-Instruct | 64K | 2048 | 85.80 | 85.66 | 84.13 | 86.08 |
| Llama-3.1-8B-Instruct | 128K | 2048 | 82.42 | 78.43 | 79.85 | 75.87 |
| Llama-3.1-8B-Instruct | 128K | 1024 | — | 79.75 | 79.62 | — |

Historical ShadowKV rows use the adapted evaluation path, not the native complete system benchmark. The base-model LRQK row uses the older adapter; Instruct LRQK uses official cache semantics with tolerance 0.01, BF16 state and FP32 solves. Budgets differ in effective support: ours includes pinned pages within the hard limit, whereas LRQK uses per-query-head top-k plus recent tokens. Consult raw protocols before interpreting cross-method differences.

Other local records include LongBench, diagnostic recall/output-error experiments, Base/Residual ablations, Loki calibration experiments, fitting and capacity tests. Their complete parameters, individual predictions where recorded, and failed-run logs remain in the local result directories and are not uploaded in this Markdown-and-code snapshot.

## Recorded launch commands

All experiments used the `basis` environment; native ShadowKV additionally used its recorded isolated dependency overlay. These commands were launched through Slurm.

- `shadow_native`: `python -m benchmarks.system.run_shadow_native`
- `shadow_matched`: `python -m benchmarks.system.run_shadow_matched`
- `cache_reuse`: `python -m benchmarks.system.run_cache_reuse`
- `persistent_slots`: `python -m benchmarks.system.run_persistent_slots`

## Native dependency provenance

The native overlay used Transformers 4.43.1, tokenizers 0.19.1, Hugging Face Hub 0.23.3, FlashInfer 0.2.5 and MInference 0.1.6; the basis environment used PyTorch 2.6 with CUDA 12.4 and vLLM 0.8.5. Extension compilation used CUDA 12.3 for L40S architecture 8.9. Native import changes invoke installed vLLM operations directly, defer inactive MInference imports and bootstrap the Llama package; the native cache and CUDA implementations were retained.

## Separate external ShadowKV working-tree patch

The repository-adjacent `external/ShadowKV` checkout has this CUDA-stream patch against revision `e51904cdeab7d4d34013370f09f2cf5fcd655e15`. This checkout is distinct from the isolated native benchmark source described above.

```diff
diff --git a/kernels/batch_gemm_softmax.cu b/kernels/batch_gemm_softmax.cu
index 763431e..4a2dc68 100644
--- a/kernels/batch_gemm_softmax.cu
+++ b/kernels/batch_gemm_softmax.cu
@@ -19,6 +19,7 @@


 #include <torch/extension.h>
+#include <c10/cuda/CUDAStream.h>

 #include <assert.h>
 #include <cuda_runtime.h>
@@ -193,5 +194,5 @@ void batch_gemm_softmax(

     CUTLASS_CHECK(batch_gemm_softmax.initialize(args));

-    CUTLASS_CHECK(batch_gemm_softmax());
+    CUTLASS_CHECK(batch_gemm_softmax(c10::cuda::getCurrentCUDAStream().stream()));
 }
\ No newline at end of file
```
