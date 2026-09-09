# C1-V80 Routing Comparison: Our Base16+R8 Router, Loki, and QUEST

Date: September 5, 2026.

This report consolidates two completed RULER 32K experiments on Qwen3-8B-Base. All methods use the same frozen C1-V80 Value payload. The experiments differ in their sparse-layer policies and selection-budget semantics; those differences are recorded explicitly below.

## 1. Aggregate results

Scores are task-balanced means, expressed as percentages. They include task-specific partial-credit scoring and are not uniformly binary exact-match accuracies.

| Experiment | Method | Full-attention layers | Mean score | Mean selected physical tokens per sparse-layer KV group |
|---|---|---|---:|---:|
| A and B | Full exact K + C1-V80 | All 36 | 85.2083% | Full context |
| A | Our Base16 + Fisher R8 | None | 80.4356% | 2033.688 |
| A | QUEST-page, shared-GQA adaptation | None | 34.4886% | 2045.969 |
| A | Loki-page R32, page adaptation | None | 57.6136% | 2038.433 |
| A | Loki-token R32, per-query-head selection | None | 79.9053% | 3865.413 |
| B | Our Base16 + Fisher R8 | Layers 0–1 | 80.7197% | 2033.744 |
| B | Official QUEST accuracy forward, C1 bridge | Layers 0–1 | 50.7386% | 3579.526 |

Layer indices are zero-based. “Full attention” means attending to the entire cached sequence; it does **not** mean replacing C1-V80 with dense V128. No dense-V128 baseline was run in these two experiments.

The Loki variants were evaluated only with all 36 layers sparse. There is no measured Loki result with layers 0–1 full in this report.

## 2. Shared evaluation configuration

| Setting | Value |
|---|---|
| Model | Qwen3-8B-Base |
| Layers | 36 |
| Query / KV heads | 32 / 8; four query heads per GQA group |
| Key head dimension | 128 |
| Value payload | Frozen C1-V80, ALS6 checkpoint |
| Precision | BF16 |
| Context limit | 32,768 tokens, including the task's generation allowance |
| Dataset | Existing RULER 32K pilot: 11 tasks × 8 prompts = 88 prompts |
| Prompt formatting | Base-model input plus answer prefix; no chat template |
| Prefill | Full C1 attention with the existing Triton backend |
| First generated token | Shared full-prefill argmax within each experiment |
| Subsequent generation | Greedy, with task-specific output caps and EOS handling |
| Cache handling | Independent decode forks from an immutable prefix |
| Hardware / environment | Four NVIDIA L40S GPUs; `basis` conda environment |
| New fitting during these comparisons | None |

The same 88 prompts were used in both experiments. The full exact-K control produced identical generated token IDs on every prompt across the two runs.

This is a reused pilot set, not a newly reserved final held-out set and not the complete RULER suite. Actual prompt lengths vary below the 32,768-token limit.

## 3. Routing algorithms and checkpoints

### 3.1 Our Base16 + Page-Fisher R8 router

The resident C1-V80 latent predicts a pre-RoPE Key component through a rank-16 Base. RoPE is applied to this component, and a rank-8 post-RoPE residual channel supplies additional routing information. The resulting proxy scores are aggregated using page log-sum-exp, normalized over non-sink pages separately for each query head, and combined by a maximum across the four query heads sharing a KV group.

The router selects 64 pages of 32 tokens, including pinned page0. Attention is then evaluated using exact selected Keys and resident C1-V80. The approximate Key representation is used for routing, not as the final attention Key payload.

The reused checkpoint records:

- 64 C4 fit windows × 32,768 tokens, with 16 additional diagnostic validation windows.
- Base16 fitted with a per-query/per-position causal non-sink raw-QK squared-error objective, using eight query positions per window.
- The Base frozen during the subsequent residual fit.
- R8 fitted with separate causal exact-teacher non-sink Page-Fisher losses, using 16 query positions per window.
- A fixed final residual sweep, with validation used for diagnostics rather than checkpoint selection.

“Q8 Base” and “Q16 residual” denote the number of sampled calibration query positions per window. They do not denote query quantization or compressed query dimensions. The query positions lie in the terminal 8K region of each 32K window, not across all query rows.

Experiment A sparsifies all 36 layers. Experiment B uses exactly the same factors but keeps layers 0–1 full.

### 3.2 Loki R32

Both Loki variants reuse a per-layer, per-KV-head rank-32 Key-PCA checkpoint. PCA was fitted on the same first 64 C4 32K windows used by the router, using centered Keys after Key RMSNorm and before RoPE. At inference, the stored basis is applied to post-RoPE Q/K to produce approximate routing scores. Final attention uses exact selected QK and C1-V80.

Two selection variants were evaluated:

- **Loki-token R32:** independently select the top 2048 tokens for every query head, without a pinned prefix. This retains Loki's token-selection structure in the repository's C1 implementation.
- **Loki-page R32:** use the PCA proxy with our normalized page-LSE/GQA-max selector, choosing 64 Page32 pages per KV group, including page0. This is a page adaptation, not the original token selector.

Neither variant was evaluated with two full layers. These runs are C1-payload evaluations, not complete reproductions of Loki's original model and benchmark settings.

### 3.3 QUEST: shared-GQA adaptation versus official forward

The shared-GQA QUEST adaptation in Experiment A uses exact post-RoPE page minima and maxima to compute query-dependent page upper bounds. It takes the maximum raw bound across query heads in each GQA group, then selects 64 shared Page32 pages including page0. All 36 layers are sparse.

Experiment B instead imports and directly calls the accuracy `forward` from the [official QUEST source](https://github.com/mit-han-lab/Quest/blob/01c1623bf9395009520874e989e29f683203b357/evaluation/quest_attention.py), at commit `01c1623bf9395009520874e989e29f683203b357`:

- Layers 0–1 remain full; layers 2–35 use QUEST.
- Page size is 32, retaining the previous comparison's setting. The official passkey shell example uses 16, which was not used here.
- Each query head independently selects pages for a 2048-token budget.
- No pinned prefix or shared-GQA max selection is added.
- The physical union across a GQA group's query heads is measured, not constrained to 2048 tokens.

The official model installer recognizes Llama/Mistral, not Qwen3. A model-interface bridge therefore preserves the existing Qwen3 Q/K normalization, RoPE, cache, and C1 decoder while invoking the upstream attention function. Already-processed Q/K pass through identity projections and identity rotary embeddings; the removed Transformers rotary-helper argument is adapted at the call interface. C1-V80 is zero-padded to width 128 for the upstream equal-width interface, and the output is sliced back to 80 before C1 decoding.

The upstream source file remains byte-identical to the pinned commit. Its bounds, page selection, mask, softmax, and weighted-Value computation execute directly. This is an official-QUEST-forward evaluation with a Qwen3/C1 bridge, not original-paper model accuracy.

## 4. Experiment A: all 36 layers sparse

All entries below use the same C1-V80 payload. “Exact K” is the full-attention reference.

| Task | Exact K | Our Base16+R8 | QUEST shared-page | Loki-page R32 | Loki-token R32 |
|---|---:|---:|---:|---:|---:|
| niah_single_1 | 100.0000% | 100.0000% | 100.0000% | 100.0000% | 100.0000% |
| niah_single_2 | 100.0000% | 100.0000% | 50.0000% | 87.5000% | 100.0000% |
| niah_single_3 | 100.0000% | 100.0000% | 0.0000% | 25.0000% | 100.0000% |
| niah_multikey_1 | 87.5000% | 87.5000% | 25.0000% | 75.0000% | 87.5000% |
| niah_multikey_2 | 87.5000% | 50.0000% | 0.0000% | 37.5000% | 50.0000% |
| niah_multiquery | 96.8750% | 96.8750% | 9.3750% | 31.2500% | 81.2500% |
| niah_multivalue | 93.7500% | 93.7500% | 12.5000% | 25.0000% | 96.8750% |
| vt | 92.5000% | 90.0000% | 70.0000% | 90.0000% | 92.5000% |
| fwe | 91.6667% | 79.1667% | 12.5000% | 75.0000% | 83.3333% |
| qa_1 | 50.0000% | 50.0000% | 50.0000% | 50.0000% | 50.0000% |
| qa_2 | 37.5000% | 37.5000% | 50.0000% | 37.5000% | 37.5000% |
| Task-balanced mean | 85.2083% | 80.4356% | 34.4886% | 57.6136% | 79.9053% |

The three page-based arms share the same maximum physical selection budget of 2048 tokens per KV group. Loki-token instead has a 2048-token budget per query head.

## 5. Experiment B: layers 0–1 full

| Task | Exact K | Our Base16+R8, full2 | Official QUEST forward, full2 |
|---|---:|---:|---:|
| niah_single_1 | 100.0000% | 100.0000% | 100.0000% |
| niah_single_2 | 100.0000% | 100.0000% | 75.0000% |
| niah_single_3 | 100.0000% | 100.0000% | 12.5000% |
| niah_multikey_1 | 87.5000% | 87.5000% | 37.5000% |
| niah_multikey_2 | 87.5000% | 50.0000% | 25.0000% |
| niah_multiquery | 96.8750% | 100.0000% | 37.5000% |
| niah_multivalue | 93.7500% | 93.7500% | 28.1250% |
| vt | 92.5000% | 90.0000% | 92.5000% |
| fwe | 91.6667% | 79.1667% | 75.0000% |
| qa_1 | 50.0000% | 50.0000% | 37.5000% |
| qa_2 | 37.5000% | 37.5000% | 37.5000% |
| Task-balanced mean | 85.2083% | 80.7197% | 50.7386% |

The layer policy is matched between the two sparse arms, but the physical selection budget is not: our budget is shared by the KV group, whereas QUEST's is per query head.

## 6. Selection counts and memory accounting

“Physical tokens” means the number of unique valid token IDs in a KV group's selected support, not measured hardware loads or PCIe transfers. A per-query-head budget can produce up to 8192 unique tokens when four heads select disjoint sets. The reported means are weighted by decode steps and KV groups along each arm's own generated trajectory. For Experiment B, the two full layers are excluded.

The page-based means fall slightly below 2048 because a selected final page can be partially filled.

Experiment A separately recorded the following peak persistent routing-cache sizes:

| Method | Materialized routing cache | Peak cache size, MiB |
|---|---|---:|
| Our Base16+R8 | Expanded Base128 plus R8 per token | 2447.851 |
| QUEST shared-page | Per-page Key minima and maxima | 144.000 |
| Loki-page R32 | R32 per-token sidecar | 575.965 |
| Loki-token R32 | R32 per-token sidecar | 575.965 |

These figures exclude fixed factor matrices and transient scoring, selection, gather, and attention buffers. They are not total model memory. In particular, the current accuracy oracle materializes Base128; its storage must not be described as residual-only R8 storage.

No equivalent persistent routing-cache measurement was reported for the official QUEST forward in Experiment B. That accuracy implementation recomputes bounds and calculates full exact QK before masking. It is not the official optimized CUDA-kernel execution path.

All exact Keys remain on GPU in both experiments. Neither experiment establishes an offload speedup, a PCIe traffic reduction, or a kernel-latency advantage. No attention-mass recall or NLL/PPL measurements were collected in these runs.

## 7. Execution and validation

| Run | Slurm job | Devices | Recorded job duration | Maximum per-worker PyTorch allocation |
|---|---|---|---|---:|
| Experiment A: evaluation + summary | 8300534 | 4 × L40S | 11 min 49 sec | 25.195785 GiB |
| Experiment B: smoke + evaluation + summary | 8300540 | 4 × L40S | 11 min 45 sec | 33.663857 GiB |

The durations cover different sets of work and are not per-method latency comparisons. Both jobs completed with exit code `0:0` in the `basis` environment.

Validation records include:

- Eight completed samples per task, with 88 unique prompt keys in each experiment.
- Matching protocol and artifact/source identities within each run.
- Exact generated-ID agreement for the full exact-K control across all 88 prompts between experiments.
- Exact agreement between saved prediction strings and token-ID decoding: 440 outputs in Experiment A and 264 in Experiment B.
- GPU smoke replay with identical logits and token IDs from the same immutable prefix.
- CPU checks against independent support/attention references and tiny-model cache/full-support checks.
- Finite logits, correct cache lengths, and unchanged prefixes during evaluation.
- For every official QUEST sample, zero upstream calls in layers 0–1 and the expected decode-step count in each of layers 2–35.

An initial Experiment A smoke attempt, job 8300528, failed during prefill because unrelated processes occupied the selected GPU. It produced no scored result. The subsequent smoke and both formal comparisons completed without OOM or runtime errors.

## 8. Comparison boundaries

- The 34.4886% QUEST shared-page score is **not** the official QUEST-forward score. The latter is 50.7386% under Experiment B's Qwen3/C1 settings.
- The change between those QUEST results combines changes in layer policy, GQA aggregation, budget sharing, prefix pinning, and implementation. It is not an isolated first-two-layer ablation.
- Loki-page is a page adaptation; Loki-token is the separate token-selection reference.
- Loki-token and official QUEST do not have the same physical budget as our router.
- Loki's all-layer-sparse result and official QUEST's full2 result do not share the same layer policy.
- Eight examples per task do not establish small score differences reliably. These results do not isolate the cause of individual errors or establish universal rankings of the original methods.

## 9. Artifacts and implementation

| Component | Artifact |
|---|---|
| C1-V80 checkpoint | `results/checkpoints/qwen3_8b_c1_v80_32f4h_s32768_als6` |
| Our Base16/R8 checkpoint | `results/checkpoints/q8_qbase_fisher16_r8` |
| Loki R32 checkpoint | `results/checkpoints/qwen3_8b_loki_r32_c4_64f_s32768` |
| Shared RULER prompts | `results/datasets/qwen3_8b_base_ruler_v1_32k_shadowkv11_s8` |
| Experiment A data | [result.json](../results/evaluation/routing_baselines_ruler32k/result.json) |
| Experiment A per-task report | [summary.md](../results/evaluation/routing_baselines_ruler32k/summary.md) |
| Experiment B data | [result.json](../results/evaluation/quest_official_ruler32k/result.json) |
| Experiment B per-task report | [summary.md](../results/evaluation/quest_official_ruler32k/summary.md) |
| Experiment A evaluator | [eval_qwen3_8b_routing_baselines_ruler.py](../evaluation/eval_qwen3_8b_routing_baselines_ruler.py) |
| Experiment B evaluator | [eval_qwen3_8b_official_quest_ruler.py](../evaluation/eval_qwen3_8b_official_quest_ruler.py) |
| Shared-GQA/Loki adapters | [c1_routing_comparison.py](../basisserve/core/c1_routing_comparison.py) |
| Official QUEST bridge | [c1_official_quest.py](../basisserve/core/c1_official_quest.py) |

Exact execution commands and detailed audit records are preserved in the [Experiment A protocol](routing_baselines_ruler_protocol.md) and [Experiment B protocol](quest_official_ruler_protocol.md). This summary consolidates existing results; no new GPU evaluation was launched to produce it.
